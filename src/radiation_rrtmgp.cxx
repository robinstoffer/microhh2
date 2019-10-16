/*
 * MicroHH
 * Copyright (c) 2011-2019 Chiel van Heerwaarden
 * Copyright (c) 2011-2019 Thijs Heus
 * Copyright (c) 2014-2019 Bart van Stratum
 *
 * This file is part of MicroHH
 *
 * MicroHH is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * MicroHH is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * You should have received a copy of the GNU General Public License
 * along with MicroHH.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/algorithm/string.hpp>
#include <numeric>
#include <string>
#include <cmath>

#include "radiation_rrtmgp.h"
#include "master.h"
#include "grid.h"
#include "fields.h"
#include "thermo.h"
#include "input.h"
#include "netcdf_interface.h"
#include "stats.h"
#include "cross.h"
#include "constants.h"
#include "timeloop.h"

// RRTMGP headers.
#include "Array.h"
#include "Optical_props.h"
#include "Gas_optics.h"
#include "Gas_concs.h"
#include "Fluxes.h"
#include "Rte_lw.h"
#include "Rte_sw.h"
#include "Source_functions.h"
#include "Cloud_optics.h"

namespace
{
    std::vector<std::string> get_variable_string(
            const std::string& var_name,
            std::vector<int> i_count,
            Netcdf_handle& input_nc,
            const int string_len,
            bool trim=true)
    {
        // Multiply all elements in i_count.
        int total_count = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Add the string length as the rightmost dimension.
        i_count.push_back(string_len);

        // Multiply all elements in i_count.
        // int total_count_char = std::accumulate(i_count.begin(), i_count.end(), 1, std::multiplies<>());

        // Read the entire char array;
        std::vector<char> var_char;
        var_char = input_nc.get_variable<char>(var_name, i_count);

        std::vector<std::string> var;

        for (int n=0; n<total_count; ++n)
        {
            std::string s(var_char.begin()+n*string_len, var_char.begin()+(n+1)*string_len);
            if (trim)
                boost::trim(s);
            var.push_back(s);
        }

        return var;
    }

    template<typename TF>
    void load_gas_concs(
            Gas_concs<TF>& gas_concs, Netcdf_handle& input_nc, const std::string& dim_name)
    {
        const int n_lay = input_nc.get_dimension_size(dim_name);

        const std::vector<std::string> possible_gases = {
                "h2o", "co2" ,"o3", "n2o", "co", "ch4", "o2", "n2",
                "ccl4", "cfc11", "cfc12", "cfc22",
                "hfc143a", "hfc125", "hfc23", "hfc32", "hfc134a",
                "cf4", "no2" };

        for (const std::string& gas : possible_gases)
        {
            if (input_nc.variable_exists(gas))
            {
                gas_concs.set_vmr(gas,
                        Array<TF,1>(input_nc.get_variable<TF>(gas, {n_lay}), {n_lay}));
            }
        };
    }

    Gas_optics<double> load_and_init_gas_optics(
            Master& master,
            const Gas_concs<double>& gas_concs,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);

        // Read k-distribution information.
        int n_temps = coef_nc.get_dimension_size("temperature");
        int n_press = coef_nc.get_dimension_size("pressure");
        int n_absorbers = coef_nc.get_dimension_size("absorber");
        int n_char = coef_nc.get_dimension_size("string_len");
        int n_minorabsorbers = coef_nc.get_dimension_size("minor_absorber");
        int n_extabsorbers = coef_nc.get_dimension_size("absorber_ext");
        int n_mixingfracs = coef_nc.get_dimension_size("mixing_fraction");
        int n_layers = coef_nc.get_dimension_size("atmos_layer");
        int n_bnds = coef_nc.get_dimension_size("bnd");
        int n_gpts = coef_nc.get_dimension_size("gpt");
        int n_pairs = coef_nc.get_dimension_size("pair");
        int n_minor_absorber_intervals_lower = coef_nc.get_dimension_size("minor_absorber_intervals_lower");
        int n_minor_absorber_intervals_upper = coef_nc.get_dimension_size("minor_absorber_intervals_upper");
        int n_contributors_lower = coef_nc.get_dimension_size("contributors_lower");
        int n_contributors_upper = coef_nc.get_dimension_size("contributors_upper");

        // Read gas names.
        Array<std::string,1> gas_names(
                get_variable_string("gas_names", {n_absorbers}, coef_nc, n_char, true), {n_absorbers});

        Array<int,3> key_species(
                coef_nc.get_variable<int>("key_species", {n_bnds, n_layers, 2}),
                {2, n_layers, n_bnds});
        Array<double,2> band_lims(coef_nc.get_variable<double>("bnd_limits_wavenumber", {n_bnds, 2}), {2, n_bnds});
        Array<int,2> band2gpt(coef_nc.get_variable<int>("bnd_limits_gpt", {n_bnds, 2}), {2, n_bnds});
        Array<double,1> press_ref(coef_nc.get_variable<double>("press_ref", {n_press}), {n_press});
        Array<double,1> temp_ref(coef_nc.get_variable<double>("temp_ref", {n_temps}), {n_temps});

        double temp_ref_p = coef_nc.get_variable<double>("absorption_coefficient_ref_P");
        double temp_ref_t = coef_nc.get_variable<double>("absorption_coefficient_ref_T");
        double press_ref_trop = coef_nc.get_variable<double>("press_ref_trop");

        Array<double,3> kminor_lower(
                coef_nc.get_variable<double>("kminor_lower", {n_temps, n_mixingfracs, n_contributors_lower}),
                {n_contributors_lower, n_mixingfracs, n_temps});
        Array<double,3> kminor_upper(
                coef_nc.get_variable<double>("kminor_upper", {n_temps, n_mixingfracs, n_contributors_upper}),
                {n_contributors_upper, n_mixingfracs, n_temps});

        Array<std::string,1> gas_minor(get_variable_string("gas_minor", {n_minorabsorbers}, coef_nc, n_char),
                {n_minorabsorbers});

        Array<std::string,1> identifier_minor(
                get_variable_string("identifier_minor", {n_minorabsorbers}, coef_nc, n_char), {n_minorabsorbers});

        Array<std::string,1> minor_gases_lower(
                get_variable_string("minor_gases_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> minor_gases_upper(
                get_variable_string("minor_gases_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,2> minor_limits_gpt_lower(
                coef_nc.get_variable<int>("minor_limits_gpt_lower", {n_minor_absorber_intervals_lower, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_lower});
        Array<int,2> minor_limits_gpt_upper(
                coef_nc.get_variable<int>("minor_limits_gpt_upper", {n_minor_absorber_intervals_upper, n_pairs}),
                {n_pairs, n_minor_absorber_intervals_upper});

        Array<int,1> minor_scales_with_density_lower(
                coef_nc.get_variable<int>("minor_scales_with_density_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> minor_scales_with_density_upper(
                coef_nc.get_variable<int>("minor_scales_with_density_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<int,1> scale_by_complement_lower(
                coef_nc.get_variable<int>("scale_by_complement_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> scale_by_complement_upper(
                coef_nc.get_variable<int>("scale_by_complement_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<std::string,1> scaling_gas_lower(
                get_variable_string("scaling_gas_lower", {n_minor_absorber_intervals_lower}, coef_nc, n_char),
                {n_minor_absorber_intervals_lower});
        Array<std::string,1> scaling_gas_upper(
                get_variable_string("scaling_gas_upper", {n_minor_absorber_intervals_upper}, coef_nc, n_char),
                {n_minor_absorber_intervals_upper});

        Array<int,1> kminor_start_lower(
                coef_nc.get_variable<int>("kminor_start_lower", {n_minor_absorber_intervals_lower}),
                {n_minor_absorber_intervals_lower});
        Array<int,1> kminor_start_upper(
                coef_nc.get_variable<int>("kminor_start_upper", {n_minor_absorber_intervals_upper}),
                {n_minor_absorber_intervals_upper});

        Array<double,3> vmr_ref(
                coef_nc.get_variable<double>("vmr_ref", {n_temps, n_extabsorbers, n_layers}),
                {n_layers, n_extabsorbers, n_temps});

        Array<double,4> kmajor(
                coef_nc.get_variable<double>("kmajor", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                {n_gpts, n_mixingfracs, n_press+1, n_temps});

        // Keep the size at zero, if it does not exist.
        Array<double,3> rayl_lower;
        Array<double,3> rayl_upper;

        if (coef_nc.variable_exists("rayl_lower"))
        {
            rayl_lower.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_upper.set_dims({n_gpts, n_mixingfracs, n_temps});
            rayl_lower = coef_nc.get_variable<double>("rayl_lower", {n_temps, n_mixingfracs, n_gpts});
            rayl_upper = coef_nc.get_variable<double>("rayl_upper", {n_temps, n_mixingfracs, n_gpts});
        }

        // Is it really LW if so read these variables as well.
        if (coef_nc.variable_exists("totplnk"))
        {
            int n_internal_sourcetemps = coef_nc.get_dimension_size("temperature_Planck");

            Array<double,2> totplnk(
                    coef_nc.get_variable<double>( "totplnk", {n_bnds, n_internal_sourcetemps}),
                    {n_internal_sourcetemps, n_bnds});
            Array<double,4> planck_frac(
                    coef_nc.get_variable<double>("plank_fraction", {n_temps, n_press+1, n_mixingfracs, n_gpts}),
                    {n_gpts, n_mixingfracs, n_press+1, n_temps});

            // Construct the k-distribution.
            return Gas_optics<double>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    totplnk,
                    planck_frac,
                    rayl_lower,
                    rayl_upper);
        }
        else
        {
            Array<double,1> solar_src(
                    coef_nc.get_variable<double>("solar_source", {n_gpts}), {n_gpts});

            return Gas_optics<double>(
                    gas_concs,
                    gas_names,
                    key_species,
                    band2gpt,
                    band_lims,
                    press_ref,
                    press_ref_trop,
                    temp_ref,
                    temp_ref_p,
                    temp_ref_t,
                    vmr_ref,
                    kmajor,
                    kminor_lower,
                    kminor_upper,
                    gas_minor,
                    identifier_minor,
                    minor_gases_lower,
                    minor_gases_upper,
                    minor_limits_gpt_lower,
                    minor_limits_gpt_upper,
                    minor_scales_with_density_lower,
                    minor_scales_with_density_upper,
                    scaling_gas_lower,
                    scaling_gas_upper,
                    scale_by_complement_lower,
                    scale_by_complement_upper,
                    kminor_start_lower,
                    kminor_start_upper,
                    solar_src,
                    rayl_lower,
                    rayl_upper);
        }
        // End reading of k-distribution.
    }

    Cloud_optics<double> load_and_init_cloud_optics(
            Master& master,
            const std::string& coef_file)
    {
        // READ THE COEFFICIENTS FOR THE OPTICAL SOLVER.
        Netcdf_file coef_nc(master, coef_file, Netcdf_mode::Read);

        // Read look-up table coefficient dimensions
        int n_band     = coef_nc.get_dimension_size("nband");
        int n_rghice   = coef_nc.get_dimension_size("nrghice");
        int n_size_liq = coef_nc.get_dimension_size("nsize_liq");
        int n_size_ice = coef_nc.get_dimension_size("nsize_ice");

        Array<double,2> band_lims_wvn(coef_nc.get_variable<double>("bnd_limits_wavenumber", {n_band, 2}), {2, n_band});

        // Read look-up table constants.
        double radliq_lwr = coef_nc.get_variable<double>("radliq_lwr");
        double radliq_upr = coef_nc.get_variable<double>("radliq_upr");
        double radliq_fac = coef_nc.get_variable<double>("radliq_fac");

        double radice_lwr = coef_nc.get_variable<double>("radice_lwr");
        double radice_upr = coef_nc.get_variable<double>("radice_upr");
        double radice_fac = coef_nc.get_variable<double>("radice_fac");

        Array<double,2> lut_extliq(
                coef_nc.get_variable<double>("lut_extliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<double,2> lut_ssaliq(
                coef_nc.get_variable<double>("lut_ssaliq", {n_band, n_size_liq}), {n_size_liq, n_band});
        Array<double,2> lut_asyliq(
                coef_nc.get_variable<double>("lut_asyliq", {n_band, n_size_liq}), {n_size_liq, n_band});

        Array<double,3> lut_extice(
                coef_nc.get_variable<double>("lut_extice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<double,3> lut_ssaice(
                coef_nc.get_variable<double>("lut_ssaice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});
        Array<double,3> lut_asyice(
                coef_nc.get_variable<double>("lut_asyice", {n_rghice, n_band, n_size_ice}), {n_size_ice, n_band, n_rghice});

        return Cloud_optics<double>(
                band_lims_wvn,
                radliq_lwr, radliq_upr, radliq_fac,
                radice_lwr, radice_upr, radice_fac,
                lut_extliq, lut_ssaliq, lut_asyliq,
                lut_extice, lut_ssaice, lut_asyice);
    }

    template<typename TF>
    void calc_tendency(
            TF* restrict thlt_rad,
            const double* restrict flux_up, const double* restrict flux_dn, // Fluxes are double precision.
            const TF* restrict rho, const TF* exner, const TF* dz,
            const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
            const int igc, const int jgc, const int kgc,
            const int jj, const int kk,
            const int jj_nogc, const int kk_nogc)
    {
        for (int k=kstart; k<kend; ++k)
        {
            // Conversion from energy to temperature.
            const TF fac = TF(1.) / (rho[k]*Constants::cp<TF>*exner[k]*dz[k]);

            for (int j=jstart; j<jend; ++j)
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    const int ijk_nogc = (i-igc) + (j-jgc)*jj_nogc + (k-kgc)*kk_nogc;

                    thlt_rad[ijk] -= fac *
                        ( flux_up[ijk_nogc+kk_nogc] - flux_up[ijk_nogc]
                        - flux_dn[ijk_nogc+kk_nogc] + flux_dn[ijk_nogc] );
                }
        }
    }

    template<typename TF>
    void add_tendency(
            TF* restrict thlt, const TF* restrict thlt_rad,
            const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
            const int jj, const int kk)
    {
        for (int k=kstart; k<kend; ++k)
            for (int j=jstart; j<jend; ++j)
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    thlt[ijk] += thlt_rad[ijk];
                }
    }

    template<typename TF>
    void solve_longwave_column(
            std::unique_ptr<Optical_props_arry<TF>>& optical_props,
            Array<TF,2>& flux_up, Array<TF,2>& flux_dn, Array<TF,2>& flux_net,
            Array<TF,2>& flux_dn_inc, const TF p_top,
            const Gas_concs<TF>& gas_concs,
            const std::unique_ptr<Gas_optics<TF>>& kdist_lw,
            const std::unique_ptr<Source_func_lw<TF>>& sources,
            const Array<TF,2>& col_dry,
            const Array<TF,2>& p_lay, const Array<TF,2>& p_lev,
            const Array<TF,2>& t_lay, const Array<TF,2>& t_lev,
            const Array<TF,1>& t_sfc, const Array<TF,2>& emis_sfc,
            const int n_lay)
    {
        const int n_col = 1;
        const int n_lev = n_lay + 1;

        // Set the number of angles to 1.
        const int n_ang = 1;

        // Check the dimension ordering.
        const int top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

        // Solve a single block, this does not require subsetting.
        kdist_lw->gas_optics(
                p_lay,
                p_lev,
                t_lay,
                t_sfc,
                gas_concs,
                optical_props,
                *sources,
                col_dry,
                t_lev);

        std::unique_ptr<Fluxes_broadband<double>> fluxes =
                std::make_unique<Fluxes_broadband<double>>(n_col, n_lev);

        const int n_gpt = kdist_lw->get_ngpt();
        Array<double,3> gpt_flux_up({n_col, n_lev, n_gpt});
        Array<double,3> gpt_flux_dn({n_col, n_lev, n_gpt});

        Rte_lw<double>::rte_lw(
                optical_props,
                top_at_1,
                *sources,
                emis_sfc,
                Array<double,2>(),
                gpt_flux_up,
                gpt_flux_dn,
                n_ang);

        fluxes->reduce(gpt_flux_up, gpt_flux_dn, optical_props, top_at_1);

        // Find the index where p_lev exceeds p_top.
        int idx_top=1;
        for (; idx_top<=n_lev; ++idx_top)
        {
            if (p_lev({1, idx_top}) < p_top)
                break;
        }

        // Calculate the interpolation factors.
        const int idx_bot = idx_top - 1;
        const double fac_bot = (p_top - p_lev({1, idx_top})) / (p_lev({1, idx_bot}) - p_lev({1, idx_top}));
        const double fac_top = 1. - fac_bot;

        // Interpolate the top boundary conditions.
        for (int igpt=1; igpt<=n_gpt; ++igpt)
            flux_dn_inc({1, igpt}) = fac_bot * gpt_flux_dn({1, idx_bot, igpt}) + fac_top * gpt_flux_dn({1, idx_top, igpt});

        // Copy the data to the output.
        for (int ilev=1; ilev<=n_lev; ++ilev)
            for (int icol=1; icol<=n_col; ++icol)
            {
                flux_up ({icol, ilev}) = fluxes->get_flux_up ()({icol, ilev});
                flux_dn ({icol, ilev}) = fluxes->get_flux_dn ()({icol, ilev});
                flux_net({icol, ilev}) = fluxes->get_flux_net()({icol, ilev});
            }
    }

    template<typename TF>
    void solve_shortwave_column(
            std::unique_ptr<Optical_props_arry<TF>>& optical_props,
            Array<TF,2>& flux_up, Array<TF,2>& flux_dn,
            Array<TF,2>& flux_dn_dir, Array<TF,2>& flux_net,
            Array<TF,2>& flux_dn_dir_inc, Array<TF,2>& flux_dn_dif_inc, const TF p_top,
            const Gas_concs<TF>& gas_concs,
            const Gas_optics<TF>& kdist_sw,
            const Array<TF,2>& col_dry,
            const Array<TF,2>& p_lay, const Array<TF,2>& p_lev,
            const Array<TF,2>& t_lay, const Array<TF,2>& t_lev,
            const Array<TF,1>& mu0,
            const Array<TF,2>& sfc_alb_dir, const Array<TF,2>& sfc_alb_dif,
            const TF tsi_scaling,
            const int n_lay)
    {
        const int n_col = 1;
        const int n_lev = n_lay + 1;

        // Check the dimension ordering.
        const int top_at_1 = p_lay({1, 1}) < p_lay({1, n_lay});

        // Create the field for the top of atmosphere source.
        const int n_gpt = kdist_sw.get_ngpt();
        Array<TF,2> toa_src({n_col, n_gpt});

        kdist_sw.gas_optics(
                p_lay,
                p_lev,
                t_lay,
                gas_concs,
                optical_props,
                toa_src,
                col_dry);

        if (tsi_scaling >= 0)
            for (int igpt=1; igpt<=n_gpt; ++igpt)
                toa_src({1, igpt}) *= tsi_scaling;

        std::unique_ptr<Fluxes_broadband<TF>> fluxes =
                std::make_unique<Fluxes_broadband<TF>>(n_col, n_lev);

        Array<double,3> gpt_flux_up    ({n_col, n_lev, n_gpt});
        Array<double,3> gpt_flux_dn    ({n_col, n_lev, n_gpt});
        Array<double,3> gpt_flux_dn_dir({n_col, n_lev, n_gpt});

        Rte_sw<TF>::rte_sw(
                optical_props,
                top_at_1,
                mu0,
                toa_src,
                sfc_alb_dir,
                sfc_alb_dif,
                Array<double,2>(),
                gpt_flux_up,
                gpt_flux_dn,
                gpt_flux_dn_dir);

        fluxes->reduce(
                gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir,
                optical_props, top_at_1);

        // Find the index where p_lev exceeds p_top.
        int idx_top=1;
        for (; idx_top<=n_lev; ++idx_top)
        {
            if (p_lev({1, idx_top}) < p_top)
                break;
        }

        // Calculate the interpolation factors.
        const int idx_bot = idx_top - 1;
        const double fac_bot = (p_top - p_lev({1, idx_top})) / (p_lev({1, idx_bot}) - p_lev({1, idx_top}));
        const double fac_top = 1. - fac_bot;

        // Interpolate the top boundary conditions.
        for (int igpt=1; igpt<=n_gpt; ++igpt)
        {
            const double flux_dn_tot = fac_bot * gpt_flux_dn    ({1, idx_bot, igpt}) + fac_top * gpt_flux_dn    ({1, idx_top, igpt});
            const double flux_dn_dir = fac_bot * gpt_flux_dn_dir({1, idx_bot, igpt}) + fac_top * gpt_flux_dn_dir({1, idx_top, igpt});
            // Divide out the cosine of the solar zenith angle.
            flux_dn_dir_inc({1, igpt}) = flux_dn_dir / mu0({1});
            flux_dn_dif_inc({1, igpt}) = flux_dn_tot - flux_dn_dir;
        }

        // Copy the data to the output.
        for (int ilev=1; ilev<=n_lev; ++ilev)
            for (int icol=1; icol<=n_col; ++icol)
            {
                flux_up    ({icol, ilev}) = fluxes->get_flux_up    ()({icol, ilev});
                flux_dn    ({icol, ilev}) = fluxes->get_flux_dn    ()({icol, ilev});
                flux_dn_dir({icol, ilev}) = fluxes->get_flux_dn_dir()({icol, ilev});
                flux_net   ({icol, ilev}) = fluxes->get_flux_net   ()({icol, ilev});
            }
    }
}

template<typename TF>
Radiation_rrtmgp<TF>::Radiation_rrtmgp(
        Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Input& inputin) :
	Radiation<TF>(masterin, gridin, fieldsin, inputin)
{
    swradiation = "rrtmgp";

    sw_longwave  = inputin.get_item<bool>("radiation", "swlongwave" , "", true);
    sw_shortwave = inputin.get_item<bool>("radiation", "swshortwave", "", true);

    sw_clear_sky_stats = inputin.get_item<bool>("radiation", "swclearskystats", "", false);

    dt_rad = inputin.get_item<double>("radiation", "dt_rad", "");

	t_sfc       = inputin.get_item<double>("radiation", "t_sfc"      , "");
    emis_sfc    = inputin.get_item<double>("radiation", "emis_sfc"   , "");
    sfc_alb_dir = inputin.get_item<double>("radiation", "sfc_alb_dir", "");
    sfc_alb_dif = inputin.get_item<double>("radiation", "sfc_alb_dif", "");
    tsi_scaling = inputin.get_item<double>("radiation", "tsi_scaling", "", -999.);

    const double sza = inputin.get_item<double>("radiation", "sza", "");
    mu0 = std::cos(sza);

    // Nc0 = inputin.get_item<double>("microphysics", "Nc0", "", 70e6);

    auto& gd = grid.get_grid_data();
    fields.init_diagnostic_field("thlt_rad", "Tendency by radiation", "K s-1", gd.sloc);
}

template<typename TF>
void Radiation_rrtmgp<TF>::init(const double ifactor)
{
    idt_rad = static_cast<unsigned long>(ifactor * dt_rad + 0.5);
}

template<typename TF>
void Radiation_rrtmgp<TF>::create(
        Input& input, Netcdf_handle& input_nc, Thermo<TF>& thermo,
        Stats<TF>& stats, Column<TF>& column, Cross<TF>& cross, Dump<TF>& dump)
{
    // Check if the thermo supports the radiation.
    if (thermo.get_switch() != "moist")
    {
        const std::string error = "Radiation does not support thermo mode " + thermo.get_switch();
        throw std::runtime_error(error);
    }

    // Initialize the tendency if the radiation is used.
    if (stats.get_switch() && (sw_longwave || sw_shortwave))
        stats.add_tendency(*fields.st.at("thl"), "z", tend_name, tend_longname);

    // Create the gas optics solver that is needed for the column and model solver.
    create_solver(input, input_nc, thermo, stats);

    // Solve the reference column to compute upper boundary conditions.
    create_column(input, input_nc, thermo, stats);

    // Get the allowed cross sections from the cross list
    std::vector<std::string> allowed_crossvars_radiation;

    if (sw_shortwave)
    {
        allowed_crossvars_radiation.push_back("sw_flux_up");
        allowed_crossvars_radiation.push_back("sw_flux_dn");
        allowed_crossvars_radiation.push_back("sw_flux_dn_dir");
        
        if (sw_clear_sky_stats)
        {
            allowed_crossvars_radiation.push_back("sw_flux_up_clear");
            allowed_crossvars_radiation.push_back("sw_flux_dn_clear");
            allowed_crossvars_radiation.push_back("sw_flux_dn_dir_clear");
        }
    }

    if (sw_longwave)
    {
        allowed_crossvars_radiation.push_back("lw_flux_up");
        allowed_crossvars_radiation.push_back("lw_flux_dn");

        if (sw_clear_sky_stats)
        {
            allowed_crossvars_radiation.push_back("lw_flux_up_clear");
            allowed_crossvars_radiation.push_back("lw_flux_dn_clear");
        }
    }

    crosslist = cross.get_enabled_variables(allowed_crossvars_radiation);
}

template<typename TF>
void Radiation_rrtmgp<TF>::create_column(
        Input& input, Netcdf_handle& input_nc, Thermo<TF>& thermo, Stats<TF>& stats)
{
    // 1. Load the available gas concentrations from the group of the netcdf file.
    Netcdf_handle& rad_nc = input_nc.get_group("radiation");

    Gas_concs<double> gas_concs_col;
    load_gas_concs<double>(gas_concs_col, rad_nc, "lay");

    // 2. Set the coordinate for the reference profiles in the stats, before calling the other creates.
    if (stats.get_switch() && (sw_longwave || sw_shortwave))
    {
        const int n_col = 1;
        const int n_lev = rad_nc.get_dimension_size("lev");
        Array<double,2> p_lev(rad_nc.get_variable<double>("p_lev", {n_lev, n_col}), {n_col, n_lev});

        stats.add_dimension("p_rad", n_lev);

        const std::string group_name = "radiation";
        const std::string root_group= "";

        // CvH, I put an vector copy here because radiation is always double.
        stats.add_fixed_prof_raw(
                "p_rad",
                "Pressure of radiation reference column",
                "Pa", "p_rad", root_group,
                std::vector<TF>(p_lev.v().begin(), p_lev.v().end()));
    }

    // 3. Call the column solvers for longwave and shortwave.
    if (sw_longwave)
        create_column_longwave (input, rad_nc, thermo, stats, gas_concs_col);
    if (sw_shortwave)
        create_column_shortwave(input, rad_nc, thermo, stats, gas_concs_col);
}

template<typename TF>
void Radiation_rrtmgp<TF>::create_column_longwave(
        Input& input, Netcdf_handle& rad_nc, Thermo<TF>& thermo, Stats<TF>& stats,
        const Gas_concs<double>& gas_concs)
{
    auto& gd = grid.get_grid_data();

    // 3. Read the atmospheric pressure and temperature.
    const int n_col = 1;

    const int n_lay = rad_nc.get_dimension_size("lay");
    const int n_lev = rad_nc.get_dimension_size("lev");

    Array<double,2> p_lay(rad_nc.get_variable<double>("p_lay", {n_lay, n_col}), {n_col, n_lay});
    Array<double,2> t_lay(rad_nc.get_variable<double>("t_lay", {n_lay, n_col}), {n_col, n_lay});
    Array<double,2> p_lev(rad_nc.get_variable<double>("p_lev", {n_lev, n_col}), {n_col, n_lev});
    Array<double,2> t_lev(rad_nc.get_variable<double>("t_lev", {n_lev, n_col}), {n_col, n_lev});

    Array<double,2> col_dry({n_col, n_lay});
    if (rad_nc.variable_exists("col_dry"))
        col_dry = rad_nc.get_variable<double>("col_dry", {n_lay, n_col});
    else
        Gas_optics<double>::get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);

    // 4. Read the boundary conditions.
    // Set the surface temperature and emissivity.
    // CvH: communicate with surface scheme.
    Array<double,1> t_sfc({1});
    t_sfc({1}) = this->t_sfc;

    const int n_bnd = kdist_lw->get_nband();
    Array<double,2> emis_sfc({n_bnd, 1});
    for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
        emis_sfc({ibnd, 1}) = this->emis_sfc;

    // Compute the longwave for the reference profile.
    std::unique_ptr<Source_func_lw<double>> sources_lw =
            std::make_unique<Source_func_lw<double>>(n_col, n_lay, *kdist_lw);

    std::unique_ptr<Optical_props_arry<double>> optical_props_lw =
            std::make_unique<Optical_props_1scl<double>>(n_col, n_lay, *kdist_lw);

    Array<double,2> lw_flux_up ({n_col, n_lev});
    Array<double,2> lw_flux_dn ({n_col, n_lev});
    Array<double,2> lw_flux_net({n_col, n_lev});

    const int n_gpt = kdist_lw->get_ngpt();
    lw_flux_dn_inc.set_dims({n_col, n_gpt});

    solve_longwave_column<double>(
            optical_props_lw,
            lw_flux_up, lw_flux_dn, lw_flux_net,
            lw_flux_dn_inc, thermo.get_ph_vector()[gd.kend],
            gas_concs,
            kdist_lw,
            sources_lw,
            col_dry,
            p_lay, p_lev,
            t_lay, t_lev,
            t_sfc, emis_sfc,
            n_lay);

    // Save the reference profile fluxes in the stats.
    if (stats.get_switch())
    {
        const std::string group_name = "radiation";

        // CvH, I put an vector copy here because radiation is always double.
        stats.add_fixed_prof_raw(
                "lw_flux_up_ref",
                "Longwave upwelling flux of reference column",
                "W m-2", "p_rad", group_name,
                std::vector<TF>(lw_flux_up.v().begin(), lw_flux_up.v().end()));
        stats.add_fixed_prof_raw(
                "lw_flux_dn_ref",
                "Longwave downwelling flux of reference column",
                "W m-2", "p_rad", group_name,
                std::vector<TF>(lw_flux_dn.v().begin(), lw_flux_dn.v().end()));
    }
}

template<typename TF>
void Radiation_rrtmgp<TF>::create_column_shortwave(
        Input& input, Netcdf_handle& rad_nc, Thermo<TF>& thermo, Stats<TF>& stats,
        const Gas_concs<double>& gas_concs)
{
    auto& gd = grid.get_grid_data();

    // 3. Read the atmospheric pressure and temperature.
    const int n_col = 1;

    const int n_lay = rad_nc.get_dimension_size("lay");
    const int n_lev = rad_nc.get_dimension_size("lev");

    Array<double,2> p_lay(rad_nc.get_variable<double>("p_lay", {n_lay, n_col}), {n_col, n_lay});
    Array<double,2> t_lay(rad_nc.get_variable<double>("t_lay", {n_lay, n_col}), {n_col, n_lay});
    Array<double,2> p_lev(rad_nc.get_variable<double>("p_lev", {n_lev, n_col}), {n_col, n_lev});
    Array<double,2> t_lev(rad_nc.get_variable<double>("t_lev", {n_lev, n_col}), {n_col, n_lev});

    Array<double,2> col_dry({n_col, n_lay});
    if (rad_nc.variable_exists("col_dry"))
        col_dry = rad_nc.get_variable<double>("col_dry", {n_lay, n_col});
    else
        Gas_optics<double>::get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev);

    // 4. Read the boundary conditions.
    const int n_bnd = kdist_sw->get_nband();

    // Set the solar zenith angle and albedo.
    Array<double,2> sfc_alb_dir({n_bnd, n_col});
    Array<double,2> sfc_alb_dif({n_bnd, n_col});

    for (int ibnd=1; ibnd<=n_bnd; ++ibnd)
    {
        sfc_alb_dir({ibnd, 1}) = this->sfc_alb_dir;
        sfc_alb_dif({ibnd, 1}) = this->sfc_alb_dif;
    }

    Array<double,1> mu0({n_col});
    mu0({1}) = this->mu0;

    std::unique_ptr<Optical_props_arry<double>> optical_props_sw =
            std::make_unique<Optical_props_2str<double>>(n_col, n_lay, *kdist_sw);

    Array<double,2> sw_flux_up    ({n_col, n_lev});
    Array<double,2> sw_flux_dn    ({n_col, n_lev});
    Array<double,2> sw_flux_dn_dir({n_col, n_lev});
    Array<double,2> sw_flux_net   ({n_col, n_lev});

    const int n_gpt = kdist_sw->get_ngpt();
    sw_flux_dn_dir_inc.set_dims({n_col, n_gpt});
    sw_flux_dn_dif_inc.set_dims({n_col, n_gpt});

    solve_shortwave_column<double>(
            optical_props_sw,
            sw_flux_up, sw_flux_dn, sw_flux_dn_dir, sw_flux_net,
            sw_flux_dn_dir_inc, sw_flux_dn_dif_inc, thermo.get_ph_vector()[gd.kend],
            gas_concs,
            *kdist_sw,
            col_dry,
            p_lay, p_lev,
            t_lay, t_lev,
            mu0,
            sfc_alb_dir, sfc_alb_dif,
            tsi_scaling,
            n_lay);

    // Save the reference profile fluxes in the stats.
    if (stats.get_switch())
    {
        const std::string group_name = "radiation";

        // CvH, I put an vector copy here because radiation is always double.
        stats.add_fixed_prof_raw(
                "sw_flux_up_ref",
                "Shortwave upwelling flux of reference column",
                "W m-2", "p_rad", group_name,
                std::vector<TF>(sw_flux_up.v().begin(), sw_flux_up.v().end()));
        stats.add_fixed_prof_raw(
                "sw_flux_dn_ref",
                "Shortwave downwelling flux of reference column",
                "W m-2", "p_rad", group_name,
                std::vector<TF>(sw_flux_dn.v().begin(), sw_flux_dn.v().end()));
        stats.add_fixed_prof_raw(
                "sw_flux_dn_dir_ref",
                "Shortwave direct downwelling flux of reference column",
                "W m-2", "p_rad", group_name,
                std::vector<TF>(sw_flux_dn_dir.v().begin(), sw_flux_dn_dir.v().end()));
    }
}

template<typename TF>
void Radiation_rrtmgp<TF>::create_solver(
        Input& input, Netcdf_handle& input_nc, Thermo<TF>& thermo, Stats<TF>& stats)
{
    // 1. Load the available gas concentrations from the group of the netcdf file.
    Netcdf_handle& rad_input_nc = input_nc.get_group("init");
    load_gas_concs<double>(gas_concs, rad_input_nc, "z");

    // 2. Pass the gas concentrations to the solver initializers.
    if (sw_longwave)
        create_solver_longwave(input, input_nc, thermo, stats, gas_concs);
    if (sw_shortwave)
        create_solver_shortwave(input, input_nc, thermo, stats, gas_concs);
}

template<typename TF>
void Radiation_rrtmgp<TF>::create_solver_longwave(
        Input& input, Netcdf_handle& input_nc, Thermo<TF>& thermo, Stats<TF>& stats,
        const Gas_concs<double>& gas_concs)
{
    const std::string group_name = "radiation";

    // Set up the gas optics classes for long and shortwave.
    kdist_lw = std::make_unique<Gas_optics<double>>(
            load_and_init_gas_optics(master, gas_concs, "coefficients_lw.nc"));

    cloud_lw = std::make_unique<Cloud_optics<double>>(
            load_and_init_cloud_optics(master, "cloud_coefficients_lw.nc"));

    // Set up the statistics.
    if (stats.get_switch())
    {
        stats.add_prof("lw_flux_up", "Longwave upwelling flux"  , "W m-2", "zh", group_name);
        stats.add_prof("lw_flux_dn", "Longwave downwelling flux", "W m-2", "zh", group_name);

        if (sw_clear_sky_stats)
        {
            stats.add_prof("lw_flux_up_clear", "Clear-sky longwave upwelling flux"  , "W m-2", "zh", group_name);
            stats.add_prof("lw_flux_dn_clear", "Clear-sky longwave downwelling flux", "W m-2", "zh", group_name);
        }
    }
}

template<typename TF>
void Radiation_rrtmgp<TF>::create_solver_shortwave(
        Input& input, Netcdf_handle& input_nc, Thermo<TF>& thermo, Stats<TF>& stats,
        const Gas_concs<double>& gas_concs)
{
    const std::string group_name = "radiation";

    // Set up the gas optics classes for long and shortwave.
    kdist_sw = std::make_unique<Gas_optics<double>>(
            load_and_init_gas_optics(master, gas_concs, "coefficients_sw.nc"));

    cloud_sw = std::make_unique<Cloud_optics<double>>(
            load_and_init_cloud_optics(master, "cloud_coefficients_sw.nc"));

    // Set up the statistics.
    if (stats.get_switch())
    {
        stats.add_prof("sw_flux_up"    , "Shortwave upwelling flux"         , "W m-2", "zh", group_name);
        stats.add_prof("sw_flux_dn"    , "Shortwave downwelling flux"       , "W m-2", "zh", group_name);
        stats.add_prof("sw_flux_dn_dir", "Shortwave direct downwelling flux", "W m-2", "zh", group_name);

        if (sw_clear_sky_stats)
        {
            stats.add_prof("sw_flux_up_clear"    , "Clear-sky shortwave upwelling flux"         , "W m-2", "zh", group_name);
            stats.add_prof("sw_flux_dn_clear"    , "Clear-sky shortwave downwelling flux"       , "W m-2", "zh", group_name);
            stats.add_prof("sw_flux_dn_dir_clear", "Clear-sky shortwave direct downwelling flux", "W m-2", "zh", group_name);
        }
    }
}

#ifndef USECUDA
template<typename TF>
void Radiation_rrtmgp<TF>::exec(
        Thermo<TF>& thermo, const double time, Timeloop<TF>& timeloop, Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();

    const bool do_radiation = (timeloop.get_itime() % idt_rad == 0);

    if (do_radiation)
    {
        // Set the tendency to zero.
        std::fill(fields.sd.at("thlt_rad")->fld.begin(), fields.sd.at("thlt_rad")->fld.end(), TF(0.));

        auto t_lay = fields.get_tmp();
        auto t_lev = fields.get_tmp();
        auto h2o   = fields.get_tmp(); // This is the volume mixing ratio, not the specific humidity of vapor.
        auto clwp  = fields.get_tmp();
        auto ciwp  = fields.get_tmp();

        // Set the input to the radiation on a 3D grid without ghost cells.
        thermo.get_radiation_fields(*t_lay, *t_lev, *h2o, *clwp, *ciwp);

        // Initialize arrays in double precision, cast when needed.
        const int nmaxh = gd.imax*gd.jmax*(gd.ktot+1);

        Array<double,2> t_lay_a(
                std::vector<double>(t_lay->fld.begin(), t_lay->fld.begin() + gd.nmax), {gd.imax*gd.jmax, gd.ktot});
        Array<double,2> t_lev_a(
                std::vector<double>(t_lev->fld.begin(), t_lev->fld.begin() + nmaxh), {gd.imax*gd.jmax, gd.ktot+1});
        Array<double,2> h2o_a(
                std::vector<double>(h2o->fld.begin(), h2o->fld.begin() + gd.nmax), {gd.imax*gd.jmax, gd.ktot});
        Array<double,2> clwp_a(
                std::vector<double>(clwp->fld.begin(), clwp->fld.begin() + gd.nmax), {gd.imax*gd.jmax, gd.ktot});
        Array<double,2> ciwp_a(
                std::vector<double>(ciwp->fld.begin(), ciwp->fld.begin() + gd.nmax), {gd.imax*gd.jmax, gd.ktot});

        Array<double,2> flux_up ({gd.imax*gd.jmax, gd.ktot+1});
        Array<double,2> flux_dn ({gd.imax*gd.jmax, gd.ktot+1});
        Array<double,2> flux_net({gd.imax*gd.jmax, gd.ktot+1});

        const bool compute_clouds = true;

        if (sw_longwave)
        {
            exec_longwave(
                    thermo, timeloop, stats,
                    flux_up, flux_dn, flux_net,
                    t_lay_a, t_lev_a, h2o_a, clwp_a, ciwp_a,
                    compute_clouds);

            calc_tendency(
                    fields.sd.at("thlt_rad")->fld.data(),
                    flux_up.ptr(), flux_dn.ptr(),
                    fields.rhoref.data(), thermo.get_exner_vector().data(),
                    gd.dz.data(),
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.igc, gd.jgc, gd.kgc,
                    gd.icells, gd.ijcells,
                    gd.imax, gd.imax*gd.jmax);
        }

        if (sw_shortwave)
        {
            Array<double,2> flux_dn_dir({gd.imax*gd.jmax, gd.ktot+1});

            exec_shortwave(
                    thermo, timeloop, stats,
                    flux_up, flux_dn, flux_dn_dir, flux_net,
                    t_lay_a, t_lev_a, h2o_a, clwp_a, ciwp_a,
                    compute_clouds);

            calc_tendency(
                    fields.sd.at("thlt_rad")->fld.data(),
                    flux_up.ptr(), flux_dn.ptr(),
                    fields.rhoref.data(), thermo.get_exner_vector().data(),
                    gd.dz.data(),
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
                    gd.igc, gd.jgc, gd.kgc,
                    gd.icells, gd.ijcells,
                    gd.imax, gd.imax*gd.jmax);
        }

        fields.release_tmp(t_lay);
        fields.release_tmp(t_lev);
        fields.release_tmp(h2o);
        fields.release_tmp(clwp);
        fields.release_tmp(ciwp);
    }

    // Always add the tendency.
    add_tendency(
            fields.st.at("thl")->fld.data(),
            fields.sd.at("thlt_rad")->fld.data(),
            gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend,
            gd.icells, gd.ijcells);

    stats.calc_tend(*fields.st.at("thl"), tend_name);
}
#endif

namespace
{
    template<typename TF>
    void add_ghost_cells(
            TF* restrict out, const double* restrict in,
            const int istart, const int iend,
            const int jstart, const int jend,
            const int kstart, const int kend_field,
            const int igc, const int jgc, const int kgc,
            const int jj, const int kk,
            const int jj_nogc, const int kk_nogc)
    {
        // Value of kend_field is either kend or kend+1.
        #pragma omp parallel for
        for (int k=kstart; k<kend_field; ++k)
            for (int j=jstart; j<jend; ++j)
                #pragma ivdep
                for (int i=istart; i<iend; ++i)
                {
                    const int ijk = i + j*jj + k*kk;
                    const int ijk_nogc = (i-igc) + (j-jgc)*jj_nogc + (k-kgc)*kk_nogc;
                    out[ijk] = in[ijk_nogc];
                }
    }
}

template<typename TF>
void Radiation_rrtmgp<TF>::exec_all_stats(
        Stats<TF>& stats, Cross<TF>& cross, Dump<TF>& dump,
        Thermo<TF>& thermo, Timeloop<TF>& timeloop,
        const unsigned long itime, const int iotime)
{
    const bool do_stats = stats.do_statistics(itime);
    const bool do_cross = cross.do_cross(itime);

    // Return in case of no stats or cross section.
    if ( !(do_stats || do_cross) )
        return;

    const TF no_offset = 0.;
    const TF no_threshold = 0.;

    // CvH: lots of code repetition with exec()
    auto& gd = grid.get_grid_data();

    auto t_lay = fields.get_tmp();
    auto t_lev = fields.get_tmp();
    auto h2o   = fields.get_tmp(); // This is the volume mixing ratio, not the specific humidity of vapor.
    auto clwp  = fields.get_tmp();
    auto ciwp  = fields.get_tmp();

    // Set the input to the radiation on a 3D grid without ghost cells.
    thermo.get_radiation_fields(*t_lay, *t_lev, *h2o, *clwp, *ciwp);

    // Initialize arrays in double precision, cast when needed.
    const int nmaxh = gd.imax*gd.jmax*(gd.ktot+1);

    Array<double,2> t_lay_a(
            std::vector<double>(t_lay->fld.begin(), t_lay->fld.begin() + gd.nmax), {gd.imax*gd.jmax, gd.ktot});
    Array<double,2> t_lev_a(
            std::vector<double>(t_lev->fld.begin(), t_lev->fld.begin() + nmaxh), {gd.imax*gd.jmax, gd.ktot+1});
    Array<double,2> h2o_a(
            std::vector<double>(h2o->fld.begin(), h2o->fld.begin() + gd.nmax), {gd.imax*gd.jmax, gd.ktot});
    Array<double,2> clwp_a(
            std::vector<double>(clwp->fld.begin(), clwp->fld.begin() + gd.nmax), {gd.imax*gd.jmax, gd.ktot});
    Array<double,2> ciwp_a(
            std::vector<double>(ciwp->fld.begin(), ciwp->fld.begin() + gd.nmax), {gd.imax*gd.jmax, gd.ktot});

    fields.release_tmp(t_lay);
    fields.release_tmp(t_lev);
    fields.release_tmp(h2o);
    fields.release_tmp(clwp);
    fields.release_tmp(ciwp);

    Array<double,2> flux_up ({gd.imax*gd.jmax, gd.ktot+1});
    Array<double,2> flux_dn ({gd.imax*gd.jmax, gd.ktot+1});
    Array<double,2> flux_net({gd.imax*gd.jmax, gd.ktot+1});

    auto tmp = fields.get_tmp();
    tmp->loc = gd.wloc;

    const bool compute_clouds = true;

    // Use a lambda function to avoid code repetition.
    auto save_stats_and_cross = [&](
            const Array<double,2>& array, const std::string& name, const std::array<int,3>& loc)
    {
        if (do_stats || do_cross)
        {
            // Make sure that the top boundary is taken into account in case of fluxes.
            const int kend = gd.kstart + array.dim(2) + 1;
            add_ghost_cells(
                    tmp->fld.data(), array.ptr(),
                    gd.istart, gd.iend,
                    gd.jstart, gd.jend,
                    gd.kstart, kend,
                    gd.igc, gd.jgc, gd.kgc,
                    gd.icells, gd.ijcells,
                    gd.imax, gd.imax*gd.jmax);
        }

        if (do_stats)
            stats.calc_stats(name, *tmp, no_offset, no_threshold);

        if (do_cross)
        {
            if (std::find(crosslist.begin(), crosslist.end(), name) != crosslist.end())
                cross.cross_simple(tmp->fld.data(), name, iotime, loc);
        }
    };

    if (sw_longwave)
    {
        exec_longwave(
                thermo, timeloop, stats,
                flux_up, flux_dn, flux_net,
                t_lay_a, t_lev_a, h2o_a, clwp_a, ciwp_a,
                compute_clouds);

        save_stats_and_cross(flux_up, "lw_flux_up", gd.wloc);
        save_stats_and_cross(flux_dn, "lw_flux_dn", gd.wloc);

        if (sw_clear_sky_stats)
        {
            exec_longwave(
                    thermo, timeloop, stats,
                    flux_up, flux_dn, flux_net,
                    t_lay_a, t_lev_a, h2o_a, clwp_a, ciwp_a,
                    !compute_clouds);

            save_stats_and_cross(flux_up, "lw_flux_up_clear", gd.wloc);
            save_stats_and_cross(flux_dn, "lw_flux_dn_clear", gd.wloc);
        }
    }

    if (sw_shortwave)
    {
        Array<double,2> flux_dn_dir({gd.imax*gd.jmax, gd.ktot+1});

        exec_shortwave(
                thermo, timeloop, stats,
                flux_up, flux_dn, flux_dn_dir, flux_net,
                t_lay_a, t_lev_a, h2o_a, clwp_a, ciwp_a,
                compute_clouds);

        save_stats_and_cross(flux_up,     "sw_flux_up"    , gd.wloc);
        save_stats_and_cross(flux_dn,     "sw_flux_dn"    , gd.wloc);
        save_stats_and_cross(flux_dn_dir, "sw_flux_dn_dir", gd.wloc);

        if (sw_clear_sky_stats)
        {
            exec_shortwave(
                    thermo, timeloop, stats,
                    flux_up, flux_dn, flux_dn_dir, flux_net,
                    t_lay_a, t_lev_a, h2o_a, clwp_a, ciwp_a,
                    !compute_clouds);

            save_stats_and_cross(flux_up,     "sw_flux_up_clear"    , gd.wloc);
            save_stats_and_cross(flux_dn,     "sw_flux_dn_clear"    , gd.wloc);
            save_stats_and_cross(flux_dn_dir, "sw_flux_dn_dir_clear", gd.wloc);
        }
    }

    fields.release_tmp(tmp);
}

template<typename TF>
void Radiation_rrtmgp<TF>::exec_longwave(
        Thermo<TF>& thermo, Timeloop<TF>& timeloop, Stats<TF>& stats,
        Array<double,2>& flux_up, Array<double,2>& flux_dn, Array<double,2>& flux_net,
        const Array<double,2>& t_lay, const Array<double,2>& t_lev,
        const Array<double,2>& h2o, const Array<double,2>& clwp, const Array<double,2>& ciwp,
        const bool compute_clouds)
{
    // How many profiles are solved simultaneously?
    constexpr int n_col_block = 4;

    auto& gd = grid.get_grid_data();

    const int n_lay = gd.ktot;
    const int n_lev = gd.ktot+1;
    const int n_col = gd.imax*gd.jmax;

    const int n_blocks = n_col / n_col_block;
    const int n_col_block_left = n_col % n_col_block;

    // Store the number of bands and gpt in a variable.
    const int n_bnd = kdist_lw->get_nband();
    const int n_gpt = kdist_lw->get_ngpt();

    // Set the number of angles to 1.
    const int n_ang = 1;

    // Check the dimension ordering. The top is not at 1 in MicroHH, but the surface is.
    const int top_at_1 = 0;

    // Define the pointers for the subsetting.
    std::unique_ptr<Optical_props_arry<double>> optical_props_subset =
            std::make_unique<Optical_props_1scl<double>>(n_col_block, n_lay, *kdist_lw);
    std::unique_ptr<Source_func_lw<double>> sources_subset =
            std::make_unique<Source_func_lw<double>>(n_col_block, n_lay, *kdist_lw);
    std::unique_ptr<Optical_props_1scl<double>> cloud_optical_props_subset =
            std::make_unique<Optical_props_1scl<double>>(n_col_block, n_lay, *cloud_lw);

    std::unique_ptr<Optical_props_arry<double>> optical_props_left =
            std::make_unique<Optical_props_1scl<double>>(n_col_block_left, n_lay, *kdist_lw);
    std::unique_ptr<Source_func_lw<double>> sources_left =
            std::make_unique<Source_func_lw<double>>(n_col_block_left, n_lay, *kdist_lw);
    std::unique_ptr<Optical_props_1scl<double>> cloud_optical_props_left =
            std::make_unique<Optical_props_1scl<double>>(n_col_block_left, n_lay, *cloud_lw);

    // Define the arrays that contain the subsets.
    Array<double,2> p_lay(std::vector<double>(thermo.get_p_vector ().begin() + gd.kstart, thermo.get_p_vector ().begin() + gd.kend    ), {1, n_lay});
    Array<double,2> p_lev(std::vector<double>(thermo.get_ph_vector().begin() + gd.kstart, thermo.get_ph_vector().begin() + gd.kend + 1), {1, n_lev});

    Array<double,1> t_sfc(std::vector<double>(1, this->t_sfc), {1});
    Array<double,2> emis_sfc(std::vector<double>(n_bnd, this->emis_sfc), {n_bnd, 1});

    gas_concs.set_vmr("h2o", h2o);
    Array<double,2> col_dry({n_col, n_lay});
    Gas_optics<double>::get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev.subset({{ {1, n_col}, {1, n_lev} }}));

    // Lambda function for solving optical properties subset.
    auto call_kernels = [&](
            const int col_s_in, const int col_e_in,
            std::unique_ptr<Optical_props_arry<double>>& optical_props_subset_in,
            std::unique_ptr<Optical_props_1scl<double>>& cloud_optical_props_in,
            Source_func_lw<double>& sources_subset_in,
            const Array<double,2>& emis_sfc_subset_in,
            const Array<double,2>& lw_flux_dn_inc_subset_in,
            std::unique_ptr<Fluxes_broadband<double>>& fluxes)
    {
        const int n_col_in = col_e_in - col_s_in + 1;
        Gas_concs<double> gas_concs_subset(gas_concs, col_s_in, n_col_in);

        kdist_lw->gas_optics(
                p_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                p_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }}),
                t_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                t_sfc.subset({{ {col_s_in, col_e_in} }}),
                gas_concs_subset,
                optical_props_subset_in,
                sources_subset_in,
                col_dry.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                t_lev  .subset({{ {col_s_in, col_e_in}, {1, n_lev} }}) );

        // 2. Solve the cloud optical properties.
        if (compute_clouds)
        {
            Array<double,2> clwp_subset(clwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}));
            Array<double,2> ciwp_subset(ciwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}));

            // Set the masks.
            constexpr double mask_min_value = 1e-12; // DALES uses 1e-20.

            Array<int,2> cld_mask_liq({n_col_in, n_lay});
            for (int i=0; i<cld_mask_liq.size(); ++i)
                cld_mask_liq.v()[i] = clwp_subset.v()[i] > mask_min_value;

            Array<int,2> cld_mask_ice({n_col_in, n_lay});
            for (int i=0; i<cld_mask_ice.size(); ++i)
                cld_mask_ice.v()[i] = ciwp_subset.v()[i] > mask_min_value;

            // Compute the effective droplet radius.
            Array<double,2> rel({n_col_in, n_lay});
            Array<double,2> rei({n_col_in, n_lay});

            const double sig_g = 1.34;
            const double fac = std::exp(std::log(sig_g)*std::log(sig_g)) * 1e6; // Conversion to micron included.

            // CvH: Numbers according to RCEMIP.
            const double Nc0 = 100.e6;
            const double Ni0 = 1.e5;

            const double four_pi_Nc0_rho_w = 4.*M_PI*Nc0*Constants::rho_w<double>;
            const double four_pi_Ni0_rho_i = 4.*M_PI*Ni0*Constants::rho_i<double>;

            for (int ilay=1; ilay<=n_lay; ++ilay)
            {
                const double layer_mass = (p_lev({1, ilay}) - p_lev({1, ilay+1})) / Constants::grav<double>;
                for (int icol=1; icol<=n_col_in; ++icol)
                {
                    // Parametrization according to Martin et al., 1994 JAS. Fac multiplication taken from DALES.
                    // CvH: Potentially better using moments from microphysics.
                    double rel_value = cld_mask_liq({icol, ilay}) * fac *
                        std::pow(3.*(clwp_subset({icol, ilay})/layer_mass) / four_pi_Nc0_rho_w, (1./3.));

                    // Limit the values between 2.5 and 60.
                    rel_value = std::max(2.5, std::min(rel_value, 60.));
                    rel({icol, ilay}) = rel_value;

                    // Calculate the effective radius of ice from the mass and the number concentration.
                    double rei_value = cld_mask_ice({icol, ilay}) * 1.e6 *
                        std::pow(3.*(ciwp_subset({icol, ilay})/layer_mass) / four_pi_Ni0_rho_i, (1./3.));

                    // Limit the values between 2.5 and 200.
                    rei_value = std::max(2.5, std::min(rei_value, 200.));
                    rei({icol, ilay}) = rei_value;
                }
            }

            // Convert to g/m2.
            for (int i=0; i<clwp_subset.size(); ++i)
                clwp_subset.v()[i] *= 1e3;

            for (int i=0; i<ciwp_subset.size(); ++i)
                ciwp_subset.v()[i] *= 1e3;

            cloud_lw->cloud_optics(
                    cld_mask_liq, cld_mask_ice,
                    clwp_subset, ciwp_subset,
                    rel, rei,
                    *cloud_optical_props_in);

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_1scl<double>&>(*optical_props_subset_in),
                    dynamic_cast<Optical_props_1scl<double>&>(*cloud_optical_props_in));
        }

        Array<double,3> gpt_flux_up({n_col_in, n_lev, n_gpt});
        Array<double,3> gpt_flux_dn({n_col_in, n_lev, n_gpt});

        Rte_lw<double>::rte_lw(
                optical_props_subset_in,
                top_at_1,
                sources_subset_in,
                emis_sfc_subset_in,
                lw_flux_dn_inc_subset_in,
                gpt_flux_up, gpt_flux_dn,
                n_ang);

        fluxes->reduce(gpt_flux_up, gpt_flux_dn, optical_props_subset_in, top_at_1);

        // Copy the data to the output.
        for (int ilev=1; ilev<=n_lev; ++ilev)
            for (int icol=1; icol<=n_col_in; ++icol)
            {
                flux_up ({icol+col_s_in-1, ilev}) = fluxes->get_flux_up ()({icol, ilev});
                flux_dn ({icol+col_s_in-1, ilev}) = fluxes->get_flux_dn ()({icol, ilev});
                flux_net({icol+col_s_in-1, ilev}) = fluxes->get_flux_net()({icol, ilev});
            }
    };

    for (int b=1; b<=n_blocks; ++b)
    {
        const int col_s = (b-1) * n_col_block + 1;
        const int col_e =  b    * n_col_block;

        Array<double,2> emis_sfc_subset = emis_sfc.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array<double,2> lw_flux_dn_inc_subset = lw_flux_dn_inc.subset({{ {col_s, col_e}, {1, n_gpt} }});
        std::unique_ptr<Fluxes_broadband<double>> fluxes_subset =
                std::make_unique<Fluxes_broadband<double>>(n_col_block, n_lev);

        call_kernels(
                col_s, col_e,
                optical_props_subset,
                cloud_optical_props_subset,
                *sources_subset,
                emis_sfc_subset,
                lw_flux_dn_inc_subset,
                fluxes_subset);
    }

    if (n_col_block_left > 0)
    {
        const int col_s = n_col - n_col_block_left + 1;
        const int col_e = n_col;

        Array<double,2> emis_sfc_left = emis_sfc.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array<double,2> lw_flux_dn_inc_left = lw_flux_dn_inc.subset({{ {col_s, col_e}, {1, n_gpt} }});
        std::unique_ptr<Fluxes_broadband<double>> fluxes_left =
                std::make_unique<Fluxes_broadband<double>>(n_col_block_left, n_lev);

        call_kernels(
                col_s, col_e,
                optical_props_left,
                cloud_optical_props_left,
                *sources_left,
                emis_sfc_left,
                lw_flux_dn_inc_left,
                fluxes_left);
    }
}

template<typename TF>
void Radiation_rrtmgp<TF>::exec_shortwave(
        Thermo<TF>& thermo, Timeloop<TF>& timeloop, Stats<TF>& stats,
        Array<double,2>& flux_up, Array<double,2>& flux_dn, Array<double,2>& flux_dn_dir, Array<double,2>& flux_net,
        const Array<double,2>& t_lay, const Array<double,2>& t_lev,
        const Array<double,2>& h2o, const Array<double,2>& clwp, const Array<double,2>& ciwp,
        const bool compute_clouds)
{
    // How many profiles are solved simultaneously?
    constexpr int n_col_block = 4;

    auto& gd = grid.get_grid_data();

    const int n_lay = gd.ktot;
    const int n_lev = gd.ktot+1;
    const int n_col = gd.imax*gd.jmax;

    const int n_blocks = n_col / n_col_block;
    const int n_col_block_left = n_col % n_col_block;

    // Store the number of bands and gpt in a variable.
    const int n_bnd = kdist_sw->get_nband();
    const int n_gpt = kdist_sw->get_ngpt();

    // Check the dimension ordering. The top is not at 1 in MicroHH, but the surface is.
    const int top_at_1 = 0;

    // Define the pointers for the subsetting.
    std::unique_ptr<Optical_props_arry<double>> optical_props_subset =
            std::make_unique<Optical_props_2str<double>>(n_col_block, n_lay, *kdist_sw);
    std::unique_ptr<Optical_props_2str<double>> cloud_optical_props_subset =
            std::make_unique<Optical_props_2str<double>>(n_col_block, n_lay, *cloud_sw);

    std::unique_ptr<Optical_props_arry<double>> optical_props_left =
            std::make_unique<Optical_props_2str<double>>(n_col_block_left, n_lay, *kdist_sw);
    std::unique_ptr<Optical_props_2str<double>> cloud_optical_props_left =
            std::make_unique<Optical_props_2str<double>>(n_col_block_left, n_lay, *cloud_sw);

    // Define the arrays that contain the subsets.
    Array<double,2> p_lay(std::vector<double>(thermo.get_p_vector ().begin() + gd.kstart, thermo.get_p_vector ().begin() + gd.kend    ), {1, n_lay});
    Array<double,2> p_lev(std::vector<double>(thermo.get_ph_vector().begin() + gd.kstart, thermo.get_ph_vector().begin() + gd.kend + 1), {1, n_lev});

    // Create the boundary conditions
    Array<double,1> mu0(std::vector<double>(1, this->mu0), {1});
    Array<double,2> sfc_alb_dir(std::vector<double>(n_bnd, this->sfc_alb_dir), {n_bnd, 1});
    Array<double,2> sfc_alb_dif(std::vector<double>(n_bnd, this->sfc_alb_dif), {n_bnd, 1});

    // Create the field for the top of atmosphere source.
    Array<double,2> toa_src({n_col, n_gpt});

    gas_concs.set_vmr("h2o", h2o);
    Array<double,2> col_dry({n_col, n_lay});
    Gas_optics<double>::get_col_dry(col_dry, gas_concs.get_vmr("h2o"), p_lev.subset({{ {1, n_col}, {1, n_lev} }}));

    // Lambda function for solving optical properties subset.
    auto call_kernels = [&](
            const int col_s_in, const int col_e_in,
            std::unique_ptr<Optical_props_arry<double>>& optical_props_subset_in,
            std::unique_ptr<Optical_props_2str<double>>& cloud_optical_props_in,
            const Array<double,1>& mu0_subset_in,
            const Array<double,2>& toa_src_subset_in,
            const Array<double,2>& sfc_alb_dir_subset_in,
            const Array<double,2>& sfc_alb_dif_subset_in,
            const Array<double,2>& sw_flux_dn_dif_inc_subset_in,
            std::unique_ptr<Fluxes_broadband<double>>& fluxes)
    {
        const int n_col_in = col_e_in - col_s_in + 1;

        Gas_concs<double> gas_concs_subset(gas_concs, col_s_in, n_col_in);
        Array<double,2> toa_src_dummy({n_col_in, n_gpt});

        // 1. Solve the gas optical properties.
        kdist_sw->gas_optics(
                p_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                p_lev.subset({{ {col_s_in, col_e_in}, {1, n_lev} }}),
                t_lay.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}),
                gas_concs_subset,
                optical_props_subset_in,
                toa_src_dummy,
                col_dry.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}) );

        // 2. Solve the cloud optical properties.
        if (compute_clouds)
        {
            Array<double,2> clwp_subset(clwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}));
            Array<double,2> ciwp_subset(ciwp.subset({{ {col_s_in, col_e_in}, {1, n_lay} }}));

            // Set the masks.
            constexpr double mask_min_value = 1e-12; // DALES uses 1e-20.

            Array<int,2> cld_mask_liq({n_col_in, n_lay});
            for (int i=0; i<cld_mask_liq.size(); ++i)
                cld_mask_liq.v()[i] = clwp_subset.v()[i] > mask_min_value;

            Array<int,2> cld_mask_ice({n_col_in, n_lay});
            for (int i=0; i<cld_mask_ice.size(); ++i)
                cld_mask_ice.v()[i] = ciwp_subset.v()[i] > mask_min_value;

            // Compute the effective droplet radius.
            Array<double,2> rel({n_col_in, n_lay});
            Array<double,2> rei({n_col_in, n_lay});

            const double sig_g = 1.34;
            const double fac = std::exp(std::log(sig_g)*std::log(sig_g)) * 1e6; // Conversion to micron included.

            // CvH: Numbers according to RCEMIP.
            const double Nc0 = 100.e6;
            const double Ni0 = 1.e5;

            const double four_pi_Nc0_rho_w = 4.*M_PI*Nc0*Constants::rho_w<double>;
            const double four_pi_Ni0_rho_i = 4.*M_PI*Ni0*Constants::rho_i<double>;

            for (int ilay=1; ilay<=n_lay; ++ilay)
            {
                const double layer_mass = (p_lev({1, ilay}) - p_lev({1, ilay+1})) / Constants::grav<double>;
                for (int icol=1; icol<=n_col_in; ++icol)
                {
                    // Parametrization according to Martin et al., 1994 JAS. Fac multiplication taken from DALES.
                    // CvH: Potentially better using moments from microphysics.
                    double rel_value = cld_mask_liq({icol, ilay}) * fac *
                        std::pow(3.*(clwp_subset({icol, ilay})/layer_mass) / four_pi_Nc0_rho_w, (1./3.));

                    // Limit the values between 2.5 and 60.
                    rel_value = std::max(2.5, std::min(rel_value, 60.));
                    rel({icol, ilay}) = rel_value;

                    // Calculate the effective radius of ice from the mass and the number concentration.
                    double rei_value = cld_mask_ice({icol, ilay}) * 1.e6 *
                        std::pow(3.*(ciwp_subset({icol, ilay})/layer_mass) / four_pi_Ni0_rho_i, (1./3.));

                    // Limit the values between 2.5 and 200.
                    rei_value = std::max(2.5, std::min(rei_value, 200.));
                    rei({icol, ilay}) = rei_value;
                }
            }

            // Convert to g/m2.
            for (int i=0; i<clwp_subset.size(); ++i)
                clwp_subset.v()[i] *= 1e3;

            for (int i=0; i<ciwp_subset.size(); ++i)
                ciwp_subset.v()[i] *= 1e3;

            cloud_sw->cloud_optics(
                    cld_mask_liq, cld_mask_ice,
                    clwp_subset, ciwp_subset,
                    rel, rei,
                    *cloud_optical_props_in);

            // Add the cloud optical props to the gas optical properties.
            add_to(
                    dynamic_cast<Optical_props_2str<double>&>(*optical_props_subset_in),
                    dynamic_cast<Optical_props_2str<double>&>(*cloud_optical_props_in));
        }

        // 3. Solve the fluxes.
        Array<double,3> gpt_flux_up    ({n_col_in, n_lev, n_gpt});
        Array<double,3> gpt_flux_dn    ({n_col_in, n_lev, n_gpt});
        Array<double,3> gpt_flux_dn_dir({n_col_in, n_lev, n_gpt});

        Rte_sw<double>::rte_sw(
                optical_props_subset_in,
                top_at_1,
                mu0_subset_in,
                toa_src_subset_in,
                sfc_alb_dir_subset_in,
                sfc_alb_dif_subset_in,
                sw_flux_dn_dif_inc_subset_in,
                gpt_flux_up,
                gpt_flux_dn,
                gpt_flux_dn_dir);

        // 4. Reduce the fluxes to the needed information.
        fluxes->reduce(
                gpt_flux_up, gpt_flux_dn, gpt_flux_dn_dir,
                optical_props_subset_in, top_at_1);

        // Copy the data to the output.
        for (int ilev=1; ilev<=n_lev; ++ilev)
            for (int icol=1; icol<=n_col_in; ++icol)
            {
                flux_up    ({icol+col_s_in-1, ilev}) = fluxes->get_flux_up    ()({icol, ilev});
                flux_dn    ({icol+col_s_in-1, ilev}) = fluxes->get_flux_dn    ()({icol, ilev});
                flux_dn_dir({icol+col_s_in-1, ilev}) = fluxes->get_flux_dn_dir()({icol, ilev});
                flux_net   ({icol+col_s_in-1, ilev}) = fluxes->get_flux_net   ()({icol, ilev});
            }
    };

    for (int b=1; b<=n_blocks; ++b)
    {
        const int col_s = (b-1) * n_col_block + 1;
        const int col_e =  b    * n_col_block;

        Array<double,1> mu0_subset = mu0.subset({{ {col_s, col_e} }});
        Array<double,2> toa_src_subset = sw_flux_dn_dir_inc.subset({{ {col_s, col_e}, {1, n_gpt} }});
        Array<double,2> sfc_alb_dir_subset = sfc_alb_dir.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array<double,2> sfc_alb_dif_subset = sfc_alb_dif.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array<double,2> sw_flux_dn_dif_inc_subset = sw_flux_dn_dif_inc.subset({{ {col_s, col_e}, {1, n_gpt} }});

        std::unique_ptr<Fluxes_broadband<double>> fluxes_subset =
                std::make_unique<Fluxes_broadband<double>>(n_col_block, n_lev);

        call_kernels(
                col_s, col_e,
                optical_props_subset,
                cloud_optical_props_subset,
                mu0_subset,
                toa_src_subset,
                sfc_alb_dir_subset,
                sfc_alb_dif_subset,
                sw_flux_dn_dif_inc_subset,
                fluxes_subset);
    }

    if (n_col_block_left > 0)
    {
        const int col_s = n_col - n_col_block_left + 1;
        const int col_e = n_col;

        Array<double,1> mu0_left = mu0.subset({{ {col_s, col_e} }});
        Array<double,2> toa_src_left = sw_flux_dn_dir_inc.subset({{ {col_s, col_e}, {1, n_gpt} }});
        Array<double,2> sfc_alb_dir_left = sfc_alb_dir.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array<double,2> sfc_alb_dif_left = sfc_alb_dif.subset({{ {1, n_bnd}, {col_s, col_e} }});
        Array<double,2> sw_flux_dn_dif_inc_left = sw_flux_dn_dif_inc.subset({{ {col_s, col_e}, {1, n_gpt} }});

        std::unique_ptr<Fluxes_broadband<double>> fluxes_left =
                std::make_unique<Fluxes_broadband<double>>(n_col_block_left, n_lev);

        call_kernels(
                col_s, col_e,
                optical_props_left,
                cloud_optical_props_left,
                mu0_left,
                toa_src_left,
                sfc_alb_dir_left,
                sfc_alb_dif_left,
                sw_flux_dn_dif_inc_left,
                fluxes_left);
    }
}
template class Radiation_rrtmgp<double>;
template class Radiation_rrtmgp<float>;
