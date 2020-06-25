/*
 * MicroHH
 * Copyright (c) 2011-2020 Chiel van Heerwaarden
 * Copyright (c) 2011-2020 Thijs Heus
 * Copyright (c) 2014-2020 Bart van Stratum
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

#include <cmath>
#include "fast_math.h"
#include "master.h"
#include "input.h"
#include "grid.h"
#include "fields.h"
#include "thermo.h"
#include "boundary_surface_bulk.h"
#include "constants.h"

namespace
{
    namespace fm = Fast_math;

    template<typename TF>
    void calculate_du(
            TF* restrict dutot, TF* restrict u, TF* restrict v, TF* restrict ubot, TF* restrict vbot,
            const int istart, const int iend, const int jstart, const int jend, const int kstart,
            const int jj, const int kk)
    {
        const int ii = 1;

        // calculate total wind
        TF du2;
        const TF minval = 1.e-1;
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;
                du2 = fm::pow2(0.5*(u[ijk] + u[ijk+ii]) - 0.5*(ubot[ij] + ubot[ij+ii]))
                    + fm::pow2(0.5*(v[ijk] + v[ijk+jj]) - 0.5*(vbot[ij] + vbot[ij+jj]));
                dutot[ij] = std::max(std::sqrt(du2), minval);
            }

    }

    template<typename TF>
    void momentum_fluxgrad(
            TF* restrict ufluxbot, TF* restrict vfluxbot,
            TF* restrict ugradbot, TF* restrict vgradbot,
            TF* restrict u, TF* restrict v,
            TF* restrict ubot, TF* restrict vbot,
            TF* restrict dutot, const TF Cm, const TF zsl,
            const int istart, const int iend, const int jstart, const int jend, const int kstart,
            const int jj, const int kk)
    {
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;

                ufluxbot[ij] = -Cm * dutot[ij] * (u[ijk]-ubot[ij]);
                vfluxbot[ij] = -Cm * dutot[ij] * (v[ijk]-vbot[ij]);
                ugradbot[ij] = (u[ijk]-ubot[ij])/zsl;
                vgradbot[ij] = (v[ijk]-vbot[ij])/zsl;
            }
    }

    template<typename TF>
    void scalar_fluxgrad(
            TF* restrict sfluxbot, TF* restrict sgradbot, TF* restrict s, TF* restrict sbot,
            TF* restrict dutot, const TF Cs, const TF zsl,
            const int istart, const int iend, const int jstart, const int jend, const int kstart,
            const int jj, const int kk)
    {
        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ij  = i + j*jj;
                const int ijk = i + j*jj + kstart*kk;

                sfluxbot[ij] = -Cs * dutot[ij] * (s[ijk]-sbot[ij]);
                sgradbot[ij] = (s[ijk]-sbot[ij])/zsl;
            }
    }

    template<typename TF>
    void surface_scaling(
            TF* restrict ustar, TF* restrict obuk, TF* restrict dutot, TF* restrict bfluxbot, const TF Cm,
            const int istart, const int iend, const int jstart, const int jend, const int jj)
    {

        const double sqrt_Cm = sqrt(Cm);

        for (int j=jstart; j<jend; ++j)
            #pragma ivdep
            for (int i=istart; i<iend; ++i)
            {
                const int ij  = i + j*jj;

                ustar[ij] = sqrt_Cm * dutot[ij];
                obuk[ij] = -fm::pow3(ustar[ij]) / (Constants::kappa<TF> * bfluxbot[ij]);
            }
    }
}

template<typename TF>
Boundary_surface_bulk<TF>::Boundary_surface_bulk(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Input& inputin) :
    Boundary<TF>(masterin, gridin, fieldsin, inputin)
{
    swboundary = "surface_bulk";

    #ifdef USECUDA
    ustar_g = 0;
    obuk_g  = 0;
    #endif
}

template<typename TF>
Boundary_surface_bulk<TF>::~Boundary_surface_bulk()
{
    #ifdef USECUDA
    clear_device();
    #endif
}

template<typename TF>
void Boundary_surface_bulk<TF>::create(Input& input, Netcdf_handle& input_nc, Stats<TF>& stats)
{
    const std::string group_name = "default";

    Boundary<TF>::process_time_dependent(input, input_nc);

    // add variables to the statistics
    if (stats.get_switch())
    {
        stats.add_time_series("ustar", "Surface friction velocity", "m s-1", group_name);
        stats.add_time_series("obuk", "Obukhov length", "m", group_name);
    }
}

template<typename TF>
void Boundary_surface_bulk<TF>::init(Input& inputin, Thermo<TF>& thermo)
{
    // 1. Process the boundary conditions now all fields are registered.
    process_bcs(inputin);

    // 2. Read and check the boundary_surface specific settings.
    process_input(inputin, thermo);

    // 3. Allocate and initialize the 2D surface fields.
    init_surface();

    // 4. Initialize the boundary cyclic.
    boundary_cyclic.init();
}

template<typename TF>
void Boundary_surface_bulk<TF>::process_input(Input& inputin, Thermo<TF>& thermo)
{
    z0m = inputin.get_item<TF>("boundary", "z0m", "");
    z0h = inputin.get_item<TF>("boundary", "z0h", "");

    // crash in case fixed gradient is prescribed
    if (mbcbot != Boundary_type::Dirichlet_type)
    {
        std::string msg = "Only \"noslip\" is allowed as mbcbot with swboundary=\"bulk\"";
        throw std::runtime_error(msg);
    }

    bulk_cm = inputin.get_item<TF>("boundary", "bulk_cm", "");

    // process the scalars
    for (auto& it : sbc)
    {
        if (it.second.bcbot != Boundary_type::Dirichlet_type)
        {
            std::string msg = "Only \"noslip\" is allowed as mbcbot with swboundary=\"bulk\"";
            throw std::runtime_error(msg);
        }
        bulk_cs[it.first] = inputin.get_item<TF>("boundary", "bulk_cs", it.first);
    }
}

template<typename TF>
void Boundary_surface_bulk<TF>::init_surface()
{
    auto& gd = grid.get_grid_data();

    obuk.resize(gd.ijcells);
    ustar.resize(gd.ijcells);

    const int jj = gd.icells;

    // Initialize the obukhov length on a small number.
    for (int j=0; j<gd.jcells; ++j)
        #pragma ivdep
        for (int i=0; i<gd.icells; ++i)
        {
            const int ij = i + j*jj;
            obuk[ij]  = Constants::dsmall;
        }
}

template<typename TF>
void Boundary_surface_bulk<TF>::exec_stats(Stats<TF>& stats)
{
    const TF no_offset = 0.;
    stats.calc_stats_2d("obuk", obuk, no_offset);
    stats.calc_stats_2d("ustar", ustar, no_offset);
}

template<typename TF>
void Boundary_surface_bulk<TF>::set_values()
{
    // Call the base class function.
    Boundary<TF>::set_values();
}

#ifndef USECUDA
template<typename TF>
void Boundary_surface_bulk<TF>::update_bcs(Thermo<TF>& thermo)
{
    auto& gd = grid.get_grid_data();
    const TF zsl = gd.z[gd.kstart];

    auto dutot = fields.get_tmp();

    // Calculate total wind speed difference with surface
    calculate_du(dutot->fld.data(), fields.mp.at("u")->fld.data(), fields.mp.at("v")->fld.data(),
                fields.mp.at("u")->fld_bot.data(), fields.mp.at("v")->fld_bot.data(),
                gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.icells, gd.ijcells);
    boundary_cyclic.exec_2d(dutot->fld.data());

    // Calculate surface momentum fluxes and gradients
    momentum_fluxgrad(fields.mp.at("u")->flux_bot.data(),fields.mp.at("v")->flux_bot.data(),
                    fields.mp.at("u")->grad_bot.data(),fields.mp.at("v")->grad_bot.data(),
                    fields.mp.at("u")->fld.data(),fields.mp.at("v")->fld.data(),
                    fields.mp.at("u")->fld_bot.data(),fields.mp.at("v")->fld_bot.data(),
                    dutot->fld.data(), bulk_cm, zsl,
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.icells, gd.ijcells);

    boundary_cyclic.exec_2d(fields.mp.at("u")->flux_bot.data());
    boundary_cyclic.exec_2d(fields.mp.at("v")->flux_bot.data());
    boundary_cyclic.exec_2d(fields.mp.at("u")->grad_bot.data());
    boundary_cyclic.exec_2d(fields.mp.at("v")->grad_bot.data());

    // Calculate surface scalar fluxes and gradients
    for (auto& it : fields.sp)
    {
        scalar_fluxgrad(it.second->flux_bot.data(), it.second->grad_bot.data(),
                        it.second->fld.data(), it.second->fld_bot.data(),
                        dutot->fld.data(), bulk_cs.at(it.first), zsl,
                        gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.icells, gd.ijcells);
        boundary_cyclic.exec_2d(it.second->flux_bot.data());
        boundary_cyclic.exec_2d(it.second->grad_bot.data());
    }

    auto b= fields.get_tmp();
    thermo.get_buoyancy_fluxbot(*b, false);
    surface_scaling(ustar.data(), obuk.data(), dutot->fld.data(), b->flux_bot.data(), bulk_cm,
                    gd.istart, gd.iend, gd.jstart, gd.jend, gd.icells);

    fields.release_tmp(b);
    fields.release_tmp(dutot);

}
#endif

template<typename TF>
void Boundary_surface_bulk<TF>::update_slave_bcs()
{
    // This function does nothing when the surface model is enabled, because
    // the fields are computed by the surface model in update_bcs.
}

template class Boundary_surface_bulk<double>;
template class Boundary_surface_bulk<float>;
