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

#include <cstdio>
#include <iostream>
#include "master.h"
#include "grid.h"
#include "fields.h"
#include "defines.h"
#include "constants.h"
#include "thermo.h"
#include "model.h"
#include "master.h"
#include "tools.h"
#include "timedep.h"
#include "monin_obukhov.h"
#include "boundary_surface.h"

namespace
{
    namespace most = Monin_obukhov;
    const int nzL = 10000; // Size of the lookup table for MO iterations.

    template<typename TF> __device__
    TF find_Obuk_g(const float* const __restrict__ zL, const float* const __restrict__ f,
                       int &n, const TF Ri, const TF zsl)
    {
        // Determine search direction.
        if ((f[n]-Ri) > 0.f)
            while ( (f[n-1]-Ri) > 0.f && n > 0) { --n; }
        else
            while ( (f[n]-Ri) < 0.f && n < (nzL-1) ) { ++n; }

        const TF zL0 = (n == 0 || n == nzL-1) ? zL[n] : zL[n-1] + (Ri-f[n-1]) / (f[n]-f[n-1]) * (zL[n]-zL[n-1]);

        return zsl/zL0;
    }


    template<typename TF> __device__
    TF calc_Obuk_noslip_flux_g(float* __restrict__ zL, float* __restrict__ f, int& n, TF du, TF bfluxbot, TF zsl)
    {
        // Calculate the appropriate Richardson number.
        const TF Ri = -Constants::kappa<TF> * bfluxbot * zsl / pow(du, TF(3));
        return find_Obuk_g(zL, f, n, Ri, zsl);
    }

    template<typename TF> __device__
    TF calc_Obuk_noslip_dirichlet_g(float* __restrict__ zL, float* __restrict__ f, int& n, TF du, TF db, TF zsl)
    {
        // Calculate the appropriate Richardson number.
        const TF Ri = Constants::kappa<TF> * db * zsl / pow(du, TF(2));
        return find_Obuk_g(zL, f, n, Ri, zsl);
    }

    /* Calculate absolute wind speed */
    template<typename TF> __global__
    void du_tot_g(TF* __restrict__ dutot,
                  TF* __restrict__ u,    TF* __restrict__ v,
                  TF* __restrict__ ubot, TF* __restrict__ vbot,
                  int istart, int jstart, int kstart,
                  int iend,   int jend, int jj, int kk)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;

        if (i < iend && j < jend)
        {
            const int ii  = 1;
            const int ij  = i + j*jj;
            const int ijk = i + j*jj + kstart*kk;
            const TF minval = 1.e-1;

            const TF du2 = pow(TF(0.5)*(u[ijk] + u[ijk+ii]) - TF(0.5)*(ubot[ij] + ubot[ij+ii]), TF(2))
                         + pow(TF(0.5)*(v[ijk] + v[ijk+jj]) - TF(0.5)*(vbot[ij] + vbot[ij+jj]), TF(2));
            dutot[ij] = fmax(pow(du2, TF(0.5)), minval);
        }
    }

    template<typename TF> __global__
    void stability_g(TF* __restrict__ ustar, TF* __restrict__ obuk,
                     TF* __restrict__ b, TF* __restrict__ bbot, TF* __restrict__ bfluxbot,
                     TF* __restrict__ dutot, float* __restrict__ zL_sl_g, float* __restrict__ f_sl_g,
                     int* __restrict__ nobuk_g,
                     TF z0m, TF z0h, TF db_ref, TF zsl,
                     int icells, int jcells, int kstart, int jj, int kk,
                     Boundary_type mbcbot, Boundary_type thermobc)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x;
        const int j = blockIdx.y*blockDim.y + threadIdx.y;

        if (i < icells && j < jcells)
        {
            const int ij  = i + j*jj;
            const int ijk = i + j*jj + kstart*kk;

            // case 1: fixed buoyancy flux and fixed ustar
            if (mbcbot == Boundary_type::Ustar_type && thermobc == Boundary_type::Flux_type)
            {
                obuk[ij] = -pow(ustar[ij], TF(3)) / (Constants::kappa<TF>*bfluxbot[ij]);
            }
            // case 2: fixed buoyancy flux and free ustar
            else if (mbcbot == Boundary_type::Dirichlet_type && thermobc == Boundary_type::Flux_type)
            {
                obuk [ij] = calc_Obuk_noslip_flux_g(zL_sl_g, f_sl_g, nobuk_g[ij], dutot[ij], bfluxbot[ij], zsl);
                ustar[ij] = dutot[ij] * most::fm(zsl, z0m, obuk[ij]);
            }
            // case 3: fixed buoyancy surface value and free ustar
            else if (mbcbot == Boundary_type::Dirichlet_type && thermobc == Boundary_type::Dirichlet_type)
            {
                TF db = b[ijk] - bbot[ij] + db_ref;
                obuk [ij] = calc_Obuk_noslip_dirichlet_g(zL_sl_g, f_sl_g, nobuk_g[ij], dutot[ij], db, zsl);
                ustar[ij] = dutot[ij] * most::fm(zsl, z0m, obuk[ij]);
            }
        }
    }

    template<typename TF> __global__
    void stability_neutral_g(TF* __restrict__ ustar, TF* __restrict__ obuk,
                             TF* __restrict__ dutot, TF z0m, TF z0h, TF zsl,
                             int icells, int jcells, int kstart, int jj, int kk,
                             Boundary_type mbcbot, Boundary_type thermobc)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x;
        const int j = blockIdx.y*blockDim.y + threadIdx.y;

        if (i < icells && j < jcells)
        {
            const int ij  = i + j*jj;

            // case 1: fixed buoyancy flux and fixed ustar
            if (mbcbot == Boundary_type::Ustar_type && thermobc == Boundary_type::Flux_type)
            {
                obuk[ij] = -Constants::dbig;
            }
            // case 2: fixed buoyancy flux and free ustar
            else if (mbcbot == Boundary_type::Dirichlet_type && thermobc == Boundary_type::Flux_type)
            {
                obuk [ij] = -Constants::dbig;
                ustar[ij] = dutot[ij] * most::fm(zsl, z0m, obuk[ij]);
            }
            // case 3: fixed buoyancy surface value and free ustar
            else if (mbcbot == Boundary_type::Dirichlet_type && thermobc == Boundary_type::Dirichlet_type)
            {
                obuk [ij] = -Constants::dbig;
                ustar[ij] = dutot[ij] * most::fm(zsl, z0m, obuk[ij]);
            }
        }
    }

    template<typename TF> __global__
    void surfm_flux_g(TF* __restrict__ ufluxbot, TF* __restrict__ vfluxbot,
                      TF* __restrict__ u,        TF* __restrict__ v,
                      TF* __restrict__ ubot,     TF* __restrict__ vbot,
                      TF* __restrict__ ustar,    TF* __restrict__ obuk,
                      TF zsl, TF z0m,
                      int istart, int jstart, int kstart,
                      int iend,   int jend, int jj, int kk,
                      Boundary_type bcbot)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x + istart;
        const int j = blockIdx.y*blockDim.y + threadIdx.y + jstart;

        if (i < iend && j < jend)
        {
            const int ii  = 1;
            const int ij  = i + j*jj;
            const int ijk = i + j*jj + kstart*kk;

            if (bcbot == Boundary_type::Dirichlet_type)
            {
                // interpolate the whole stability function rather than ustar or obuk
                ufluxbot[ij] = -(u[ijk]-ubot[ij])*TF(0.5)*(ustar[ij-ii]*most::fm(zsl, z0m, obuk[ij-ii]) + ustar[ij]*most::fm(zsl, z0m, obuk[ij]));
                vfluxbot[ij] = -(v[ijk]-vbot[ij])*TF(0.5)*(ustar[ij-jj]*most::fm(zsl, z0m, obuk[ij-jj]) + ustar[ij]*most::fm(zsl, z0m, obuk[ij]));
            }
            else if (bcbot == Boundary_type::Ustar_type)
            {
                const TF minval = 1.e-2;

                // minimize the wind at 0.01, thus the wind speed squared at 0.0001
                const TF vonu2 = fmax(minval, TF(0.25)*( pow(v[ijk-ii]-vbot[ij-ii], TF(2)) + pow(v[ijk-ii+jj]-vbot[ij-ii+jj], TF(2))
                                                       + pow(v[ijk   ]-vbot[ij   ], TF(2)) + pow(v[ijk   +jj]-vbot[ij   +jj], TF(2))) );
                const TF uonv2 = fmax(minval, TF(0.25)*( pow(u[ijk-jj]-ubot[ij-jj], TF(2)) + pow(u[ijk+ii-jj]-ubot[ij+ii-jj], TF(2))
                                                       + pow(u[ijk   ]-ubot[ij   ], TF(2)) + pow(u[ijk+ii   ]-ubot[ij+ii   ], TF(2))) );

                const TF u2 = fmax(minval, pow(u[ijk]-ubot[ij], TF(2)));
                const TF v2 = fmax(minval, pow(v[ijk]-vbot[ij], TF(2)));

                const TF ustaronu4 = TF(0.5)*(pow(ustar[ij-ii], TF(4)) + pow(ustar[ij], TF(4)));
                const TF ustaronv4 = TF(0.5)*(pow(ustar[ij-jj], TF(4)) + pow(ustar[ij], TF(4)));

                ufluxbot[ij] = -copysign(TF(1.), u[ijk]-ubot[ij]) * pow(ustaronu4 / (TF(1.) + vonu2 / u2), TF(0.5));
                vfluxbot[ij] = -copysign(TF(1.), v[ijk]-vbot[ij]) * pow(ustaronv4 / (TF(1.) + uonv2 / v2), TF(0.5));
            }
        }
    }

    template<typename TF> __global__
    void surfm_grad_g(TF* __restrict__ ugradbot, TF* __restrict__ vgradbot,
                      TF* __restrict__ u,        TF* __restrict__ v,
                      TF* __restrict__ ubot,     TF* __restrict__ vbot, TF zsl,
                      int icells, int jcells, int kstart, int jj, int kk)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x;
        const int j = blockIdx.y*blockDim.y + threadIdx.y;

        if (i < icells && j < jcells)
        {
            const int ij  = i + j*jj;
            const int ijk = i + j*jj + kstart*kk;

            ugradbot[ij] = (u[ijk]-ubot[ij])/zsl;
            vgradbot[ij] = (v[ijk]-vbot[ij])/zsl;
        }
    }

    template<typename TF> __global__
    void surfs_g(TF* __restrict__ varfluxbot, TF* __restrict__ vargradbot,
                 TF* __restrict__ varbot,     TF* __restrict__ var,
                 TF* __restrict__ ustar,      TF* __restrict__ obuk, TF zsl, TF z0h,
                 int icells, int jcells, int kstart,
                 int jj, int kk,
                 Boundary_type bcbot)
    {
        const int i = blockIdx.x*blockDim.x + threadIdx.x;
        const int j = blockIdx.y*blockDim.y + threadIdx.y;

        if (i < icells && j < jcells)
        {
            const int ij  = i + j*jj;
            const int ijk = i + j*jj + kstart*kk;

            if (bcbot == Boundary_type::Dirichlet_type)
            {
                varfluxbot[ij] = -(var[ijk]-varbot[ij])*ustar[ij]*most::fh(zsl, z0h, obuk[ij]);
                vargradbot[ij] = (var[ijk]-varbot[ij])/zsl;
            }
            else if (bcbot == Boundary_type::Flux_type)
            {
                varbot[ij]     = varfluxbot[ij] / (ustar[ij]*most::fh(zsl, z0h, obuk[ij])) + var[ijk];
                vargradbot[ij] = (var[ijk]-varbot[ij])/zsl;
            }
        }
    }
}

template<typename TF>
void Boundary_surface<TF>::prepare_device()
{
    auto& gd = grid.get_grid_data();

    const int dmemsize2d = gd.ijcells*sizeof(TF);
    const int imemsize2d = gd.ijcells*sizeof(int);
    const int dimemsize  = gd.icells*sizeof(TF);
    const int iimemsize  = gd.icells*sizeof(int);

    cuda_safe_call(cudaMalloc(&obuk_g,  dmemsize2d));
    cuda_safe_call(cudaMalloc(&ustar_g, dmemsize2d));
    cuda_safe_call(cudaMalloc(&nobuk_g, imemsize2d));

    cuda_safe_call(cudaMalloc(&zL_sl_g, nzL*sizeof(float)));
    cuda_safe_call(cudaMalloc(&f_sl_g,  nzL*sizeof(float)));

    cuda_safe_call(cudaMemcpy2D(obuk_g,  dimemsize, obuk.data(),  dimemsize, dimemsize, gd.jcells, cudaMemcpyHostToDevice));
    cuda_safe_call(cudaMemcpy2D(ustar_g, dimemsize, ustar.data(), dimemsize, dimemsize, gd.jcells, cudaMemcpyHostToDevice));
    cuda_safe_call(cudaMemcpy2D(nobuk_g, iimemsize, nobuk.data(), iimemsize, iimemsize, gd.jcells, cudaMemcpyHostToDevice));

    cuda_safe_call(cudaMemcpy(zL_sl_g, zL_sl.data(), nzL*sizeof(float), cudaMemcpyHostToDevice));
    cuda_safe_call(cudaMemcpy(f_sl_g,  f_sl.data(),  nzL*sizeof(float), cudaMemcpyHostToDevice));
}

// TMP BVS
template<typename TF>
void Boundary_surface<TF>::forward_device()
{
    auto& gd = grid.get_grid_data();

    const int dimemsize   = gd.icells  * sizeof(TF);
    const int iimemsize   = gd.icells  * sizeof(int);

    cuda_safe_call(cudaMemcpy2D(obuk_g,  dimemsize, obuk.data(),  dimemsize, dimemsize, gd.jcells, cudaMemcpyHostToDevice));
    cuda_safe_call(cudaMemcpy2D(ustar_g, dimemsize, ustar.data(), dimemsize, dimemsize, gd.jcells, cudaMemcpyHostToDevice));
    cuda_safe_call(cudaMemcpy2D(nobuk_g, iimemsize, nobuk.data(), iimemsize, iimemsize, gd.jcells, cudaMemcpyHostToDevice));
}

// TMP BVS
template<typename TF>
void Boundary_surface<TF>::backward_device()
{
    auto& gd = grid.get_grid_data();

    const int dimemsize = gd.icells * sizeof(TF);
    const int iimemsize = gd.icells * sizeof(int);

    cuda_safe_call(cudaMemcpy2D(obuk.data(),  dimemsize, obuk_g,  dimemsize, dimemsize, gd.jcells, cudaMemcpyDeviceToHost));
    cuda_safe_call(cudaMemcpy2D(ustar.data(), dimemsize, ustar_g, dimemsize, dimemsize, gd.jcells, cudaMemcpyDeviceToHost));
    cuda_safe_call(cudaMemcpy2D(nobuk.data(), iimemsize, nobuk_g, iimemsize, iimemsize, gd.jcells, cudaMemcpyDeviceToHost));
}

template<typename TF>
void Boundary_surface<TF>::clear_device()
{
    cuda_safe_call(cudaFree(obuk_g ));
    cuda_safe_call(cudaFree(ustar_g));
    cuda_safe_call(cudaFree(nobuk_g));
    cuda_safe_call(cudaFree(zL_sl_g));
    cuda_safe_call(cudaFree(f_sl_g ));
}

#ifdef USECUDA
template<typename TF>
void Boundary_surface<TF>::update_bcs(Thermo<TF>& thermo)
{
    auto& gd = grid.get_grid_data();

    const int blocki = gd.ithread_block;
    const int blockj = gd.jthread_block;

    // For 2D field excluding ghost cells
    int gridi = gd.imax/blocki + (gd.imax%blocki > 0);
    int gridj = gd.jmax/blockj + (gd.jmax%blockj > 0);
    dim3 gridGPU (gridi,  gridj,  1);
    dim3 blockGPU(blocki, blockj, 1);

    // For 2D field including ghost cells
    gridi = gd.icells/blocki + (gd.icells%blocki > 0);
    gridj = gd.jcells/blockj + (gd.jcells%blockj > 0);
    dim3 gridGPU2 (gridi,  gridj,  1);
    dim3 blockGPU2(blocki, blockj, 1);

    // Calculate dutot in tmp2
    auto dutot = fields.get_tmp_g();

    du_tot_g<<<gridGPU, blockGPU>>>(
        dutot->fld_g,
        fields.mp.at("u")->fld_g,     fields.mp.at("v")->fld_g,
        fields.mp.at("u")->fld_bot_g, fields.mp.at("v")->fld_bot_g,
        gd.istart, gd.jstart, gd.kstart,
        gd.iend, gd.jend, gd.icells, gd.ijcells);
    cuda_check_error();

    // 2D cyclic boundaries on dutot
    boundary_cyclic.exec_2d_g(dutot->fld_g);

    // start with retrieving the stability information
    if (thermo.get_switch() == "0")
    {
        // Calculate ustar and Obukhov length, including ghost cells
        stability_neutral_g<<<gridGPU2, blockGPU2>>>(
            ustar_g, obuk_g,
            dutot->fld_g, z0m, z0h, gd.z[gd.kstart],
            gd.icells, gd.jcells, gd.kstart, gd.icells, gd.ijcells, mbcbot, thermobc);
        cuda_check_error();
    }
    else
    {
        auto buoy = fields.get_tmp_g();
        thermo.get_buoyancy_surf_g(*buoy);
        const TF db_ref = thermo.get_db_ref();

        // Calculate ustar and Obukhov length, including ghost cells
        stability_g<<<gridGPU2, blockGPU2>>>(
            ustar_g, obuk_g,
            buoy->fld_g, buoy->fld_bot_g, buoy->flux_bot_g,
            dutot->fld_g, zL_sl_g, f_sl_g,
            nobuk_g,
            z0m, z0h, db_ref, gd.z[gd.kstart],
            gd.icells, gd.jcells, gd.kstart, gd.icells, gd.ijcells,
            mbcbot, thermobc);
        cuda_check_error();

        fields.release_tmp_g(buoy);
    }

    fields.release_tmp_g(dutot);

    // Calculate surface momentum fluxes, excluding ghost cells
    surfm_flux_g<<<gridGPU, blockGPU>>>(
        fields.mp.at("u")->flux_bot_g, fields.mp.at("v")->flux_bot_g,
        fields.mp.at("u")->fld_g,      fields.mp.at("v")->fld_g,
        fields.mp.at("u")->fld_bot_g,  fields.mp.at("v")->fld_bot_g,
        ustar_g, obuk_g, gd.z[gd.kstart], z0m,
        gd.istart, gd.jstart, gd.kstart,
        gd.iend, gd.jend, gd.icells, gd.ijcells, mbcbot);
    cuda_check_error();

    // 2D cyclic boundaries on the surface fluxes
    boundary_cyclic.exec_2d_g(fields.mp.at("u")->flux_bot_g);
    boundary_cyclic.exec_2d_g(fields.mp.at("v")->flux_bot_g);

    // Calculate surface gradients, including ghost cells
    surfm_grad_g<<<gridGPU2, blockGPU2>>>(
        fields.mp.at("u")->grad_bot_g, fields.mp.at("v")->grad_bot_g,
        fields.mp.at("u")->fld_g,      fields.mp.at("v")->fld_g,
        fields.mp.at("u")->fld_bot_g,  fields.mp.at("v")->fld_bot_g,
        gd.z[gd.kstart], gd.icells, gd.jcells, gd.kstart, gd.icells, gd.ijcells);
    cuda_check_error();

    // Calculate scalar fluxes, gradients and/or values, including ghost cells
    for (auto it : fields.sp)
        surfs_g<<<gridGPU2, blockGPU2>>>(
            it.second->flux_bot_g, it.second->grad_bot_g,
            it.second->fld_bot_g,  it.second->fld_g,
            ustar_g, obuk_g, gd.z[gd.kstart], z0h,
            gd.icells,  gd.jcells, gd.kstart,
            gd.icells, gd.ijcells, sbc.at(it.first).bcbot);
    cuda_check_error();
}
#endif

template class Boundary_surface<double>;
template class Boundary_surface<float>;
