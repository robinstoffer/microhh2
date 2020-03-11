/*
 * MicroHH
 * Copyright (c) 2011-2018 Chiel van Heerwaarden
 * Copyright (c) 2011-2018 Thijs Heus
 * Copyright (c) 2014-2018 Bart van Stratum
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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "grid.h"
#include "fields.h"
#include "master.h"
#include "defines.h"
#include "constants.h"
#include "thermo.h"
#include "boundary.h"
#include "stats.h"

#include "diff_smag2.h"
#include "diff_nn.h"

extern "C"
{
    //#include <cblas.h>
    #include <mkl.h>
    #include <netcdf.h>
}

namespace
{
    template<typename TF>
    void diff_c(TF* restrict at, const TF* restrict a, const TF visc,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk, const TF dx, const TF dy, const TF* restrict dzi, const TF* restrict dzhi)
    {
        const int ii = 1;
        const TF dxidxi = 1/(dx*dx);
        const TF dyidyi = 1/(dy*dy);

        for (int k=kstart; k<kend; k++)
            for (int j=jstart; j<jend; j++)
                #pragma ivdep
                for (int i=istart; i<iend; i++)
                {
                    const int ijk = i + j*jj + k*kk;
                    at[ijk] += visc * (
                            + ( (a[ijk+ii] - a[ijk   ])
                              - (a[ijk   ] - a[ijk-ii]) ) * dxidxi
                            + ( (a[ijk+jj] - a[ijk   ])
                              - (a[ijk   ] - a[ijk-jj]) ) * dyidyi
                            + ( (a[ijk+kk] - a[ijk   ]) * dzhi[k+1]
                              - (a[ijk   ] - a[ijk-kk]) * dzhi[k]   ) * dzi[k] );
                }
    }

    template<typename TF>
    void diff_w(TF* restrict wt, const TF* restrict w, const TF visc,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk, const TF dx, const TF dy, const TF* restrict dzi, const TF* restrict dzhi)
    {
        const int ii = 1;
        const TF dxidxi = 1/(dx*dx);
        const TF dyidyi = 1/(dy*dy);

        for (int k=kstart+1; k<kend; k++)
            for (int j=jstart; j<jend; j++)
                #pragma ivdep
                for (int i=istart; i<iend; i++)
                {
                    const int ijk = i + j*jj + k*kk;
                    wt[ijk] += visc * (
                            + ( (w[ijk+ii] - w[ijk   ])
                              - (w[ijk   ] - w[ijk-ii]) ) * dxidxi
                            + ( (w[ijk+jj] - w[ijk   ])
                              - (w[ijk   ] - w[ijk-jj]) ) * dyidyi
                            + ( (w[ijk+kk] - w[ijk   ]) * dzi[k]
                              - (w[ijk   ] - w[ijk-kk]) * dzi[k-1] ) * dzhi[k] );
                }
    }
}


// Function that loops over the whole flow field, and calculates for each grid cell the fluxes
template<typename TF>
void Diff_NN<TF>::calc_diff_flux_u(
    TF* restrict const uflux,
    const TF* restrict const u,
    const TF* restrict const v,
    const TF* restrict const w
    )
{
    auto& gd  = grid.get_grid_data();

    // Initialize std::vectors for storing results MLP
    std::vector<float> result(N_output, 0.0f);
    std::vector<float> result_zw(N_output_zw, 0.0f);
    
    //Calculate inverse height differences
    const TF dxi = 1.f / gd.dx;
    const TF dyi = 1.f / gd.dy;

    //Loop over field
    //NOTE1: offset factors included to ensure alternate sampling
    for (int k = gd.kstart; k < gd.kend; ++k)
    {
        int k_offset = k % 2;
        for (int j = gd.jstart; j < gd.jend; ++j)
        {
            int offset = static_cast<int>((j % 2) == k_offset); //Calculate offset in such a way that the alternation swaps for each vertical level.
            for (int i = gd.istart+offset; i < gd.iend; i+=2)
            {
                //Extract grid box flow fields
                select_box(u, m_input_ctrlu_u.data(), k, j, i, boxsize, 0, 0, 0, 0, 0, 0);
                select_box(v, m_input_ctrlu_v.data(), k, j, i, boxsize, 0, 0, 1, 0, 0, 1);
                select_box(w, m_input_ctrlu_w.data(), k, j, i, boxsize, 1, 0, 0, 0, 0, 1);
                select_box(u, m_input_ctrlv_u.data(), k, j, i, boxsize, 0, 0, 0, 1, 1, 0);
                select_box(v, m_input_ctrlv_v.data(), k, j, i, boxsize, 0, 0, 0, 0, 0, 0);
                select_box(w, m_input_ctrlv_w.data(), k, j, i, boxsize, 1, 0, 0, 1, 0, 0);
                select_box(u, m_input_ctrlw_u.data(), k, j, i, boxsize, 0, 1, 0, 0, 1, 0);
                select_box(v, m_input_ctrlw_v.data(), k, j, i, boxsize, 0, 1, 1, 0, 0, 0);
                select_box(w, m_input_ctrlw_w.data(), k, j, i, boxsize, 0, 0, 0, 0, 0, 0);
                

                //Execute MLP once for selected grid box
                Inference(
                    m_input_ctrlu_u.data(), m_input_ctrlu_v.data(), m_input_ctrlu_w.data(),
                    m_hiddenu_wgth.data(), m_hiddenu_bias.data(), m_hiddenu_alpha,
                    m_outputu_wgth.data(), m_outputu_bias.data(),
                    m_input_ctrlv_u.data(), m_input_ctrlv_v.data(), m_input_ctrlv_w.data(),
                    m_hiddenv_wgth.data(), m_hiddenv_bias.data(), m_hiddenv_alpha,
                    m_outputv_wgth.data(), m_outputv_bias.data(),
                    m_input_ctrlw_u.data(), m_input_ctrlw_v.data(),  m_input_ctrlw_w.data(),
                    m_hiddenw_wgth.data(), m_hiddenw_bias.data(), m_hiddenw_alpha,
                    m_outputw_wgth.data(), m_outputw_bias.data(),
                    m_mean_input.data(), m_stdev_input.data(),
                    m_mean_label.data(), m_stdev_label.data(),
                    m_utau_ref, m_output_denorm_utau2,
                    m_output.data(), result.data(), false
                    );

                //Check whether a horizontal boundary is reached, and if so make use of horizontal periodic BCs.
                int i_upbound = 0;
                int i_downbound = 0;
                int j_upbound = 0;
                int j_downbound = 0;
                // upstream boundary
                if (i == (gd.istart))
                {
                    i_upbound = gd.iend - 1;
                }
                else
                {
                    i_upbound = i - 1;
                }
                if (j == (gd.jstart))
                {
                    j_upbound = gd.jend - 1;
                }
                else
                {
                    j_upbound = j - 1;
                }
                // downstream boundary
                if (i == (gd.iend - 1))
                {
                    i_downbound = gd.istart;
                }
                else
                {
                    i_downbound = i + 1;
                }
                if (j == (gd.jend - 1))
                {
                    j_downbound = gd.jstart;
                }
                else
                {
                    j_downbound = j + 1;
                }

                //Calculate damping factor for calculated transports
                float fac=1;//Don't impose a damping factor
                //float fac=std::min(std::min((gd.zh[k]/(0.25*gd.zh[gd.kend]))+0.1,0.3),((gd.zh[gd.kend]-gd.zh[k])/(0.25*gd.zh[gd.kend])+0.1)); //Apply damping close to the surface
                
                //Calculate tendencies using predictions from MLP
                //zu_upstream
                if (k == gd.kstart)
                {
                    uflux[k*gd.ijcells + j * gd.icells + i]     =  - (fields.visc * (u[k*gd.ijcells + j * gd.icells + i] - u[(k-1)*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k]);
                }
                else
                {
                    uflux[k*gd.ijcells + j * gd.icells + i]     =  result[4] * fac - (fields.visc * (u[k*gd.ijcells + j * gd.icells + i] - u[(k-1)*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k]);
                }

                //zu_downstream
                if (k == (gd.kend - 1))
                {
                    uflux[(k+1)*gd.ijcells + j * gd.icells + i] =  - (fields.visc * (u[(k+1)*gd.ijcells + j * gd.icells + i] - u[k*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k+1]);
                }
                else
                {
                    uflux[(k+1)*gd.ijcells + j * gd.icells + i] =  result[5] * fac - (fields.visc * (u[(k+1)*gd.ijcells + j * gd.icells + i] - u[k*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k+1]);
                }

                /////
                if (k != gd.kstart) //Don't calculate horizontal fluxes for bottom layer, should be 0
                {
                  //xw_upstream
                  uflux[k*gd.ijcells + j * gd.icells + i]         =  result[12] * fac - (fields.visc * (w[k*gd.ijcells + j * gd.icells + i] - w[k*gd.ijcells + j * gd.icells + (i-1)]) * gd.dxi);

                  //xw_downstream
                  uflux[k*gd.ijcells + j * gd.icells + i_downbound] =  result[13] * fac - (fields.visc * (w[k*gd.ijcells + j * gd.icells + (i+1)] - w[k*gd.ijcells + j * gd.icells + i]) * gd.dxi);
                }
                ////NOTE: no separate treatment for walls needed since w should be 0 at the top and bottom wall (and thus there are no horizontal gradients and horizontal fluxes)
 
                // Calculate for each iteration in the bottom layer, and for each iteration in the top layer, 
                // the resolved transport for a second grid cell to calculate 'missing' values due to alternation.
                if ((k == (gd.kend - 1)) || (k == (gd.kstart)))
                {
                    //Determine the second grid cell based on the offset.
                    int i_2grid = 0;
                    if (offset == 1)
                    {
                        i_2grid = i - 1;
                    }
                    else
                    {
                        i_2grid = i + 1;
                    }
                
                    //Calculate resolved fluxes
                    //zu_upstream
                    if (k == gd.kstart)
                    {
                        uflux[k*gd.ijcells + j * gd.icells + i_2grid]     =  - (fields.visc * (u[k*gd.ijcells + j * gd.icells + i_2grid] - u[(k-1)*gd.ijcells + j * gd.icells + i_2grid]) * gd.dzhi[k]);
                    }
                    //zu_downstream
                    else if (k == (gd.kend - 1))
                    {                        
                        uflux[(k+1)*gd.ijcells + j * gd.icells + i_2grid] =  - (fields.visc * (u[(k+1)*gd.ijcells + j * gd.icells + i_2grid] - u[k*gd.ijcells + j * gd.icells + i_2grid]) * gd.dzhi[k+1]);
                    }
                }
            }
        }
    }
}

template<typename TF>
void Diff_NN<TF>::calc_diff_flux_v(
    TF* restrict const vflux,
    const TF* restrict const u,
    const TF* restrict const v,
    const TF* restrict const w
    )
{
    auto& gd  = grid.get_grid_data();

    // Initialize std::vectors for storing results MLP
    std::vector<float> result(N_output, 0.0f);
    std::vector<float> result_zw(N_output_zw, 0.0f);
    
    //Calculate inverse height differences
    const TF dxi = 1.f / gd.dx;
    const TF dyi = 1.f / gd.dy;

    //Loop over field
    //NOTE1: offset factors included to ensure alternate sampling
    for (int k = gd.kstart; k < gd.kend; ++k)
    {
        int k_offset = k % 2;
        for (int j = gd.jstart; j < gd.jend; ++j)
        {
            int offset = static_cast<int>((j % 2) == k_offset); //Calculate offset in such a way that the alternation swaps for each vertical level.
            for (int i = gd.istart+offset; i < gd.iend; i+=2)
            {
                //Extract grid box flow fields
                select_box(u, m_input_ctrlu_u.data(), k, j, i, boxsize, 0, 0, 0, 0, 0, 0);
                select_box(v, m_input_ctrlu_v.data(), k, j, i, boxsize, 0, 0, 1, 0, 0, 1);
                select_box(w, m_input_ctrlu_w.data(), k, j, i, boxsize, 1, 0, 0, 0, 0, 1);
                select_box(u, m_input_ctrlv_u.data(), k, j, i, boxsize, 0, 0, 0, 1, 1, 0);
                select_box(v, m_input_ctrlv_v.data(), k, j, i, boxsize, 0, 0, 0, 0, 0, 0);
                select_box(w, m_input_ctrlv_w.data(), k, j, i, boxsize, 1, 0, 0, 1, 0, 0);
                select_box(u, m_input_ctrlw_u.data(), k, j, i, boxsize, 0, 1, 0, 0, 1, 0);
                select_box(v, m_input_ctrlw_v.data(), k, j, i, boxsize, 0, 1, 1, 0, 0, 0);
                select_box(w, m_input_ctrlw_w.data(), k, j, i, boxsize, 0, 0, 0, 0, 0, 0);
                

                //Execute MLP once for selected grid box
                Inference(
                    m_input_ctrlu_u.data(), m_input_ctrlu_v.data(), m_input_ctrlu_w.data(),
                    m_hiddenu_wgth.data(), m_hiddenu_bias.data(), m_hiddenu_alpha,
                    m_outputu_wgth.data(), m_outputu_bias.data(),
                    m_input_ctrlv_u.data(), m_input_ctrlv_v.data(), m_input_ctrlv_w.data(),
                    m_hiddenv_wgth.data(), m_hiddenv_bias.data(), m_hiddenv_alpha,
                    m_outputv_wgth.data(), m_outputv_bias.data(),
                    m_input_ctrlw_u.data(), m_input_ctrlw_v.data(),  m_input_ctrlw_w.data(),
                    m_hiddenw_wgth.data(), m_hiddenw_bias.data(), m_hiddenw_alpha,
                    m_outputw_wgth.data(), m_outputw_bias.data(),
                    m_mean_input.data(), m_stdev_input.data(),
                    m_mean_label.data(), m_stdev_label.data(),
                    m_utau_ref, m_output_denorm_utau2,
                    m_output.data(), result.data(), false
                    );

                //Check whether a horizontal boundary is reached, and if so make use of horizontal periodic BCs.
                int i_upbound = 0;
                int i_downbound = 0;
                int j_upbound = 0;
                int j_downbound = 0;
                // upstream boundary
                if (i == (gd.istart))
                {
                    i_upbound = gd.iend - 1;
                }
                else
                {
                    i_upbound = i - 1;
                }
                if (j == (gd.jstart))
                {
                    j_upbound = gd.jend - 1;
                }
                else
                {
                    j_upbound = j - 1;
                }
                // downstream boundary
                if (i == (gd.iend - 1))
                {
                    i_downbound = gd.istart;
                }
                else
                {
                    i_downbound = i + 1;
                }
                if (j == (gd.jend - 1))
                {
                    j_downbound = gd.jstart;
                }
                else
                {
                    j_downbound = j + 1;
                }

                //Calculate damping factor for calculated transports
                float fac=1;//Don't impose a damping factor
                //float fac=std::min(std::min((gd.zh[k]/(0.25*gd.zh[gd.kend]))+0.1,0.3),((gd.zh[gd.kend]-gd.zh[k])/(0.25*gd.zh[gd.kend])+0.1)); //Apply damping close to the surface
                
                //Calculate tendencies using predictions from MLP
                
                //Additional testing code in block below!
                /////////////////////////////////
                ////xu
                //vflux[k*gd.ijcells + j * gd.icells + i]           =  result[0] * fac;
                //vflux[k*gd.ijcells + j * gd.icells + i_downbound] =  result[1] * fac ;
                ////xv
                //vflux[k*gd.ijcells + j * gd.icells + i]           =  result[6] * fac;
                //vflux[k*gd.ijcells + j * gd.icells + i_downbound] =  result[7] * fac ;
                ////yu
                //vflux[k*gd.ijcells + j * gd.icells + i]           =  result[2] * fac;
                //vflux[k*gd.ijcells + j_downbound * gd.icells + i] =  result[3] * fac ;
                ////yv
                //vflux[k*gd.ijcells + j * gd.icells + i]           = result[8] * fac;
                //vflux[k*gd.ijcells + j_downbound * gd.icells + i] = result[9] * fac;
                ////xw
                //if (k != gd.kstart) //Don't calculate horizontal fluxes for bottom layer, should be 0
                //{
                //  //xw_upstream
                //  vflux[k*gd.ijcells + j * gd.icells + i]         =  result[12] * fac;

                //  //xw_downstream
                //  vflux[k*gd.ijcells + j * gd.icells + i_downbound] =  result[13] * fac;
                //}
                ////yw
                //if (k != gd.kstart) //Don't calculate horizontal fluxes for bottom layer, should be 0
                //{
                //  //yw_upstream
                //  vflux[k*gd.ijcells + j * gd.icells + i]         =  result[14] * fac;

                //  //yw_downstream
                //  vflux[k*gd.ijcells + j_downbound * gd.icells + i] =  result[15] * fac;
                //}
                ////zw
                //if (k > gd.kstart)
                //{
                //    //zw_upstream
                //    vflux[k*gd.ijcells + j * gd.icells + i]           =  result[16] * fac;

                //    //zw_downstream
                //    if (k != (gd.kend - 1))
                //    {
                //        vflux[(k+1)*gd.ijcells + j * gd.icells + i] = result[17] * fac;
                //    }
                //}
                //// Calculate for each iteration in the first layer above the bottom layer, and for each iteration in the top layer, the resolved transport for a second grid cell to calculate 'missing' values zw due to alternation.
                //if (k == (gd.kstart+1)) //|| ((k == (gd.kend - 1))
                //{
                //    //Determine the second grid cell based on the offset.
                //    int i_2grid = 0;
                //    if (offset == 1)
                //    {
                //        i_2grid = i - 1;
                //    }
                //    else
                //    {
                //        i_2grid = i + 1;
                //    }

                //    //Select second grid box
                //    select_box(u, m_input_ctrlu_u.data(), k, j, i_2grid, boxsize, 0, 0, 0, 0, 0, 0);
                //    select_box(v, m_input_ctrlu_v.data(), k, j, i_2grid, boxsize, 0, 0, 1, 0, 0, 1);
                //    select_box(w, m_input_ctrlu_w.data(), k, j, i_2grid, boxsize, 1, 0, 0, 0, 0, 1);
                //    select_box(u, m_input_ctrlv_u.data(), k, j, i_2grid, boxsize, 0, 0, 0, 1, 1, 0);
                //    select_box(v, m_input_ctrlv_v.data(), k, j, i_2grid, boxsize, 0, 0, 0, 0, 0, 0);
                //    select_box(w, m_input_ctrlv_w.data(), k, j, i_2grid, boxsize, 1, 0, 0, 1, 0, 0);
                //    select_box(u, m_input_ctrlw_u.data(), k, j, i_2grid, boxsize, 0, 1, 0, 0, 1, 0);
                //    select_box(v, m_input_ctrlw_v.data(), k, j, i_2grid, boxsize, 0, 1, 1, 0, 0, 0);
                //    select_box(w, m_input_ctrlw_w.data(), k, j, i_2grid, boxsize, 0, 0, 0, 0, 0, 0);

                //    //Execute mlp for selected second grid cell
                //    Inference(
                //        m_input_ctrlu_u.data(), m_input_ctrlu_v.data(), m_input_ctrlu_w.data(),
                //        m_hiddenu_wgth.data(), m_hiddenu_bias.data(), m_hiddenu_alpha,
                //        m_outputu_wgth.data(), m_outputu_bias.data(),
                //        m_input_ctrlv_u.data(), m_input_ctrlv_v.data(), m_input_ctrlv_w.data(),
                //        m_hiddenv_wgth.data(), m_hiddenv_bias.data(), m_hiddenv_alpha,
                //        m_outputv_wgth.data(), m_outputv_bias.data(),
                //        m_input_ctrlw_u.data(), m_input_ctrlw_v.data(), m_input_ctrlw_w.data(),
                //        m_hiddenw_wgth.data(), m_hiddenw_bias.data(), m_hiddenw_alpha,
                //        m_outputw_wgth.data(), m_outputw_bias.data(),
                //        m_mean_input.data(), m_stdev_input.data(),
                //        m_mean_label.data(), m_stdev_label.data(),
                //        m_utau_ref, m_output_denorm_utau2,
                //        m_output_zw.data(), result_zw.data(), true
                //    );
                //    //Store fluxes
                //    //zw_upstream
                //    if (k == (gd.kstart+1))
                //    {
                //        vflux[k*gd.ijcells + j * gd.icells + i_2grid]     =  result_zw[0]  * fac;
                //    }
                //}
                /////////////////////////////////
                //zv_upstream
                if (k == gd.kstart)
                {
                    vflux[k*gd.ijcells + j * gd.icells + i]     =  - (fields.visc * (v[k*gd.ijcells + j * gd.icells + i] - v[(k-1)*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k]);
                }
                else
                {
                    vflux[k*gd.ijcells + j * gd.icells + i]     =  result[10]  * fac - (fields.visc * (v[k*gd.ijcells + j * gd.icells + i] - v[(k-1)*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k]);
                }

                //zv_downstream
                if (k == (gd.kend - 1))
                {
                    vflux[(k+1)*gd.ijcells + j * gd.icells + i] =  - (fields.visc * (v[(k+1)*gd.ijcells + j * gd.icells + i] - v[k*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k+1]);
                }
                else
                {
                    vflux[(k+1)*gd.ijcells + j * gd.icells + i] =  result[11] * fac - (fields.visc * (v[(k+1)*gd.ijcells + j * gd.icells + i] - v[k*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k+1]);
                }

                /////
                if (k != gd.kstart) //Don't calculate horizontal fluxes for bottom layer, should be 0
                {
                    //yw_upstream
                    vflux[k*gd.ijcells + j * gd.icells + i]         =  result[14] * fac - (fields.visc * (w[k*gd.ijcells + j * gd.icells + i] - w[k*gd.ijcells + j * gd.icells + (i-1)]) * gd.dyi);

                    //yw_downstream
                    vflux[k*gd.ijcells + j * gd.icells + i_downbound] =  result[15] * fac - (fields.visc * (w[k*gd.ijcells + j * gd.icells + (i+1)] - w[k*gd.ijcells + j * gd.icells + i]) * gd.dyi);
                }
                //NOTE: no separate treatment for walls needed since w should be 0 at the top and bottom wall (and thus there are no horizontal gradients and horizontal fluxes)
                
                // Calculate for each iteration in the bottom layer, and for each iteration in the top layer, 
                // the resolved transport for a second grid cell to calculate 'missing' values due to alternation.
                if ((k == (gd.kend - 1)) || (k == (gd.kstart)))
                {
                    //Determine the second grid cell based on the offset.
                    int i_2grid = 0;
                    if (offset == 1)
                    {
                        i_2grid = i - 1;
                    }
                    else
                    {
                        i_2grid = i + 1;
                    }
                
                    //Calculate resolved fluxes
                    //zv_upstream
                    if (k == gd.kstart)
                    {
                        vflux[k*gd.ijcells + j * gd.icells + i_2grid]     =  - (fields.visc * (v[k*gd.ijcells + j * gd.icells + i_2grid] - v[(k-1)*gd.ijcells + j * gd.icells + i_2grid]) * gd.dzhi[k]);
                    }
                    //zv_downstream
                    else if (k == (gd.kend - 1))
                    {                        
                        vflux[(k+1)*gd.ijcells + j * gd.icells + i_2grid] =  - (fields.visc * (v[(k+1)*gd.ijcells + j * gd.icells + i_2grid] - v[k*gd.ijcells + j * gd.icells + i_2grid]) * gd.dzhi[k+1]);
                    }
                }
            }
        }
    }
}

template<typename TF>
void Diff_NN<TF>::select_box(
    const TF* restrict const field_var,
    float* restrict const box_var,
    const int k_center,
    const int j_center,
    const int i_center,
    const int boxsize,
    const int skip_firstz,
    const int skip_lastz,
    const int skip_firsty,
    const int skip_lasty,
    const int skip_firstx,
    const int skip_lastx
    )
// NOTE: the skip_* integers specify whether the index indicated in the name should be skipped in the selection of box (0=don't skip, 1=skip it).
{
    auto& gd = grid.get_grid_data();

    // Calculate number of grid cells that the grid box extends from the center for looping
    int b = boxsize / 2; // NOTE: on purpose fractional part dropped
    //Loop over all three indices to extract grid box
    int ji_box  = (boxsize - skip_firsty - skip_lasty) * (boxsize - skip_firstx - skip_lastx);
    int k_box = 0;
    for (int k_field = k_center - b + skip_firstz; k_field < (k_center + b + 1 - skip_lastz); ++k_field)
    {
        int j_box = 0;
        for (int j_field = j_center - b + skip_firsty; j_field < (j_center + b + 1 - skip_lasty); ++j_field)
        {
            int i_box = 0;
            for (int i_field = i_center - b + skip_firstx; i_field < (i_center + b + 1 - skip_lastx); ++i_field)
            {
                //Extract grid box flow field
                box_var[k_box * ji_box + j_box * (boxsize - skip_firstx - skip_lastx) + i_box] = static_cast<float>(field_var[k_field * gd.ijcells + j_field * gd.icells + i_field]);
                i_box += 1;
            }
            j_box += 1;
        }
        k_box += 1;
    }
}

//Function that creates Gaussian 2D kernel for smoothing of horizontal tendency fields
template<typename TF>
void Diff_NN<TF>::gkernelcreation(
   TF* restrict const gkernel
   )
{
    //Initialise standard deviation
    TF sigma = 1.0;
    TF r,s   = 2.0 * sigma * sigma;

    //Sum for normalisation
    TF sum = 0.0;

    //Generating 5*5 kernel (assuming an equidistant horizontal grid!)
    for (int j = -2; j < 3; ++j)
    {
        for (int i = -2; i <3; ++i)
        {
            r = pow((i*i+j*j),0.5);
            gkernel[(j+2)*5 + (i+2)] = exp(-r*r/s)/(M_PI*s);
            sum += gkernel[(j+2)*5 + (i+2)];
        }
    }

    //Normalizing the 5*5 kernel
    for (int j = 0; j < 5; ++j)
    {
        for (int i = 0; i < 5; ++i)
        {
            gkernel[j*5 + i] /= sum;
        }
    }
}

// Function that loops over the whole flow field, and calculates for each grid cell the tendencies
template<typename TF>
void Diff_NN<TF>::diff_U(
    TF* restrict const u,
    TF* restrict const v,
    TF* restrict const w,
    TF* restrict const ut,
    TF* restrict const vt,
    TF* restrict const wt
    )
{
    auto& gd  = grid.get_grid_data();

    // Initialize std::vectors for storing results mlp
    std::vector<float> result(N_output, 0.0f);
    std::vector<float> result_zw(N_output_zw, 0.0f);
    
    //Calculate inverse height differences
    const TF dxi = 1.f / gd.dx;
    const TF dyi = 1.f / gd.dy;

    //Set counters to track how many values are set to 0.
    int limit_count_xuup = 0;
    int limit_count_xudown = 0;
    int limit_count_yuup = 0;
    int limit_count_yudown = 0;
    int limit_count_zuup = 0;
    int limit_count_zudown = 0;
    int limit_count_xvup = 0;
    int limit_count_xvdown = 0;
    int limit_count_yvup = 0;
    int limit_count_yvdown = 0;
    int limit_count_zvup = 0;
    int limit_count_zvdown = 0;
    int limit_count_xwup = 0;
    int limit_count_xwdown = 0;
    int limit_count_ywup = 0;
    int limit_count_ywdown = 0;
    int limit_count_zwup = 0;
    int limit_count_zwdown = 0;

    ////Calculate Gaussian 2D filter
    //std::array<TF,25> gkernel;
    //gkernelcreation(gkernel.data());

    ////Loop over field to limit velocity fields
    //for (int k = 0; k < gd.kcells; ++k)
    //{
    //    for (int j = 0; j < gd.jcells; ++j)
    //    {
    //        for (int i = 0; i < gd.icells; ++i)
    //        {
    //            //Limit high values
    //            u[k*gd.ijcells + j * gd.icells + i] = std::min(u[k*gd.ijcells + j * gd.icells + i], static_cast<TF>(m_ucmean[k] + 2*m_ucstd[k]));
    //            v[k*gd.ijcells + j * gd.icells + i] = std::min(v[k*gd.ijcells + j * gd.icells + i], static_cast<TF>(m_vcmean[k] + 2*m_vcstd[k]));
    //            w[k*gd.ijcells + j * gd.icells + i] = std::min(w[k*gd.ijcells + j * gd.icells + i], static_cast<TF>(m_wcmean[k] + 2*m_wcstd[k]));
    //            //Limit low values
    //            u[k*gd.ijcells + j * gd.icells + i] = std::max(u[k*gd.ijcells + j * gd.icells + i], static_cast<TF>(m_ucmean[k] - 2*m_ucstd[k]));
    //            v[k*gd.ijcells + j * gd.icells + i] = std::max(v[k*gd.ijcells + j * gd.icells + i], static_cast<TF>(m_vcmean[k] - 2*m_vcstd[k]));
    //            w[k*gd.ijcells + j * gd.icells + i] = std::max(w[k*gd.ijcells + j * gd.icells + i], static_cast<TF>(m_wcmean[k] - 2*m_wcstd[k]));
    //        }
    //    }
    //}

    //Loop over field
    //NOTE1: offset factors included to ensure alternate sampling
    for (int k = gd.kstart; k < gd.kend; ++k)
    {
        int k_offset = k % 2;
        for (int j = gd.jstart; j < gd.jend; ++j)
        {
            int offset = static_cast<int>((j % 2) == k_offset); //Calculate offset in such a way that the alternation swaps for each vertical level.
            for (int i = gd.istart+offset; i < gd.iend; i+=2)
            {
                //Extract grid box flow fields
                select_box(u, m_input_ctrlu_u.data(), k, j, i, boxsize, 0, 0, 0, 0, 0, 0);
                select_box(v, m_input_ctrlu_v.data(), k, j, i, boxsize, 0, 0, 1, 0, 0, 1);
                select_box(w, m_input_ctrlu_w.data(), k, j, i, boxsize, 1, 0, 0, 0, 0, 1);
                select_box(u, m_input_ctrlv_u.data(), k, j, i, boxsize, 0, 0, 0, 1, 1, 0);
                select_box(v, m_input_ctrlv_v.data(), k, j, i, boxsize, 0, 0, 0, 0, 0, 0);
                select_box(w, m_input_ctrlv_w.data(), k, j, i, boxsize, 1, 0, 0, 1, 0, 0);
                select_box(u, m_input_ctrlw_u.data(), k, j, i, boxsize, 0, 1, 0, 0, 1, 0);
                select_box(v, m_input_ctrlw_v.data(), k, j, i, boxsize, 0, 1, 1, 0, 0, 0);
                select_box(w, m_input_ctrlw_w.data(), k, j, i, boxsize, 0, 0, 0, 0, 0, 0);
                
                ////Implement limiter on inputs, 4 separate loops needed because of differences in dims
                //int b = boxsize / 2; // NOTE: on purpose fractional part dropped
                //int i_start = 0;
                //for (int k_input = (k - b); k_input < (k + b + 1); k_input+=1)
                //{
                //    for (int i_input = i_start; i_input < (i_start+(boxsize-1)*(boxsize-1)); i_input+=1)
                //    {
                //        //Limit highest values
                //        m_input_ctrlu_v[i_input] = std::min(m_input_ctrlu_v[i_input], m_vcmean[k_input] + m_vcstd[k_input]);
                //        m_input_ctrlv_u[i_input] = std::min(m_input_ctrlv_u[i_input], m_ucmean[k_input] + m_ucstd[k_input]);

                //        //Limit lowest values
                //        m_input_ctrlu_v[i_input] = std::max(m_input_ctrlu_v[i_input], m_vcmean[k_input] - m_vcstd[k_input]);
                //        m_input_ctrlv_u[i_input] = std::max(m_input_ctrlv_u[i_input], m_ucmean[k_input] - m_ucstd[k_input]);
                //    }
                //    i_start += ((boxsize-1)*(boxsize-1));
                //}

                //i_start = 0;
                //for (int k_input = (k - b + 1); k_input < (k + b + 1); k_input+=1)
                //{
                //    for (int i_input = i_start; i_input < (i_start+(boxsize)*(boxsize-1)); i_input+=1)
                //    {
                //        //Limit highest values
                //        m_input_ctrlu_w[i_input] = std::min(m_input_ctrlu_w[i_input], m_wcmean[k_input] + m_wcstd[k_input]);
                //        m_input_ctrlv_w[i_input] = std::min(m_input_ctrlv_w[i_input], m_wcmean[k_input] + m_wcstd[k_input]);

                //        //Limit lowest values
                //        m_input_ctrlu_w[i_input] = std::max(m_input_ctrlu_w[i_input], m_wcmean[k_input] - m_wcstd[k_input]);
                //        m_input_ctrlv_w[i_input] = std::max(m_input_ctrlv_w[i_input], m_wcmean[k_input] - m_wcstd[k_input]);
                //    }
                //    i_start += ((boxsize-1)*(boxsize));
                //}

                //i_start = 0;
                //for (int k_input = (k - b); k_input < (k + b); k_input+=1)
                //{
                //    for (int i_input = i_start; i_input < (i_start+(boxsize-1)*(boxsize)); i_input+=1)
                //    {
                //        //Limit highest values
                //        m_input_ctrlw_u[i_input] = std::min(m_input_ctrlw_u[i_input], m_ucmean[k_input] + m_ucstd[k_input]);
                //        m_input_ctrlw_v[i_input] = std::min(m_input_ctrlw_v[i_input], m_vcmean[k_input] + m_vcstd[k_input]);
                //    
                //        //Limit lowest values
                //        m_input_ctrlw_u[i_input] = std::max(m_input_ctrlw_u[i_input], m_ucmean[k_input] - m_ucstd[k_input]);
                //        m_input_ctrlw_v[i_input] = std::max(m_input_ctrlw_v[i_input], m_vcmean[k_input] - m_vcstd[k_input]);
                //    }
                //    i_start += ((boxsize-1)*(boxsize));
                //}

                //i_start = 0;
                //for (int k_input = (k - b); k_input < (k + b + 1); k_input+=1)
                //{
                //    for (int i_input = i_start; i_input < (i_start+(boxsize)*(boxsize)); i_input+=1)
                //    {
                //        //Limit highest values
                //        m_input_ctrlu_u[i_input] = std::min(m_input_ctrlu_u[i_input], m_ucmean[k_input] + m_ucstd[k_input]);
                //        m_input_ctrlv_v[i_input] = std::min(m_input_ctrlv_v[i_input], m_vcmean[k_input] + m_vcstd[k_input]);
                //        m_input_ctrlw_w[i_input] = std::min(m_input_ctrlw_w[i_input], m_wcmean[k_input] + m_wcstd[k_input]);
                //        
                //        //Limit lowest values
                //        m_input_ctrlu_u[i_input] = std::max(m_input_ctrlu_u[i_input], m_ucmean[k_input] - m_ucstd[k_input]);
                //        m_input_ctrlv_v[i_input] = std::max(m_input_ctrlv_v[i_input], m_vcmean[k_input] - m_vcstd[k_input]);
                //        m_input_ctrlw_w[i_input] = std::max(m_input_ctrlw_w[i_input], m_wcmean[k_input] - m_wcstd[k_input]);
                //    }
                //    i_start += ((boxsize)*(boxsize));
                //}
                
                //Execute MLP once for selected grid box
                Inference(
                    m_input_ctrlu_u.data(), m_input_ctrlu_v.data(), m_input_ctrlu_w.data(),
                    m_hiddenu_wgth.data(), m_hiddenu_bias.data(), m_hiddenu_alpha,
                    m_outputu_wgth.data(), m_outputu_bias.data(),
                    m_input_ctrlv_u.data(), m_input_ctrlv_v.data(), m_input_ctrlv_w.data(),
                    m_hiddenv_wgth.data(), m_hiddenv_bias.data(), m_hiddenv_alpha,
                    m_outputv_wgth.data(), m_outputv_bias.data(),
                    m_input_ctrlw_u.data(), m_input_ctrlw_v.data(),  m_input_ctrlw_w.data(),
                    m_hiddenw_wgth.data(), m_hiddenw_bias.data(), m_hiddenw_alpha,
                    m_outputw_wgth.data(), m_outputw_bias.data(),
                    m_mean_input.data(), m_stdev_input.data(),
                    m_mean_label.data(), m_stdev_label.data(),
                    m_utau_ref, m_output_denorm_utau2,
                    m_output.data(), result.data(), false
                    );

                //Implement backscatter limiter on predicted transports (i.e. -tau_ij * filtS_ij < 0)
                //NOTE: on some vertical levels fluxes are not set to zero, in correspondence with the if-statements applied in the tendency calculation below
                
                //Introduce backscattering limiting factor, which determines how much backscattering remains (purpose: show LES can handle some backscattering, as long as the total dissipation is large enough to balance the production)
                TF backscatter_limit_fac = 0.5; // Observed to introduce numerical instability in Moser case channel flow (friction Reynolds number = 590)
                //TF backscatter_limit_fac = 0.9; // Observed to introduce numerical instability in Moser case channel flow (friction Reynolds number = 590)
                //TF backscatter_limit_fac = 0.2; // Observed to achieve numerical stability (but with already some smaller indications of instability eventually) in Moser case channel flow (friction Reynolds number = 590)
                //backscatter_limit_fac = 0.; // Observed to achieve numerical stability in Moser case channel flow (friction Reynolds number = 590)

                //xu_upstream
                if ((-result[0] * (u[k*gd.ijcells + j * gd.icells + i] - u[k*gd.ijcells + j * gd.icells + (i-1)]) * dxi) < 0.)
                {
                    result[0] = backscatter_limit_fac * result[0];
                    limit_count_xuup += 1;
                }
                //xu_downstream
                if ((-result[1] * (u[k*gd.ijcells + j * gd.icells + (i+1)] - u[k*gd.ijcells + j * gd.icells + i]) * dxi) < 0.)
                {
                    result[1] = backscatter_limit_fac * result[1];
                    limit_count_xudown += 1;
                }
                //yu_upstream
                if ((-result[2] * 0.5 * (((v[k*gd.ijcells + j * gd.icells + i] - v[k*gd.ijcells + j * gd.icells + (i-1)]) * dxi) + ((u[k*gd.ijcells + j * gd.icells + i] - u[k*gd.ijcells + (j-1) * gd.icells + i]) * dyi))) < 0.)
                {
                    result[2] = backscatter_limit_fac * result[2];
                    limit_count_yuup += 1;
                }
                //yu_downstream
                if ((-result[3] * 0.5 * (((v[k*gd.ijcells + (j+1) * gd.icells + i] - v[k*gd.ijcells + (j+1) * gd.icells + (i-1)]) * dxi) + ((u[k*gd.ijcells + (j+1) * gd.icells + i] - u[k*gd.ijcells + j * gd.icells + i]) * dyi))) < 0.)
                {
                    result[3] = backscatter_limit_fac * result[3];
                    limit_count_yudown += 1;
                }
                //zu_upstream
                if ((k != gd.kstart) and (-result[4] * 0.5 * (((u[k*gd.ijcells + j * gd.icells + i] - u[(k-1)*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k]) + ((w[k*gd.ijcells + j * gd.icells + i] - w[k*gd.ijcells + j * gd.icells + (i-1)]) * dxi))) < 0.)
                {
                    result[4] = backscatter_limit_fac * result[4];
                    limit_count_zuup += 1;
                }
                //zu_downstream
                if ((k != (gd.kend - 1)) and (-result[5] * 0.5 * (((u[(k+1)*gd.ijcells + j * gd.icells + i] - u[k*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k+1]) + ((w[(k+1)*gd.ijcells + j * gd.icells + i] - w[(k+1)*gd.ijcells + j * gd.icells + (i-1)]) * dxi))) < 0.)
                {
                    result[5] = backscatter_limit_fac * result[5];
                    limit_count_zudown += 1;
                }
                
                //xv_upstream
                if ((-result[6] * 0.5 * (((v[k*gd.ijcells + j * gd.icells + i] - v[k*gd.ijcells + j * gd.icells + (i-1)]) * dxi) + ((u[k*gd.ijcells + j * gd.icells + i] - u[k*gd.ijcells + (j-1) * gd.icells + i]) * dyi))) < 0.)
                {
                    result[6] = backscatter_limit_fac * result[6];
                    limit_count_xvup += 1;
                }
                //xv_downstream
                if ((-result[7] * 0.5 * (((v[k*gd.ijcells + j * gd.icells + (i+1)] - v[k*gd.ijcells + j * gd.icells + i]) * dxi) + ((u[k*gd.ijcells + j * gd.icells + (i+1)] - u[k*gd.ijcells + (j-1) * gd.icells + (i+1)]) * dyi))) < 0.)
                {
                    result[7] = backscatter_limit_fac * result[7];
                    limit_count_xvdown += 1;
                }
                
                //yv_upstream
                if ((-result[8] * (v[k*gd.ijcells + j * gd.icells + i] - v[k*gd.ijcells + (j-1) * gd.icells + i]) * dyi) < 0.)
                {
                    result[8] = backscatter_limit_fac * result[8];
                    limit_count_yvup += 1;
                }
                //yv_downstream
                if ((-result[9] * (v[k*gd.ijcells + (j+1) * gd.icells + i] - v[k*gd.ijcells + j * gd.icells + i]) * dyi) < 0.)
                {
                    result[9] = backscatter_limit_fac * result[9];
                    limit_count_yvdown += 1;
                }
                //zv_upstream
                if ((k != gd.kstart) and (-result[10] * 0.5 * (((v[k*gd.ijcells + j * gd.icells + i] - v[(k-1)*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k]) + ((w[k*gd.ijcells + j * gd.icells + i] - w[k*gd.ijcells + (j-1) * gd.icells + i]) * dyi))) < 0.)
                {
                    result[10] = backscatter_limit_fac * result[10];
                    limit_count_zvup += 1;
                }
                //zv_downstream
                if ((k != (gd.kend - 1)) and (-result[11] * 0.5 * (((v[(k+1)*gd.ijcells + j * gd.icells + i] - v[k*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k+1]) + ((w[(k+1)*gd.ijcells + j * gd.icells + i] - w[(k+1)*gd.ijcells + (j-1) * gd.icells + i]) * dyi))) < 0.)
                {
                    result[11] = backscatter_limit_fac * result[11];
                    limit_count_zvdown += 1;
                }

                if (k != gd.kstart)
                {
                    //xw_upstream
                    if ((-result[12] * 0.5 * (((u[k*gd.ijcells + j * gd.icells + i] - u[(k-1)*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k]) + ((w[k*gd.ijcells + j * gd.icells + i] - w[k*gd.ijcells + j * gd.icells + (i-1)]) * dxi))) < 0.)
                    {
                        result[12] = backscatter_limit_fac * result[12];
                        limit_count_xwup += 1;
                    }
                    //xw_downstream
                    if ((-result[13] * 0.5 * (((u[k*gd.ijcells + j * gd.icells + (i+1)] - u[(k-1)*gd.ijcells + j * gd.icells + (i+1)]) * gd.dzhi[k]) + ((w[k*gd.ijcells + j * gd.icells + (i+1)] - w[k*gd.ijcells + j * gd.icells + i]) * dxi))) < 0.)
                    {
                        result[13] = backscatter_limit_fac * result[13];
                        limit_count_xwdown += 1;
                    }
                
                    //yw_upstream
                    if ((-result[14] * 0.5 * (((v[k*gd.ijcells + j * gd.icells + i] - v[(k-1)*gd.ijcells + j * gd.icells + i]) * gd.dzhi[k]) + ((w[k*gd.ijcells + j * gd.icells + i] - w[k*gd.ijcells + (j-1) * gd.icells + i]) * dyi))) < 0.)
                    {
                        result[14] = backscatter_limit_fac * result[14];
                        limit_count_ywup += 1;
                    }
                    //yw_downstream
                    if ((-result[15] * 0.5 * (((v[k*gd.ijcells + (j+1) * gd.icells + i] - v[(k-1)*gd.ijcells + (j+1) * gd.icells + i]) * gd.dzhi[k]) + ((w[k*gd.ijcells + (j+1) * gd.icells + i] - w[k*gd.ijcells + j * gd.icells + i]) * dyi))) < 0.)
                    {
                        result[15] = backscatter_limit_fac * result[15];
                        limit_count_ywdown += 1;
                    }
                    //zw_upstream
                    if ((-result[16] * (w[k*gd.ijcells + j * gd.icells + i] - w[(k-1)*gd.ijcells + j * gd.icells + i]) * gd.dzi[k-1]) < 0.)
                    {
                        result[16] = backscatter_limit_fac * result[16];
                        limit_count_zwup += 1;
                    }
                    //zw_downstream
                    if ((-result[17] * (w[(k+1)*gd.ijcells + j * gd.icells + i] - w[k*gd.ijcells + j * gd.icells + i]) * gd.dzi[k]) < 0.)
                    {
                        result[17] = backscatter_limit_fac * result[17];
                        limit_count_zwdown += 1;
                    } 
                }

                ////Implement limiter on outputs
                //
                ////Limit highest values
                //result[0]  = std::min(result[0],  m_xumean[k-gd.kstart] + 1*m_xustd[k-gd.kstart]);
                //result[1]  = std::min(result[1],  m_xumean[k-gd.kstart] + 1*m_xustd[k-gd.kstart]);
                //result[2]  = std::min(result[2],  m_yumean[k-gd.kstart] + 1*m_yustd[k-gd.kstart]);
                //result[3]  = std::min(result[3],  m_yumean[k-gd.kstart] + 1*m_yustd[k-gd.kstart]);
                //result[4]  = std::min(result[4],  m_zumean[k-gd.kstart] + 1*m_zustd[k-gd.kstart]);
                //result[5]  = std::min(result[5],  m_zumean[k-gd.kstart] + 1*m_zustd[k-gd.kstart]);
                //result[6]  = std::min(result[6],  m_xvmean[k-gd.kstart] + 1*m_xvstd[k-gd.kstart]);
                //result[7]  = std::min(result[7],  m_xvmean[k-gd.kstart] + 1*m_xvstd[k-gd.kstart]);
                //result[8]  = std::min(result[8],  m_yvmean[k-gd.kstart] + 1*m_yvstd[k-gd.kstart]);
                //result[9]  = std::min(result[9],  m_yvmean[k-gd.kstart] + 1*m_yvstd[k-gd.kstart]);
                //result[10] = std::min(result[10], m_zvmean[k-gd.kstart] + 1*m_zvstd[k-gd.kstart]);
                //result[11] = std::min(result[11], m_zvmean[k-gd.kstart] + 1*m_zvstd[k-gd.kstart]);
                //if (k != gd.kstart) //Transport on lowest grid cells are not evaluated, and thus do not need to be limited
                //{
                //    result[12] = std::min(result[12], m_xwmean[k-gd.kstart] + 1*m_xwstd[k-gd.kstart]);
                //    result[13] = std::min(result[13], m_xwmean[k-gd.kstart] + 1*m_xwstd[k-gd.kstart]);
                //    result[14] = std::min(result[14], m_ywmean[k-gd.kstart] + 1*m_ywstd[k-gd.kstart]);
                //    result[15] = std::min(result[15], m_ywmean[k-gd.kstart] + 1*m_ywstd[k-gd.kstart]);
                //    result[16] = std::min(result[16], m_zwmean[k-gd.kstart-1] + 1*m_zwstd[k-gd.kstart-1]);
                //    result[17] = std::min(result[17], m_zwmean[k-gd.kstart-1] + 1*m_zwstd[k-gd.kstart-1]); //NOTE: -1 is included in index to take into account the staggered grid orientation
                //}

                ////Limit lowest values
                //result[0]  = std::max(result[0],  m_xumean[k-gd.kstart] - 1*m_xustd[k-gd.kstart]);
                //result[1]  = std::max(result[1],  m_xumean[k-gd.kstart] - 1*m_xustd[k-gd.kstart]);
                //result[2]  = std::max(result[2],  m_yumean[k-gd.kstart] - 1*m_yustd[k-gd.kstart]);
                //result[3]  = std::max(result[3],  m_yumean[k-gd.kstart] - 1*m_yustd[k-gd.kstart]);
                //result[4]  = std::max(result[4],  m_zumean[k-gd.kstart] - 1*m_zustd[k-gd.kstart]);
                //result[5]  = std::max(result[5],  m_zumean[k-gd.kstart] - 1*m_zustd[k-gd.kstart]);
                //result[6]  = std::max(result[6],  m_xvmean[k-gd.kstart] - 1*m_xvstd[k-gd.kstart]);
                //result[7]  = std::max(result[7],  m_xvmean[k-gd.kstart] - 1*m_xvstd[k-gd.kstart]);
                //result[8]  = std::max(result[8],  m_yvmean[k-gd.kstart] - 1*m_yvstd[k-gd.kstart]);
                //result[9]  = std::max(result[9],  m_yvmean[k-gd.kstart] - 1*m_yvstd[k-gd.kstart]);
                //result[10] = std::max(result[10], m_zvmean[k-gd.kstart] - 1*m_zvstd[k-gd.kstart]);
                //result[11] = std::max(result[11], m_zvmean[k-gd.kstart] - 1*m_zvstd[k-gd.kstart]);
                //if (k != gd.kstart) //Transport on lowest grid cells are not evaluated, and thus do not need to be limited
                //{
                //    result[12] = std::max(result[12], m_xwmean[k-gd.kstart] - 1*m_xwstd[k-gd.kstart]);
                //    result[13] = std::max(result[13], m_xwmean[k-gd.kstart] - 1*m_xwstd[k-gd.kstart]);
                //    result[14] = std::max(result[14], m_ywmean[k-gd.kstart] - 1*m_ywstd[k-gd.kstart]);
                //    result[15] = std::max(result[15], m_ywmean[k-gd.kstart] - 1*m_ywstd[k-gd.kstart]);
                //    result[16] = std::max(result[16], m_zwmean[k-gd.kstart-1] - 1*m_zwstd[k-gd.kstart-1]);
                //    result[17] = std::max(result[17], m_zwmean[k-gd.kstart-1] - 1*m_zwstd[k-gd.kstart-1]); //NOTE: -1 is included in index to take into account the staggered grid orientation
                //}


                //Check whether a horizontal boundary is reached, and if so make use of horizontal periodic BCs.
                int i_upbound = 0;
                int i_downbound = 0;
                int j_upbound = 0;
                int j_downbound = 0;
                // upstream boundary
                if (i == (gd.istart))
                {
                    i_upbound = gd.iend - 1;
                }
                else
                {
                    i_upbound = i - 1;
                }
                if (j == (gd.jstart))
                {
                    j_upbound = gd.jend - 1;
                }
                else
                {
                    j_upbound = j - 1;
                }
                // downstream boundary
                if (i == (gd.iend - 1))
                {
                    i_downbound = gd.istart;
                }
                else
                {
                    i_downbound = i + 1;
                }
                if (j == (gd.jend - 1))
                {
                    j_downbound = gd.jstart;
                }
                else
                {
                    j_downbound = j + 1;
                }

                //Calculate damping factor for calculated transports
                float fac=1;//Don't impose a damping factor
                //float fac=std::min(std::min((gd.zh[k]/(0.25*gd.zh[gd.kend]))+0.1,0.3),((gd.zh[gd.kend]-gd.zh[k])/(0.25*gd.zh[gd.kend])+0.1)); //Apply damping close to the surface

                //Calculate tendencies using predictions from MLP
                //xu_upstream
                ut[k*gd.ijcells + j * gd.icells + i]         +=  result[0] * dxi * fac;
                ut[k*gd.ijcells + j * gd.icells + i_upbound] += -result[0] * dxi * fac;

                //xu_downstream
                ut[k*gd.ijcells + j * gd.icells+ i]           += -result[1] * dxi * fac;
                ut[k*gd.ijcells + j * gd.icells+ i_downbound] +=  result[1] * dxi * fac;

                //yu_upstream
                ut[k*gd.ijcells + j * gd.icells + i]         +=  result[2] * dyi * fac;
                ut[k*gd.ijcells + j_upbound * gd.icells + i] += -result[2] * dyi * fac;

                //yu_downstream
                ut[k*gd.ijcells + j * gd.icells + i]           += -result[3] * dyi * fac;
                ut[k*gd.ijcells + j_downbound * gd.icells + i] +=  result[3] * dyi * fac;

                //zu_upstream
                if (k != gd.kstart)
                    // NOTES: 1) zu_upstream is in this way implicitly set to 0 at the bottom layer
                    // 2) ghost cell is not assigned.
                {
                    ut[(k-1)*gd.ijcells + j * gd.icells + i] += -result[4] * gd.dzi[k-1] * fac;
                    ut[k*gd.ijcells + j * gd.icells + i]     +=  result[4] * gd.dzi[k] * fac;
                }

                //zu_downstream
                if (k != (gd.kend - 1))
                    // NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
                    // 2) ghost cell is not assigned.
                {
                    ut[k*gd.ijcells + j * gd.icells + i]     += -result[5] * gd.dzi[k] * fac;
                    ut[(k+1)*gd.ijcells + j * gd.icells + i] +=  result[5] * gd.dzi[k+1] * fac;
                }

                //xv_upstream
                vt[k*gd.ijcells + j * gd.icells + i]         +=  result[6] * dxi * fac;
                vt[k*gd.ijcells + j * gd.icells + i_upbound] += -result[6] * dxi * fac;

                //xv_downstream
                vt[k*gd.ijcells + j * gd.icells + i]           += -result[7] * dxi * fac;
                vt[k*gd.ijcells + j * gd.icells + i_downbound] +=  result[7] * dxi * fac;

                //yv_upstream
                vt[k*gd.ijcells + j * gd.icells + i]         +=  result[8] * dyi * fac;
                vt[k*gd.ijcells + j_upbound * gd.icells + i] += -result[8] * dyi * fac;

                //yv_downstream
                vt[k*gd.ijcells + j * gd.icells + i]           += -result[9] * dyi * fac;
                vt[k*gd.ijcells + j_downbound * gd.icells + i] +=  result[9] * dyi * fac;

                //zv_upstream
                if (k != gd.kstart)
                    // NOTES: 1) zu_upstream is in this way implicitly set to 0 at the bottom layer
                    // 2) ghost cell is not assigned.
                {
                    vt[(k - 1)*gd.ijcells + j * gd.icells + i] += -result[10] * gd.dzi[k - 1] * fac;
                    vt[k*gd.ijcells + j * gd.icells + i]       +=  result[10] * gd.dzi[k] * fac;
                }

                //zv_downstream
                if (k != (gd.kend - 1))
                    // NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
                    // 2) ghost cell is not assigned.
                {
                    vt[k*gd.ijcells + j * gd.icells + i]       += -result[11] * gd.dzi[k] * fac;
                    vt[(k + 1)*gd.ijcells + j * gd.icells + i] +=  result[11] * gd.dzi[k + 1] * fac;
                }

                if (k != gd.kstart) //Don't adjust wt for bottom layer, should stay 0
                {
                    //xw_upstream
                    wt[k*gd.ijcells + j * gd.icells + i]         +=  result[12] * dxi * fac;
                    wt[k*gd.ijcells + j * gd.icells + i_upbound] += -result[12] * dxi * fac;

                    //xw_downstream
                    wt[k*gd.ijcells + j * gd.icells + i]           += -result[13] * dxi * fac;
                    wt[k*gd.ijcells + j * gd.icells + i_downbound] +=  result[13] * dxi * fac;

                    //yw_upstream
                    wt[k*gd.ijcells + j * gd.icells + i]         +=  result[14] * dyi * fac;
                    wt[k*gd.ijcells + j_upbound * gd.icells + i] += -result[14] * dyi * fac;

                    //yw_downstream
                    wt[k*gd.ijcells + j * gd.icells + i]           += -result[15] * dyi * fac;
                    wt[k*gd.ijcells + j_downbound * gd.icells + i] +=  result[15] * dyi * fac;

                    //zw_upstream
                    if (k != (gd.kstart+1))
                    //NOTE: Dont'adjust wt for bottom layer, should stay 0
                    {
                        wt[(k - 1)*gd.ijcells + j * gd.icells + i] += -result[16] * gd.dzhi[k - 1] * fac;
                    }
                    wt[k*gd.ijcells + j * gd.icells + i]           +=  result[16] * gd.dzhi[k] * fac;

                    //zw_downstream
                    wt[k*gd.ijcells + j * gd.icells + i]           += -result[17] * gd.dzhi[k] * fac;
                    if (k != (gd.kend - 1))
                    // NOTE:although this does not change wt at the bottom layer, 
                    // it is still not included for k=0 to keep consistency between the top and bottom of the domain.
                    {
                        wt[(k + 1)*gd.ijcells + j * gd.icells + i] += result[17] * gd.dzhi[k + 1] * fac;
                    }
                }

                // Execute for each iteration in the first layer above the bottom layer, and for each iteration in the top layer, 
                // the MLP for a second grid cell to calculate 'missing' zw-values.
                if ((k == (gd.kend - 1)) || (k == (gd.kstart + 1)))
                {
                    //Determine the second grid cell based on the offset.
                    int i_2grid = 0;
                    if (offset == 1)
                    {
                        i_2grid = i - 1;
                    }
                    else
                    {
                        i_2grid = i + 1;
                    }

                    //Select second grid box
                    select_box(u, m_input_ctrlu_u.data(), k, j, i_2grid, boxsize, 0, 0, 0, 0, 0, 0);
                    select_box(v, m_input_ctrlu_v.data(), k, j, i_2grid, boxsize, 0, 0, 1, 0, 0, 1);
                    select_box(w, m_input_ctrlu_w.data(), k, j, i_2grid, boxsize, 1, 0, 0, 0, 0, 1);
                    select_box(u, m_input_ctrlv_u.data(), k, j, i_2grid, boxsize, 0, 0, 0, 1, 1, 0);
                    select_box(v, m_input_ctrlv_v.data(), k, j, i_2grid, boxsize, 0, 0, 0, 0, 0, 0);
                    select_box(w, m_input_ctrlv_w.data(), k, j, i_2grid, boxsize, 1, 0, 0, 1, 0, 0);
                    select_box(u, m_input_ctrlw_u.data(), k, j, i_2grid, boxsize, 0, 1, 0, 0, 1, 0);
                    select_box(v, m_input_ctrlw_v.data(), k, j, i_2grid, boxsize, 0, 1, 1, 0, 0, 0);
                    select_box(w, m_input_ctrlw_w.data(), k, j, i_2grid, boxsize, 0, 0, 0, 0, 0, 0);

                    //Execute MLP for selected second grid cell
                    Inference(
                        m_input_ctrlu_u.data(), m_input_ctrlu_v.data(), m_input_ctrlu_w.data(),
                        m_hiddenu_wgth.data(), m_hiddenu_bias.data(), m_hiddenu_alpha,
                        m_outputu_wgth.data(), m_outputu_bias.data(),
                        m_input_ctrlv_u.data(), m_input_ctrlv_v.data(), m_input_ctrlv_w.data(),
                        m_hiddenv_wgth.data(), m_hiddenv_bias.data(), m_hiddenv_alpha,
                        m_outputv_wgth.data(), m_outputv_bias.data(),
                        m_input_ctrlw_u.data(), m_input_ctrlw_v.data(), m_input_ctrlw_w.data(),
                        m_hiddenw_wgth.data(), m_hiddenw_bias.data(), m_hiddenw_alpha,
                        m_outputw_wgth.data(), m_outputw_bias.data(),
                        m_mean_input.data(), m_stdev_input.data(),
                        m_mean_label.data(), m_stdev_label.data(),
                        m_utau_ref, m_output_denorm_utau2,
                        m_output_zw.data(), result_zw.data(), true
                    );
                    
                    //Implement backscatter limiter on predicted transports (i.e. -tau_ij * filtS_ij < 0)
                    //zw_upstream
                    if ((-result_zw[0] * (w[k*gd.ijcells + j * gd.icells + i_2grid] - w[(k-1)*gd.ijcells + j * gd.icells + i_2grid]) * gd.dzi[k-1]) < 0.)
                    {
                        result_zw[0] = backscatter_limit_fac * result_zw[0];
                        limit_count_zwup += 1;
                    }
                    //zw_downstream
                    if ((-result_zw[1] * (w[(k+1)*gd.ijcells + j * gd.icells + i_2grid] - w[k*gd.ijcells + j * gd.icells + i_2grid]) * gd.dzi[k]) < 0.)
                    {
                        result_zw[1] = backscatter_limit_fac * result_zw[1];
                        limit_count_zwdown += 1;
                    }

                    ////Limit highest values
                    //result_zw[0] = std::min(result_zw[0], m_zwmean[k-gd.kstart-1] + 1*m_zwstd[k-gd.kstart-1]);
                    //result_zw[1] = std::min(result_zw[1], m_zwmean[k-gd.kstart-1] + 1*m_zwstd[k-gd.kstart-1]); //NOTE: -1 is included in index to take into account the staggered grid orientation

                    ////Limit lowest values
                    //result_zw[0] = std::max(result_zw[0], m_zwmean[k-gd.kstart-1] - 1*m_zwstd[k-gd.kstart-1]);
                    //result_zw[1] = std::max(result_zw[1], m_zwmean[k-gd.kstart-1] - 1*m_zwstd[k-gd.kstart-1]); //NOTE: -1 is included in index to take into account the staggered grid orientation

                    //Store calculated tendencies
                    //zw_upstream
                    if (k == (gd.kstart + 1))
                    {
                        wt[k * gd.ijcells + j * gd.icells + i_2grid] +=  result_zw[0] * gd.dzhi[k] * fac;
                    }
                    //zw_downstream
                    else
                    {
                        wt[k * gd.ijcells + j * gd.icells + i_2grid] += -result_zw[1] * gd.dzhi[k] * fac;
                    }           
                }
            }
        }
    }
    master.print_message("Start ANN iteration \n");
    master.print_message("Number of values removed by backscatter filter xuup: %d \n", limit_count_xuup);
    master.print_message("Number of values removed by backscatter filter xudown: %d \n", limit_count_xudown);
    master.print_message("Number of values removed by backscatter filter yuup: %d \n", limit_count_yuup);
    master.print_message("Number of values removed by backscatter filter yudown: %d \n", limit_count_yudown);
    master.print_message("Number of values removed by backscatter filter zuup: %d \n", limit_count_zuup);
    master.print_message("Number of values removed by backscatter filter zudown: %d \n", limit_count_zudown);
    master.print_message("Number of values removed by backscatter filter xvup: %d \n", limit_count_xvup);
    master.print_message("Number of values removed by backscatter filter xvdown: %d \n", limit_count_xvdown);
    master.print_message("Number of values removed by backscatter filter yvup: %d \n", limit_count_yvup);
    master.print_message("Number of values removed by backscatter filter yvdown: %d \n", limit_count_yvdown);
    master.print_message("Number of values removed by backscatter filter zvup: %d \n", limit_count_zvup);
    master.print_message("Number of values removed by backscatter filter zvdown: %d \n", limit_count_zvdown);
    master.print_message("Number of values removed by backscatter filter xwup: %d \n", limit_count_xwup);
    master.print_message("Number of values removed by backscatter filter xwdown: %d \n", limit_count_xwdown);
    master.print_message("Number of values removed by backscatter filter ywup: %d \n", limit_count_ywup);
    master.print_message("Number of values removed by backscatter filter ywdown: %d \n", limit_count_ywdown);
    master.print_message("Number of values removed by backscatter filter zwup: %d \n", limit_count_zwup);
    master.print_message("Number of values removed by backscatter filter zwdown: %d \n", limit_count_zwdown);
    master.print_message("Finish ANN iteration \n");

    ////Loop over tendencies to apply Gaussian filter for smoothing

    ////Define temporary variables for intermediate storage
    //std::vector<TF> ut_temp_field(gd.ncells, 0.0f);
    //std::vector<TF> vt_temp_field(gd.ncells, 0.0f);
    //std::vector<TF> wt_temp_field(gd.ncells, 0.0f);
    //TF ut_temp = 0.0;
    //TF vt_temp = 0.0;
    //TF wt_temp = 0.0;
    //int j_filter_idx = 0;
    //int i_filter_idx = 0;
    //int j_filter = 0; //Initialize to 0
    //int i_filter = 0; //Initialize to 0

    //for (int k = gd.kstart; k < gd.kend; ++k)
    //{
    //    for (int j = gd.jstart; j < gd.jend; ++j)
    //    {
    //        for (int i = gd.istart; i < gd.iend; ++i)
    //        {
    //            //Loop and sum over local horizontal 5*5 kernel to get the tendency
    //            //NOTE: make use of periodic BCs in horizontal directions
    //            ut_temp = 0.0; //Initialize to 0
    //            vt_temp = 0.0;
    //            wt_temp = 0.0;
    //    

    //            for (int j_gkernel_idx = -2; j_gkernel_idx < 3; ++j_gkernel_idx)
    //            {
    //                j_filter = j + j_gkernel_idx;
    //                if (j_filter < gd.jstart)
    //                {
    //                    j_filter_idx = gd.jend - (gd.jstart - j_filter);
    //                }
    //                else if (j_filter >= gd.jend)
    //                {
    //                    j_filter_idx = gd.jstart + (j_filter - gd.jend);
    //                }
    //                else
    //                {
    //                    j_filter_idx = j_filter;
    //                }
    //                for (int i_gkernel_idx = -2; i_gkernel_idx < 3; ++i_gkernel_idx)
    //                {
    //                    i_filter = i + i_gkernel_idx;
    //                    if (i_filter < gd.istart)
    //                    {
    //                        i_filter_idx = gd.iend - (gd.istart - i_filter);
    //                    }
    //                    else if (i_filter >= gd.iend)
    //                    {
    //                        i_filter_idx = gd.istart + (i_filter - gd.iend);
    //                    }
    //                    else
    //                    {
    //                        i_filter_idx = i_filter;
    //                    }

    //                    ut_temp += ut[k * gd.ijcells + j_filter_idx * gd.icells + i_filter_idx] * gkernel[(j_gkernel_idx+2) * 5 + (i_gkernel_idx+2)];
    //                    vt_temp += vt[k * gd.ijcells + j_filter_idx * gd.icells + i_filter_idx] * gkernel[(j_gkernel_idx+2) * 5 + (i_gkernel_idx+2)];
    //                    wt_temp += wt[k * gd.ijcells + j_filter_idx * gd.icells + i_filter_idx] * gkernel[(j_gkernel_idx+2) * 5 + (i_gkernel_idx+2)];        
    //                }
    //            }
    //            
    //            //Assign calculated smoothed tendencies to temporary fields
    //            ut_temp_field[k*gd.ijcells + j * gd.icells + i] = ut_temp;
    //            vt_temp_field[k*gd.ijcells + j * gd.icells + i] = vt_temp;
    //            wt_temp_field[k*gd.ijcells + j * gd.icells + i] = wt_temp;
    //        }
    //    }
    //}

    ////Assign temporary fields to the actual tendency fields
    //for (int k = gd.kstart; k < gd.kend; ++k)
    //{
    //    for (int j = gd.jstart; j < gd.jend; ++j)
    //    {
    //        for (int i = gd.istart; i < gd.iend; ++i)
    //        {
    //            ut[k*gd.ijcells + j * gd.icells + i] = ut_temp_field[k*gd.ijcells + j * gd.icells + i];
    //            vt[k*gd.ijcells + j * gd.icells + i] = vt_temp_field[k*gd.ijcells + j * gd.icells + i];
    //            wt[k*gd.ijcells + j * gd.icells + i] = wt_temp_field[k*gd.ijcells + j * gd.icells + i];
    //        }
    //    }
    //}
}
template<typename TF>
void Diff_NN<TF>::hidden_layer1(
    const float* restrict const weights,
    const float* restrict const bias,
    const float* restrict const input,
    float* restrict const layer_out,
    const float alpha
)
{
    // Calculate hidden neurons outputs as matrix vector multiplication using BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N_hidden, N_input_tot_adjusted, 
        1., weights, N_input_tot_adjusted, input, 1, 0, layer_out, 1);
    
    //Loop over hidden neurons to add bias and calculate activations using Leaky ReLu
    for (int hiddenidx = 0; hiddenidx < N_hidden; ++hiddenidx)
    {
        layer_out[hiddenidx] += bias[hiddenidx];
        layer_out[hiddenidx] = std::max(alpha * layer_out[hiddenidx], layer_out[hiddenidx]);
    }
}  
//output layer
template<typename TF>
void Diff_NN<TF>::output_layer(
    const float* restrict const weights,
    const float* restrict const bias,
    const float* restrict const layer_in,
    float* restrict const layer_out
)
{
    // Calculate hidden neurons outputs as matrix vector multiplication using BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans, N_output_control, N_hidden,
        1., weights, N_hidden, layer_in, 1, 0, layer_out, 1);

    //Loop over hidden neurons to add bias
    for (int outputidx = 0; outputidx < N_output_control; ++outputidx)
    {
        layer_out[outputidx] += bias[outputidx];
    }
}

template<typename TF>
void Diff_NN<TF>::Inference(
    float* restrict const input_ctrlu_u,
    float* restrict const input_ctrlu_v,
    float* restrict const input_ctrlu_w,
    const float* restrict const hiddenu_wgth,
    const float* restrict const hiddenu_bias,
    const float hiddenu_alpha,
    const float* restrict const outputu_wgth,
    const float* restrict const outputu_bias,
    float* restrict const input_ctrlv_u,
    float* restrict const input_ctrlv_v,
    float* restrict const input_ctrlv_w,
    const float* restrict const hiddenv_wgth,
    const float* restrict const hiddenv_bias,
    const float hiddenv_alpha,
    const float* restrict const outputv_wgth,
    const float* restrict const outputv_bias,
    float* restrict const input_ctrlw_u,
    float* restrict const input_ctrlw_v,
    float* restrict const input_ctrlw_w,
    const float* restrict const hiddenw_wgth,
    const float* restrict const hiddenw_bias,
    const float hiddenw_alpha,
    const float* restrict const outputw_wgth,
    const float* restrict const outputw_bias,
    const float* restrict const mean_input,
    const float* restrict const stdev_input,
    const float* restrict const mean_label,
    const float* restrict const stdev_label,
    const float utau_ref,
    const float output_denorm_utau2,
    float* restrict const output,
    float* restrict const output_denorm,
    const bool zw_flag)

{   
    // Initialize fixed arrays for input layers
    std::array<float, N_input_tot_adjusted> input_ctrlu;
    std::array<float, N_input_tot_adjusted> input_ctrlv;
    std::array<float, N_input_tot_adjusted> input_ctrlw;

        // Normalize with mean, st. dev, and utau_ref.
    constexpr int N_input_adjusted2 = 2 * N_input_adjusted;
    constexpr int N_input_comb2     = N_input_adjusted + N_input;
    for (int inpidx = 0; inpidx < N_input;++inpidx)
    {
        input_ctrlu[inpidx]                     = (((input_ctrlu_u[inpidx] / utau_ref) - mean_input[0]) / stdev_input[0]);
        input_ctrlv[N_input_adjusted + inpidx]  = (((input_ctrlv_v[inpidx] / utau_ref) - mean_input[1]) / stdev_input[1]);
        input_ctrlw[N_input_adjusted2 + inpidx] = (((input_ctrlw_w[inpidx] / utau_ref) - mean_input[2]) / stdev_input[2]);
    }
    for (int inpidx = 0; inpidx < N_input_adjusted; ++inpidx)
    {
        input_ctrlu[N_input + inpidx]           = (((input_ctrlu_v[inpidx] / utau_ref) - mean_input[1]) / stdev_input[1]);
        input_ctrlu[N_input_comb2 + inpidx]     = (((input_ctrlu_w[inpidx] / utau_ref) - mean_input[2]) / stdev_input[2]);
        input_ctrlv[inpidx]                     = (((input_ctrlv_u[inpidx] / utau_ref) - mean_input[0]) / stdev_input[0]);
        input_ctrlv[N_input_comb2 + inpidx]     = (((input_ctrlv_w[inpidx] / utau_ref) - mean_input[2]) / stdev_input[2]);
        input_ctrlw[inpidx]                     = (((input_ctrlw_u[inpidx] / utau_ref) - mean_input[0]) / stdev_input[0]);
        input_ctrlw[N_input_adjusted + inpidx]  = (((input_ctrlw_v[inpidx] / utau_ref) - mean_input[1]) / stdev_input[1]);
    }

    //control volume u
    
    //hidden layer
    std::array<float, N_hidden> hiddenu;
    hidden_layer1(hiddenu_wgth, hiddenu_bias,
        input_ctrlu.data(), hiddenu.data(), hiddenu_alpha);

    //output layer
    std::array<float, N_output_control> outputu;
    output_layer(outputu_wgth, outputu_bias, hiddenu.data(), outputu.data());

    //control volume v

    //hidden layer
    std::array<float, N_hidden> hiddenv;
    hidden_layer1(hiddenv_wgth, hiddenv_bias,
        input_ctrlv.data(), hiddenv.data(), hiddenv_alpha);

    //output layer
    std::array<float, N_output_control> outputv;
    output_layer(outputv_wgth, outputv_bias, hiddenv.data(), outputv.data());

    //control volume w

    //hidden layer
    std::array<float, N_hidden> hiddenw;
    hidden_layer1(hiddenw_wgth, hiddenw_bias,
        input_ctrlw.data(), hiddenw.data(), hiddenw_alpha);

    //output layer
    std::array<float, N_output_control> outputw;
    output_layer(outputw_wgth, outputw_bias, hiddenw.data(), outputw.data());

    //Concatenate output layers & denormalize
    if (zw_flag)
    {
        output[0] = outputw[4]; // zw_upstream
        output[1] = outputw[5]; // zw_downstream

        //Denormalize
        output_denorm[0] = ((output[0] * stdev_label[16]) + mean_label[16]) * output_denorm_utau2;
        output_denorm[1] = ((output[1] * stdev_label[17]) + mean_label[17]) * output_denorm_utau2;
    }
    else
    {
        for (int outputidx = 0; outputidx < 6; ++outputidx)
        {
            output[outputidx    ] = outputu[outputidx];
        }
        for (int outputidx = 0; outputidx < 6; ++outputidx)
        {
            output[outputidx + 6] = outputv[outputidx];
        }
        for (int outputidx = 0; outputidx < 6; ++outputidx)
        {
            output[outputidx + 12] = outputw[outputidx];
        }
        
        //Denormalize
        for (int outputidx = 0; outputidx < N_output; ++outputidx)
        {
            output_denorm[outputidx] = ((output[outputidx] * stdev_label[outputidx]) + mean_label[outputidx]) * output_denorm_utau2;
        }
    }
}
template<typename TF>
void Diff_NN<TF>::file_reader(
        float* const weights,
        const std::string& filename,
        const int N)
{
    std::ifstream file (filename); // open file in read mode, filename instead of filename.c_str()
    //Test whether file has been read
    try
    {
        if (!file.is_open())
        {
            throw "Couldn't read file specified as: " + filename;
        }
    }
    catch(std::string exception)
    {
        std::cerr << "Error: " << exception << "\n";
    }
    for ( int i=0; i<N;++i)
        file>> weights[i];
    file.close();
}

template<typename TF>
Diff_NN<TF>::Diff_NN(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Boundary<TF>& boundaryin, Input& inputin) :
    Diff<TF>(masterin, gridin, fieldsin, boundaryin, inputin),
    boundary_cyclic(master, grid),
    field3d_operators(master, grid, fields)
{
    const int igc = 2;
    const int jgc = 2;
    const int kgc = 2;
    grid.set_minimum_ghost_cells(igc, jgc, kgc);

    fields.mp.at("u")->fld.data(),
    fields.mp.at("v")->fld.data(),
    fields.mp.at("w")->fld.data(),


////////////////

    dnmax = inputin.get_item<TF>("diff", "dnmax", "", 0.4  );

//   if (grid.get_spatial_order() != Grid_order::Second)
//      throw std::runtime_error("Diff_NN only runs with second order grids");

    //Hard-code file directory where variables MLP are stored
    std::string var_filepath = "../../inferenceNN/Variables_MLP13/";
    
    // Define names of text files, which is ok assuming that ONLY the directory of the text files change and not the text file names themselves.
    std::string hiddenu_wgth_str(var_filepath + "MLPu_hidden_kernel.txt");
    std::string hiddenv_wgth_str(var_filepath + "MLPv_hidden_kernel.txt");
    std::string hiddenw_wgth_str(var_filepath + "MLPw_hidden_kernel.txt");
    std::string outputu_wgth_str(var_filepath + "MLPu_output_kernel.txt");
    std::string outputv_wgth_str(var_filepath + "MLPv_output_kernel.txt");
    std::string outputw_wgth_str(var_filepath + "MLPw_output_kernel.txt");
    std::string hiddenu_bias_str(var_filepath + "MLPu_hidden_bias.txt");
    std::string hiddenv_bias_str(var_filepath + "MLPv_hidden_bias.txt");
    std::string hiddenw_bias_str(var_filepath + "MLPw_hidden_bias.txt");
    std::string outputu_bias_str(var_filepath + "MLPu_output_bias.txt");
    std::string outputv_bias_str(var_filepath + "MLPv_output_bias.txt");
    std::string outputw_bias_str(var_filepath + "MLPw_output_bias.txt");
    std::string hiddenu_alpha_str(var_filepath + "MLPu_hidden_alpha.txt");
    std::string hiddenv_alpha_str(var_filepath + "MLPv_hidden_alpha.txt");
    std::string hiddenw_alpha_str(var_filepath + "MLPw_hidden_alpha.txt");
    
    std::string mean_input_str(var_filepath + "means_inputs.txt");
    std::string mean_label_str(var_filepath + "means_labels.txt");
    std::string stdev_input_str(var_filepath + "stdevs_inputs.txt");
    std::string stdev_label_str(var_filepath + "stdevs_labels.txt");
    
    std::string utau_ref_str(var_filepath + "utau_ref.txt");
    std::string output_denorm_utau2_str(var_filepath + "output_denorm_utau2.txt");
    
    // Initialize dynamically bias parameters, weights, means/stdevs, and other variables according to values stored in files specified with input strings.
    std::vector<float> hiddenu_wgth_notr(N_hidden*N_input_tot_adjusted); // Store tranposed variants of weights in class, see below. These matrices are temporary.
    std::vector<float> hiddenv_wgth_notr(N_hidden*N_input_tot_adjusted);
    std::vector<float> hiddenw_wgth_notr(N_hidden*N_input_tot_adjusted);
    std::vector<float> outputu_wgth_notr(N_output_control*N_hidden);
    std::vector<float> outputv_wgth_notr(N_output_control*N_hidden);
    std::vector<float> outputw_wgth_notr(N_output_control*N_hidden);
    m_hiddenu_wgth.resize(N_hidden*N_input_tot_adjusted);
    m_hiddenv_wgth.resize(N_hidden*N_input_tot_adjusted);
    m_hiddenw_wgth.resize(N_hidden*N_input_tot_adjusted);
    m_outputu_wgth.resize(N_output_control*N_hidden);
    m_outputv_wgth.resize(N_output_control*N_hidden);
    m_outputw_wgth.resize(N_output_control*N_hidden);
    m_hiddenu_bias.resize(N_hidden);
    m_hiddenv_bias.resize(N_hidden);
    m_hiddenw_bias.resize(N_hidden);
    m_outputu_bias.resize(N_output_control);
    m_outputv_bias.resize(N_output_control);
    m_outputw_bias.resize(N_output_control);
    m_mean_input.resize(N_inputvar);
    m_stdev_input.resize(N_inputvar);
    m_mean_label.resize(N_output);
    m_stdev_label.resize(N_output);
    m_hiddenu_alpha = 0.f;
    m_hiddenv_alpha = 0.f;
    m_hiddenw_alpha = 0.f;
    m_utau_ref = 0.f;
    m_output_denorm_utau2 = 0.f;
    
    file_reader(hiddenu_wgth_notr.data(),hiddenu_wgth_str,N_hidden*N_input_tot_adjusted);
    file_reader(hiddenv_wgth_notr.data(),hiddenv_wgth_str,N_hidden*N_input_tot_adjusted);
    file_reader(hiddenw_wgth_notr.data(),hiddenw_wgth_str,N_hidden*N_input_tot_adjusted);
    file_reader(outputu_wgth_notr.data(),outputu_wgth_str,N_output*N_hidden);
    file_reader(outputv_wgth_notr.data(),outputv_wgth_str,N_output*N_hidden);
    file_reader(outputw_wgth_notr.data(),outputw_wgth_str,N_output*N_hidden);
    file_reader(m_hiddenu_bias.data(), hiddenu_bias_str, N_hidden);
    file_reader(m_hiddenv_bias.data(), hiddenv_bias_str, N_hidden);
    file_reader(m_hiddenw_bias.data(), hiddenw_bias_str, N_hidden);
    file_reader(m_outputu_bias.data(), outputu_bias_str, N_output);
    file_reader(m_outputv_bias.data(), outputv_bias_str, N_output);
    file_reader(m_outputw_bias.data(), outputw_bias_str, N_output);
    file_reader(m_mean_input.data(),mean_input_str,N_inputvar);
    file_reader(m_stdev_input.data(),stdev_input_str,N_inputvar);
    file_reader(m_mean_label.data(),mean_label_str,N_output);
    file_reader(m_stdev_label.data(),stdev_label_str,N_output);
    file_reader(&m_hiddenu_alpha,hiddenu_alpha_str,1);
    file_reader(&m_hiddenv_alpha,hiddenv_alpha_str,1);
    file_reader(&m_hiddenw_alpha,hiddenw_alpha_str,1);
    file_reader(&m_utau_ref, utau_ref_str, 1);
    file_reader(&m_output_denorm_utau2,output_denorm_utau2_str,1);
    
    // Take transpose of weights and store those in the class
    for (int hiddenidx = 0; hiddenidx < N_hidden; ++hiddenidx)
    {
        for (int inpidx = 0; inpidx < N_input_tot_adjusted; ++inpidx)
        {
            int idx_tr = inpidx + hiddenidx * N_input_tot_adjusted;
            int idx_notr = inpidx * N_hidden + hiddenidx;
            m_hiddenu_wgth[idx_tr] = hiddenu_wgth_notr[idx_notr];
            m_hiddenv_wgth[idx_tr] = hiddenv_wgth_notr[idx_notr];
            m_hiddenw_wgth[idx_tr] = hiddenw_wgth_notr[idx_notr];
        }
    }
    for (int outputidx = 0; outputidx < N_output_control; ++outputidx)
    {
        for (int hiddenidx = 0; hiddenidx < N_hidden; ++hiddenidx)
        {
            int idx_tr = hiddenidx + outputidx * N_hidden;
            int idx_notr = hiddenidx * N_output_control + outputidx;
            m_outputu_wgth[idx_tr] = outputu_wgth_notr[idx_notr];
            m_outputv_wgth[idx_tr] = outputv_wgth_notr[idx_notr];
            m_outputw_wgth[idx_tr] = outputw_wgth_notr[idx_notr];
        }
    }
    
    // Define dynamic arrays hidden/ouptut layers and initialize them with zeros
    m_input_ctrlu_u.resize(N_input,0.0f);
    m_input_ctrlu_v.resize(N_input_adjusted,0.0f);
    m_input_ctrlu_w.resize(N_input_adjusted,0.0f);
    m_input_ctrlv_u.resize(N_input_adjusted, 0.0f);
    m_input_ctrlv_v.resize(N_input, 0.0f);
    m_input_ctrlv_w.resize(N_input_adjusted, 0.0f);
    m_input_ctrlw_u.resize(N_input_adjusted, 0.0f);
    m_input_ctrlw_v.resize(N_input_adjusted, 0.0f);
    m_input_ctrlw_w.resize(N_input, 0.0f);
    m_output.resize(N_output,0.0f);
    m_output_zw.resize(N_output_zw,0.0f);

    //Read means+stdevs of training data for limiters in fluxes calculation
    std::string meansstd_filepath = "../../inferenceNN/Variables_MLP13/tavg_vert_prof.nc";

    //Read grid data needed to process means+stdevs
    auto& gd  = grid.get_grid_data();

    //Define IDs for netCDF-file needed for reading
    int retval = 0; //Status code for netCDF-function, needed for error handling
    int ncid_reading = 0;
    int varid_ucmean = 0;
    int varid_vcmean = 0;
    int varid_wcmean = 0;
    int varid_xumean = 0;
    int varid_yumean = 0;
    int varid_zumean = 0;
    int varid_xvmean = 0;
    int varid_yvmean = 0;
    int varid_zvmean = 0;
    int varid_xwmean = 0;
    int varid_ywmean = 0;
    int varid_zwmean = 0;
    int varid_ucstd  = 0;
    int varid_vcstd  = 0;
    int varid_wcstd  = 0;
    int varid_xustd  = 0;
    int varid_yustd  = 0;
    int varid_zustd  = 0;
    int varid_xvstd  = 0;
    int varid_yvstd  = 0;
    int varid_zvstd  = 0;
    int varid_xwstd  = 0;
    int varid_ywstd  = 0;
    int varid_zwstd  = 0;
    size_t count_zgc[1]  = {}; //initialize fixed arrays to 0
    size_t count_zhgc[1] = {};
    size_t count_z[1]  = {};
    size_t count_zh[1] = {};
    size_t start_z[1]  = {};

    //Resize dynamically allocated arrays means and stdevs
    int kcells = gd.ktot + 2*gd.kgc; //gd.kcells in for some reason undefined in this function, and needs to be calculated explicitly
    m_ucmean.resize(kcells);
    m_vcmean.resize(kcells);
    m_wcmean.resize(kcells+1);
    m_ucstd.resize(kcells);
    m_vcstd.resize(kcells);
    m_wcstd.resize(kcells+1);
    m_xumean.resize(gd.ktot);
    m_yumean.resize(gd.ktot);
    m_zumean.resize(gd.ktot+1);
    m_xvmean.resize(gd.ktot);
    m_yvmean.resize(gd.ktot);
    m_zvmean.resize(gd.ktot+1);
    m_xwmean.resize(gd.ktot+1);
    m_ywmean.resize(gd.ktot+1);
    m_zwmean.resize(gd.ktot);
    m_xustd.resize(gd.ktot);
    m_yustd.resize(gd.ktot);
    m_zustd.resize(gd.ktot+1);
    m_xvstd.resize(gd.ktot);
    m_yvstd.resize(gd.ktot);
    m_zvstd.resize(gd.ktot+1);
    m_xwstd.resize(gd.ktot+1);
    m_ywstd.resize(gd.ktot+1);
    m_zwstd.resize(gd.ktot);
    
    // Open nc-file  for reading
    if ((retval = nc_open(meansstd_filepath.c_str(), NC_NOWRITE, &ncid_reading)))
    {
        nc_error_print(retval);
    }

    // Get the varids of the variables based on their names
    if ((retval = nc_inq_varid(ncid_reading, "ucavgfields", &varid_ucmean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "vcavgfields", &varid_vcmean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "wcavgfields", &varid_wcmean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "ucstdfields", &varid_ucstd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "vcstdfields", &varid_vcstd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "wcstdfields", &varid_wcstd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresxuavgfields", &varid_xumean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresyuavgfields", &varid_yumean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unreszuavgfields", &varid_zumean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresxvavgfields", &varid_xvmean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresyvavgfields", &varid_yvmean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unreszvavgfields", &varid_zvmean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresxwavgfields", &varid_xwmean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresywavgfields", &varid_ywmean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unreszwavgfields", &varid_zwmean)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresxustdfields", &varid_xustd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresyustdfields", &varid_yustd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unreszustdfields", &varid_zustd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresxvstdfields", &varid_xvstd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresyvstdfields", &varid_yvstd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unreszvstdfields", &varid_zvstd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresxwstdfields", &varid_xwstd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unresywstdfields", &varid_ywstd)))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_inq_varid(ncid_reading, "unreszwstdfields", &varid_zwstd)))
    {
        nc_error_print(retval);
    }

    // Define settings such that the entire vertical profile is read from the nc-file
    count_zgc[0]  = kcells;
    count_zhgc[0] = kcells + 1;
    count_z[0]  = gd.ktot;
    count_zh[0] = gd.ktot + 1;
    start_z[0] = 0;

    //Extract vertical profiles from nc-file
    if ((retval = nc_get_vara_float(ncid_reading, varid_ucmean, start_z, count_zgc, &m_ucmean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_vcmean, start_z, count_zgc, &m_vcmean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_wcmean, start_z, count_zhgc, &m_wcmean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_ucstd , start_z, count_zgc, &m_ucstd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_vcstd , start_z, count_zgc, &m_vcstd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_wcstd , start_z, count_zhgc, &m_wcstd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_xumean, start_z, count_z, &m_xumean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_yumean, start_z, count_z, &m_yumean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_zumean, start_z, count_zh, &m_zumean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_xvmean, start_z, count_z, &m_xvmean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_yvmean, start_z, count_z, &m_yvmean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_zvmean, start_z, count_zh, &m_zvmean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_xwmean, start_z, count_zh, &m_xwmean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_ywmean, start_z, count_zh, &m_ywmean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_zwmean, start_z, count_z, &m_zwmean[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_xustd, start_z, count_z, &m_xustd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_yustd, start_z, count_z, &m_yustd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_zustd, start_z, count_zh, &m_zustd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_xvstd, start_z, count_z, &m_xvstd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_yvstd, start_z, count_z, &m_yvstd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_zvstd, start_z, count_zh, &m_zvstd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_xwstd, start_z, count_zh, &m_xwstd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_ywstd, start_z, count_zh, &m_ywstd[0])))
    {
        nc_error_print(retval);
    }
    if ((retval = nc_get_vara_float(ncid_reading, varid_zwstd, start_z, count_z, &m_zwstd[0])))
    {
        nc_error_print(retval);
    }

    //Close opened nc-file
    if((retval = nc_close(ncid_reading)))
    {
        nc_error_print(retval);
    }
}

template<typename TF>
Diff_NN<TF>::~Diff_NN()
{
}

template<typename TF>
void Diff_NN<TF>::init()
{
    boundary_cyclic.init();
}

template<typename TF>
Diffusion_type Diff_NN<TF>::get_switch() const
{
    return swdiff;
}

#ifndef USECUDA
template<typename TF>
unsigned long Diff_NN<TF>::get_time_limit(const unsigned long idt, const double dt)
{
    return idt * dnmax / (dt * dnmul);
}
#endif

#ifndef USECUDA
template<typename TF>
double Diff_NN<TF>::get_dn(const double dt)
{
    return dnmul*dt;
}
#endif

template<typename TF>
void Diff_NN<TF>::create(Stats<TF>& stats)
{
    auto& gd = grid.get_grid_data();

    // Get the maximum viscosity
    
    TF viscmax = fields.visc;
    for (auto& it : fields.sp)
        viscmax = std::max(it.second->visc, viscmax);

    // Calculate time step multiplier for diffusion number
    dnmul = 0;
    for (int k=gd.kstart; k<gd.kend; ++k)
        dnmul = std::max(dnmul, std::abs(viscmax * (1./(gd.dx*gd.dx) + 1./(gd.dy*gd.dy) + 1./(gd.dz[k]*gd.dz[k]))));

    stats.add_tendency(*fields.mt.at("u"), "z", tend_name, tend_longname);
    stats.add_tendency(*fields.mt.at("v"), "z", tend_name, tend_longname);
    stats.add_tendency(*fields.mt.at("w"), "zh", tend_name, tend_longname);
    for (auto it : fields.st)
        stats.add_tendency(*it.second, "z", tend_name, tend_longname);
}

#ifndef USECUDA
template<typename TF>
void Diff_NN<TF>::exec(Stats<TF>& stats)
{
    auto& gd  = grid.get_grid_data();
    
    diff_U(
    fields.mp.at("u")->fld.data(),
    fields.mp.at("v")->fld.data(),
    fields.mp.at("w")->fld.data(),
    fields.mt.at("u")->fld.data(),
    fields.mt.at("v")->fld.data(),
    fields.mt.at("w")->fld.data()
    );


    diff_c<TF>(fields.mt.at("u")->fld.data(), fields.mp.at("u")->fld.data(), fields.visc,
               gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend, gd.icells, gd.ijcells,
               gd.dx, gd.dy, gd.dzi.data(), gd.dzhi.data());

    diff_c<TF>(fields.mt.at("v")->fld.data(), fields.mp.at("v")->fld.data(), fields.visc,
               gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend, gd.icells, gd.ijcells,
               gd.dx, gd.dy, gd.dzi.data(), gd.dzhi.data());

    diff_w<TF>(fields.mt.at("w")->fld.data(), fields.mp.at("w")->fld.data(), fields.visc,
               gd.istart, gd.iend, gd.jstart, gd.jend, gd.kstart, gd.kend, gd.icells, gd.ijcells,
               gd.dx, gd.dy, gd.dzi.data(), gd.dzhi.data());

    stats.calc_tend(*fields.mt.at("u"), tend_name);
    stats.calc_tend(*fields.mt.at("v"), tend_name);
    stats.calc_tend(*fields.mt.at("w"), tend_name);
    for (auto it : fields.st)
        stats.calc_tend(*it.second, tend_name);
}
#endif

template<typename TF>
void Diff_NN<TF>::diff_flux(Field3d<TF>& restrict out, const Field3d<TF>& restrict fld_in)
{
    if (fld_in.loc[0] == 1)
    {
        calc_diff_flux_u(
                out.fld.data(), fld_in.fld.data(), 
                fields.mp.at("v")->fld.data(), fields.mp.at("w")->fld.data()
                );
    }
    else if (fld_in.loc[1] == 1)
    {
        calc_diff_flux_v(
                out.fld.data(), fields.mp.at("u")->fld.data(), 
                fld_in.fld.data(), fields.mp.at("w")->fld.data()
                );
    }
}

template class Diff_NN<float>;
template class Diff_NN<double>;
