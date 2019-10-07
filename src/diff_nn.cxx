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
}

namespace
{
    template<typename TF>
    void diff_c(TF* restrict at, const TF* restrict a, const TF visc,
                const int istart, const int iend, const int jstart, const int jend, const int kstart, const int kend,
                const int jj, const int kk, const TF dx, const TF dy, const TF* restrict dzi, const TF* restrict dzhi)
    {
        const int ii = 1;
        const double dxidxi = 1/(dx*dx);
        const double dyidyi = 1/(dy*dy);

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
        const double dxidxi = 1/(dx*dx);
        const double dyidyi = 1/(dy*dy);

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

// Function that loops over the whole flow field, and calculates for each grid cell the tendencies
template<typename TF>
void Diff_NN<TF>::diff_U(
	const TF* restrict const u,
	const TF* restrict const v,
	const TF* restrict const w,
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
	const float dxi = 1.f / gd.dx;
	const float dyi = 1.f / gd.dy;

	//Loop over field
	//NOTE1: offset factors included to ensure alternate sampling
	for (int k = gd.kstart; k < gd.kend; ++k)
	{
        std::cout << "k: " << std::to_string(k) << "\n";
        std::cout << "wt: " << std::to_string(wt[0,0,1]) << "\n";
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
				

				//Execute mlp once for selected grid box
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

				//Calculate indices without ghost cells for storage of the tendencies
				int k_nogc = k - gd.kstart;
				int j_nogc = j - gd.jstart;
				int i_nogc = i - gd.istart;
				int k_1gc = k_nogc + 1;

				//Check whether a horizontal boundary is reached, and if so make use of horizontal periodic BCs.
				int i_nogc_upbound = 0;
				int i_nogc_downbound = 0;
				int j_nogc_upbound = 0;
				int j_nogc_downbound = 0;
				// upstream boundary
				if (i == (gd.istart))
				{
					i_nogc_upbound = gd.itot - 1;
				}
				else
				{
					i_nogc_upbound = i_nogc - 1;
				}
				if (j == (gd.jstart))
				{
					j_nogc_upbound = gd.jtot - 1;
				}
				else
				{
					j_nogc_upbound = j_nogc - 1;
				}
				// downstream boundary
				if (i == (gd.iend - 1))
				{
					i_nogc_downbound = 0;
				}
				else
				{
					i_nogc_downbound = i_nogc + 1;
				}
				if (j == (gd.jend - 1))
				{
					j_nogc_downbound = 0;
				}
				else
				{
					j_nogc_downbound = j_nogc + 1;
				}

				//Calculate tendencies using predictions from mlp
				//xu_upstream
				ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]         += -result[0] * dxi;
				ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc_upbound] +=  result[0] * dxi;

				//xu_downstream
				ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot+ i_nogc]           +=  result[1] * dxi;
				ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot+ i_nogc_downbound] += -result[1] * dxi;

				//yu_upstream
				ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]         += -result[2] * dyi;
				ut[k_nogc*gd.jtot*gd.itot + j_nogc_upbound * gd.itot + i_nogc] +=  result[2] * dyi;

				//yu_downstream
				ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]           +=  result[3] * dyi;
				ut[k_nogc*gd.jtot*gd.itot + j_nogc_downbound * gd.itot + i_nogc] += -result[3] * dyi;

				//zu_upstream
				if (k != gd.kstart)
					// NOTES: 1) zu_upstream is in this way implicitly set to 0 at the bottom layer
					// 2) ghost cell is not assigned.
				{
					ut[(k_nogc-1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] +=  result[4] * gd.dzi[k_1gc-1];
					ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]     += -result[4] * gd.dzi[k_1gc];
				}

				//zu_downstream
				if (k != (gd.kend - 1))
					// NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
					// 2) ghost cell is not assigned.
				{
					ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]     +=  result[5] * gd.dzi[k_1gc];
					ut[(k_nogc+1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] += -result[5] * gd.dzi[k_1gc+1];
				}

				//xv_upstream
				vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]         += -result[6] * dxi;
				vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc_upbound] +=  result[6] * dxi;

				//xv_downstream
				vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]           +=  result[7] * dxi;
				vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc_downbound] += -result[7] * dxi;

				//yv_upstream
				vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]         += -result[8] * dyi;
				vt[k_nogc*gd.jtot*gd.itot + j_nogc_upbound * gd.itot + i_nogc] +=  result[8] * dyi;

				//yv_downstream
				vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]           +=  result[9] * dyi;
				vt[k_nogc*gd.jtot*gd.itot + j_nogc_downbound * gd.itot + i_nogc] += -result[9] * dyi;

				//zv_upstream
				if (k != gd.kstart)
					// NOTES: 1) zu_upstream is in this way implicitly set to 0 at the bottom layer
					// 2) ghost cell is not assigned.
				{
					vt[(k_nogc - 1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] +=  result[10] * gd.dzi[k_1gc - 1];
					vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]       += -result[10] * gd.dzi[k_1gc];
				}

				//zv_downstream
				if (k != (gd.kend - 1))
					// NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
					// 2) ghost cell is not assigned.
				{
					vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]       +=  result[11] * gd.dzi[k_1gc];
					vt[(k_nogc + 1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] += -result[11] * gd.dzi[k_1gc + 1];
				}

				if (k != gd.kstart) //Don't adjust wt for bottom layer, should stay 0
				{
					//xw_upstream
					wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]         += -result[12] * dxi;
					wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc_upbound] +=  result[12] * dxi;

					//xw_downstream
					wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]           +=  result[13] * dxi;
					wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc_downbound] += -result[13] * dxi;

					//yw_upstream
					wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]         += -result[14] * dyi;
					wt[k_nogc*gd.jtot*gd.itot + j_nogc_upbound * gd.itot + i_nogc] +=  result[14] * dyi;

					//yw_downstream
					wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]           +=  result[15] * dyi;
					wt[k_nogc*gd.jtot*gd.itot + j_nogc_downbound * gd.itot + i_nogc] += -result[15] * dyi;

					//zu_upstream
					if (k != (gd.kstart+1))
						//NOTE: Dont'adjust wt for bottom layer, should stay 0
					{
						wt[(k_nogc - 1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] +=  result[16] * gd.dzhi[k_1gc - 1];
					}
					wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]           += -result[16] * gd.dzhi[k_1gc];

					//zu_downstream
					wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]           +=  result[17] * gd.dzhi[k_1gc];
					if (k != (gd.kend - 1))
					// NOTE:although this does not change wt at the bottom layer, 
					// it is still not included for k=0 to keep consistency between the top and bottom of the domain.
					{
						wt[(k_nogc + 1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] += -result[17] * gd.dzhi[k_1gc + 1];
					}
				}

				// Execute for each iteration in the first layer above the bottom layer, and for each iteration in the top layer, 
				// the mlp for a second grid cell to calculate 'missing' zw-values.
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

					//Execute mlp for selected second grid cell
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

					//Calculate new indices for storage
					int k_nogc2 = k - gd.kstart;
					int j_nogc2 = j - gd.jstart;
					int i_nogc2 = i_2grid - gd.istart;
					int k_1gc2  = k_nogc2 + 1;

					//Store calculated tendencies
					//zw_upstream
					if (k == (gd.kstart + 1))
					{
						wt[k_nogc2 * gd.jtot*gd.itot + j_nogc2 * gd.itot + i_nogc2] += -result_zw[0] * gd.dzhi[k_1gc2];
					}
					//zw_downstream
					else
					{
						wt[k_nogc2 * gd.jtot*gd.itot + j_nogc2 * gd.itot + i_nogc2] += result_zw[1] * gd.dzhi[k_1gc2];
					}			
				}
			}
		}
	}
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

}

template<typename TF>
Diff_NN<TF>::~Diff_NN()
{
}

template<typename TF>
void Diff_NN<TF>::init()
{
}

template<typename TF>
Diffusion_type Diff_NN<TF>::get_switch() const
{
}

#ifndef USECUDA
template<typename TF>
unsigned long Diff_NN<TF>::get_time_limit(const unsigned long idt, const double dt)
{
}
#endif

#ifndef USECUDA
template<typename TF>
double Diff_NN<TF>::get_dn(const double dt)
{
}
#endif

template<typename TF>
void Diff_NN<TF>::create(Stats<TF>& stats)
{
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

}

template<typename TF>
void Diff_NN<TF>::exec_viscosity(Thermo<TF>& thermo)
{
}
#endif

template<typename TF>
void Diff_NN<TF>::create_stats(Stats<TF>& stats)
{
}

template<typename TF>
void Diff_NN<TF>::exec_stats(Stats<TF>& stats)
{
}

template<typename TF>
void Diff_NN<TF>::diff_flux(Field3d<TF>& restrict out, const Field3d<TF>& restrict fld_in)
{
}

template class Diff_NN<float>;
template class Diff_NN<double>;
