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
#include <vector>

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
#include "network.h"

namespace
{
	void select_box(
		const float* restrict const field_var,
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
		const int skip_lastx,
	)
	// NOTE: the skip_* integers specify whether the index indicated in the name should be skipped in the selection of box (0=don't skip, 1=skip it).
	{
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
					box_var[k_box * ji_box + j_box * (boxsize - skip_firstx - skip_lastx) + i_box] = field_var[k_field * gd.ijcells + j_field * gd.icells + i_field];
					i_box += 1;
				}
				j_box += 1;
			}
			k_box += 1;
		}
	}
	
	// Function that loops over the whole flow field, and calculates for each grid cell the tendencies
	void diff_U(
		const float* restrict const u,
		const float* restrict const v,
		const float* restrict const w,
		float* restrict const ut,
		float* restrict const vt,
		float* restrict const wt,
	)
	{
		auto& gd  = grid.get_grid_data()

		// Initialize std::vectors for storing results mlp
		std::vector<float> result(Network::N_output, 0.0f);
		std::vector<float> result_zw(Network::N_output_zw, 0.0f);
		
		//Calculate inverse height differences
		const float dxi = 1.f / gd.dx;
		const float dyi = 1.f / gd.dy;
	
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
					select_box(u, mlp->m_input_ctrlu_u.data(), k, j, i, Network::boxsize, 0, 0, 0, 0, 0, 0);
					select_box(v, mlp->m_input_ctrlu_v.data(), k, j, i, Network::boxsize, 0, 0, 1, 0, 0, 1);
					select_box(w, mlp->m_input_ctrlu_w.data(), k, j, i, Network::boxsize, 1, 0, 0, 0, 0, 1);
					select_box(u, mlp->m_input_ctrlv_u.data(), k, j, i, Network::boxsize, 0, 0, 0, 1, 1, 0);
					select_box(v, mlp->m_input_ctrlv_v.data(), k, j, i, Network::boxsize, 0, 0, 0, 0, 0, 0);
					select_box(w, mlp->m_input_ctrlv_w.data(), k, j, i, Network::boxsize, 1, 0, 0, 1, 0, 0);
					select_box(u, mlp->m_input_ctrlw_u.data(), k, j, i, Network::boxsize, 0, 1, 0, 0, 1, 0);
					select_box(v, mlp->m_input_ctrlw_v.data(), k, j, i, Network::boxsize, 0, 1, 1, 0, 0, 0);
					select_box(w, mlp->m_input_ctrlw_w.data(), k, j, i, Network::boxsize, 0, 0, 0, 0, 0, 0);
					
	
					//Execute mlp once for selected grid box
					Inference(
						mlp->m_input_ctrlu_u.data(), mlp->m_input_ctrlu_v.data(), mlp->m_input_ctrlu_w.data(),
						mlp->m_hiddenu_wgth.data(), mlp->m_hiddenu_bias.data(), mlp->m_hiddenu_alpha,
						mlp->m_outputu_wgth.data(), mlp->m_outputu_bias.data(),
						mlp->m_input_ctrlv_u.data(), mlp->m_input_ctrlv_v.data(), mlp->m_input_ctrlv_w.data(),
						mlp->m_hiddenv_wgth.data(), mlp->m_hiddenv_bias.data(), mlp->m_hiddenv_alpha,
						mlp->m_outputv_wgth.data(), mlp->m_outputv_bias.data(),
						mlp->m_input_ctrlw_u.data(), mlp->m_input_ctrlw_v.data(),  mlp->m_input_ctrlw_w.data(),
						mlp->m_hiddenw_wgth.data(), mlp->m_hiddenw_bias.data(), mlp->m_hiddenw_alpha,
						mlp->m_outputw_wgth.data(), mlp->m_outputw_bias.data(),
						mlp->m_mean_input.data(), mlp->m_stdev_input.data(),
						mlp->m_mean_label.data(), mlp->m_stdev_label.data(),
						mlp->m_utau_ref, mlp->m_output_denorm_utau2,
						mlp->m_output.data(), result.data(), false
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
						ut[(k_nogc-1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] +=  result[4] * dzi[k_1gc-1];
						ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]     += -result[4] * dzi[k_1gc];
					}
	
					//zu_downstream
					if (k != (gd.kend - 1))
						// NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
						// 2) ghost cell is not assigned.
					{
						ut[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]     +=  result[5] * dzi[k_1gc];
						ut[(k_nogc+1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] += -result[5] * dzi[k_1gc+1];
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
						vt[(k_nogc - 1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] +=  result[10] * dzi[k_1gc - 1];
						vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]       += -result[10] * dzi[k_1gc];
					}
	
					//zv_downstream
					if (k != (gd.kend - 1))
						// NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
						// 2) ghost cell is not assigned.
					{
						vt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]       +=  result[11] * dzi[k_1gc];
						vt[(k_nogc + 1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] += -result[11] * dzi[k_1gc + 1];
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
							wt[(k_nogc - 1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] +=  result[16] * dzhi[k_1gc - 1];
						}
						wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]           += -result[16] * dzhi[k_1gc];
	
						//zu_downstream
						wt[k_nogc*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc]           +=  result[17] * dzhi[k_1gc];
						if (k != (gd.kend - 1))
						// NOTE:although this does not change wt at the bottom layer, 
						// it is still not included for k=0 to keep consistency between the top and bottom of the domain.
						{
							wt[(k_nogc + 1)*gd.jtot*gd.itot + j_nogc * gd.itot + i_nogc] += -result[17] * dzhi[k_1gc + 1];
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
						select_box(u, mlp->m_input_ctrlu_u.data(), k, j, i_2grid, Network::boxsize, 0, 0, 0, 0, 0, 0, grid);
						select_box(v, mlp->m_input_ctrlu_v.data(), k, j, i_2grid, Network::boxsize, 0, 0, 1, 0, 0, 1, grid);
						select_box(w, mlp->m_input_ctrlu_w.data(), k, j, i_2grid, Network::boxsize, 1, 0, 0, 0, 0, 1, grid);
						select_box(u, mlp->m_input_ctrlv_u.data(), k, j, i_2grid, Network::boxsize, 0, 0, 0, 1, 1, 0, grid);
						select_box(v, mlp->m_input_ctrlv_v.data(), k, j, i_2grid, Network::boxsize, 0, 0, 0, 0, 0, 0, grid);
						select_box(w, mlp->m_input_ctrlv_w.data(), k, j, i_2grid, Network::boxsize, 1, 0, 0, 1, 0, 0, grid);
						select_box(u, mlp->m_input_ctrlw_u.data(), k, j, i_2grid, Network::boxsize, 0, 1, 0, 0, 1, 0, grid);
						select_box(v, mlp->m_input_ctrlw_v.data(), k, j, i_2grid, Network::boxsize, 0, 1, 1, 0, 0, 0, grid);
						select_box(w, mlp->m_input_ctrlw_w.data(), k, j, i_2grid, Network::boxsize, 0, 0, 0, 0, 0, 0, grid);
	
						//Execute mlp for selected second grid cell
						Inference(
							mlp->m_input_ctrlu_u.data(), mlp->m_input_ctrlu_v.data(), mlp->m_input_ctrlu_w.data(),
							mlp->m_hiddenu_wgth.data(), mlp->m_hiddenu_bias.data(), mlp->m_hiddenu_alpha,
							mlp->m_outputu_wgth.data(), mlp->m_outputu_bias.data(),
							mlp->m_input_ctrlv_u.data(), mlp->m_input_ctrlv_v.data(), mlp->m_input_ctrlv_w.data(),
							mlp->m_hiddenv_wgth.data(), mlp->m_hiddenv_bias.data(), mlp->m_hiddenv_alpha,
							mlp->m_outputv_wgth.data(), mlp->m_outputv_bias.data(),
							mlp->m_input_ctrlw_u.data(), mlp->m_input_ctrlw_v.data(), mlp->m_input_ctrlw_w.data(),
							mlp->m_hiddenw_wgth.data(), mlp->m_hiddenw_bias.data(), mlp->m_hiddenw_alpha,
							mlp->m_outputw_wgth.data(), mlp->m_outputw_bias.data(),
							mlp->m_mean_input.data(), mlp->m_stdev_input.data(),
							mlp->m_mean_label.data(), mlp->m_stdev_label.data(),
							mlp->m_utau_ref, mlp->m_output_denorm_utau2,
							mlp->m_output_zw.data(), result_zw.data(), true
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
							wt[k_nogc2 * gd.jtot*gd.itot + j_nogc2 * gd.itot + i_nogc2] += -result_zw[0] * dzhi[k_1gc2];
						}
						//zw_downstream
						else
						{
							wt[k_nogc2 * gd.jtot*gd.itot + j_nogc2 * gd.itot + i_nogc2] += result_zw[1] * dzhi[k_1gc2];
						}			
					}
				}
			}
		}
	}

	void hidden_layer1(
		const float* restrict const weights,
		const float* restrict const bias,
		const float* restrict const input,
		float* restrict const layer_out,
		const float alpha
	)
	{
		// Calculate hidden neurons outputs as matrix vector multiplication using BLAS
		cblas_sgemv(CblasRowMajor, CblasNoTrans, Network::N_hidden, Network::N_input_tot_adjusted, 
			1., weights, Network::N_input_tot_adjusted, input, 1, 0, layer_out, 1);
		
		//Loop over hidden neurons to add bias and calculate activations using Leaky ReLu
		for (int hiddenidx = 0; hiddenidx < Network::N_hidden; ++hiddenidx)
		{
			layer_out[hiddenidx] += bias[hiddenidx];
			layer_out[hiddenidx] = std::max(alpha * layer_out[hiddenidx], layer_out[hiddenidx]);
		}
	}  
	//output layer
	void output_layer(
		const float* restrict const weights,
		const float* restrict const bias,
		const float* restrict const layer_in,
		float* restrict const layer_out
	)
	{
		// Calculate hidden neurons outputs as matrix vector multiplication using BLAS
		cblas_sgemv(CblasRowMajor, CblasNoTrans, Network::N_output_control, Network::N_hidden,
			1., weights, Network::N_hidden, layer_in, 1, 0, layer_out, 1);
	
		//Loop over hidden neurons to add bias
		for (int outputidx = 0; outputidx < Network::N_output_control; ++outputidx)
		{
			layer_out[outputidx] += bias[outputidx];
		}
	}
	
	
	void Inference(
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
		std::array<float, Network::N_input_tot_adjusted> input_ctrlu;
		std::array<float, Network::N_input_tot_adjusted> input_ctrlv;
		std::array<float, Network::N_input_tot_adjusted> input_ctrlw;
	
	    	// Normalize with mean, st. dev, and utau_ref.
		constexpr int N_input           = Network::N_input;
		constexpr int N_input_adjusted  = Network::N_input_adjusted;
		constexpr int N_input_adjusted2 = 2 * N_input_adjusted;
		constexpr int N_input_comb2     = N_input_adjusted + N_input;
		for (int inpidx = 0; inpidx < Network::N_input;++inpidx)
		{
			input_ctrlu[inpidx]                     = (((input_ctrlu_u[inpidx] / utau_ref) - mean_input[0]) / stdev_input[0]);
			input_ctrlv[N_input_adjusted + inpidx]  = (((input_ctrlv_v[inpidx] / utau_ref) - mean_input[1]) / stdev_input[1]);
			input_ctrlw[N_input_adjusted2 + inpidx] = (((input_ctrlw_w[inpidx] / utau_ref) - mean_input[2]) / stdev_input[2]);
		}
		for (int inpidx = 0; inpidx < Network::N_input_adjusted; ++inpidx)
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
		std::array<float, Network::N_hidden> hiddenu;
		hidden_layer1(hiddenu_wgth, hiddenu_bias,
			input_ctrlu.data(), hiddenu.data(), hiddenu_alpha);
	
		//output layer
		std::array<float, Network::N_output_control> outputu;
		output_layer(outputu_wgth, outputu_bias, hiddenu.data(), outputu.data());
	
		//control volume v
	
		//hidden layer
		std::array<float, Network::N_hidden> hiddenv;
		hidden_layer1(hiddenv_wgth, hiddenv_bias,
			input_ctrlv.data(), hiddenv.data(), hiddenv_alpha);
	
		//output layer
		std::array<float, Network::N_output_control> outputv;
		output_layer(outputv_wgth, outputv_bias, hiddenv.data(), outputv.data());
	
		//control volume w
	
		//hidden layer
		std::array<float, Network::N_hidden> hiddenw;
		hidden_layer1(hiddenw_wgth, hiddenw_bias,
			input_ctrlw.data(), hiddenw.data(), hiddenw_alpha);
	
		//output layer
		std::array<float, Network::N_output_control> outputw;
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
			for (int outputidx = 0; outputidx < Network::N_output; ++outputidx)
			{
				output_denorm[outputidx] = ((output[outputidx] * stdev_label[outputidx]) + mean_label[outputidx]) * output_denorm_utau2;
			}
		}
	}
} // End namespace.

template<typename TF>
Diff_NN<TF>::Diff_NN(Master& masterin, Grid<TF>& gridin, Fields<TF>& fieldsin, Boundary<TF>& boundaryin, Input& inputin) :
    Diff<TF>(masterin, gridin, fieldsin, boundaryin, inputin),
    boundary_cyclic(master, grid)
    field3d_operators(master, grid, fields)
{
    dnmax = inputin.get_item<TF>("diff", "dnmax", "", 0.4  );

    if (grid.get_spatial_order() != Grid_order::Second)
        throw std::runtime_error("Diff_NN only runs with second order grids");
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
    void diff_U(
	fields.mp.at("u")->fld.data(),
	fields.mp.at("v")->fld.data(),
	fields.mp.at("w")->fld.data(),
	gd.dzi.data(),
	gd.dzhi.data(),
	fields.mt.at("u")->fld.data(),
	fields.mt.at("v")->fld.data(),
	fields.mt.at("w")->fld.data(),
	)
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
template class Diff_NN<double>;
template class Diff_NN<float>;
