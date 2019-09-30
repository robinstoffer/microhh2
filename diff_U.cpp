// Function to calculate tendencies for unresolved momentum fluxes
#include <vector>
#include "Network.h"
#include "Grid.h"
#define restrict __restrict
 
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
	const Grid& grid
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
				box_var[k_box * ji_box + j_box * (boxsize - skip_firstx - skip_lastx) + i_box] = field_var[k_field * grid.m_ijcells + j_field * grid.m_icells + i_field];
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
	const float* restrict const dzi,
	const float* restrict const dzhi,
	float* restrict const ut,
	float* restrict const vt,
	float* restrict const wt,
	const Grid& grid,
	Network& MLP
)
{
	// Initialize std::vectors for storing results MLP
	std::vector<float> result(Network::N_output, 0.0f);
	std::vector<float> result_zw(Network::N_output_zw, 0.0f);
	
	//Calculate inverse height differences
	const float dxi = 1.f / grid.m_dx;
	const float dyi = 1.f / grid.m_dy;

	//Loop over field
	//NOTE1: offset factors included to ensure alternate sampling
	for (int k = grid.m_kstart; k < grid.m_kend; ++k)
	{
		int k_offset = k % 2;
		for (int j = grid.m_jstart; j < grid.m_jend; ++j)
		{
			int offset = static_cast<int>((j % 2) == k_offset); //Calculate offset in such a way that the alternation swaps for each vertical level.
			for (int i = grid.m_istart+offset; i < grid.m_iend; i+=2)
			{
				//Extract grid box flow fields
				select_box(u, MLP.m_input_ctrlu_u.data(), k, j, i, Network::boxsize, 0, 0, 0, 0, 0, 0, grid);
				select_box(v, MLP.m_input_ctrlu_v.data(), k, j, i, Network::boxsize, 0, 0, 1, 0, 0, 1, grid);
				select_box(w, MLP.m_input_ctrlu_w.data(), k, j, i, Network::boxsize, 1, 0, 0, 0, 0, 1, grid);
				select_box(u, MLP.m_input_ctrlv_u.data(), k, j, i, Network::boxsize, 0, 0, 0, 1, 1, 0, grid);
				select_box(v, MLP.m_input_ctrlv_v.data(), k, j, i, Network::boxsize, 0, 0, 0, 0, 0, 0, grid);
				select_box(w, MLP.m_input_ctrlv_w.data(), k, j, i, Network::boxsize, 1, 0, 0, 1, 0, 0, grid);
				select_box(u, MLP.m_input_ctrlw_u.data(), k, j, i, Network::boxsize, 0, 1, 0, 0, 1, 0, grid);
				select_box(v, MLP.m_input_ctrlw_v.data(), k, j, i, Network::boxsize, 0, 1, 1, 0, 0, 0, grid);
				select_box(w, MLP.m_input_ctrlw_w.data(), k, j, i, Network::boxsize, 0, 0, 0, 0, 0, 0, grid);
				

				//Execute MLP once for selected grid box
				Inference(
					MLP.m_input_ctrlu_u.data(), MLP.m_input_ctrlu_v.data(), MLP.m_input_ctrlu_w.data(),
					MLP.m_hiddenu_wgth.data(), MLP.m_hiddenu_bias.data(), MLP.m_hiddenu_alpha,
					MLP.m_outputu_wgth.data(), MLP.m_outputu_bias.data(),
					MLP.m_input_ctrlv_u.data(), MLP.m_input_ctrlv_v.data(), MLP.m_input_ctrlv_w.data(),
					MLP.m_hiddenv_wgth.data(), MLP.m_hiddenv_bias.data(), MLP.m_hiddenv_alpha,
					MLP.m_outputv_wgth.data(), MLP.m_outputv_bias.data(),
					MLP.m_input_ctrlw_u.data(), MLP.m_input_ctrlw_v.data(),  MLP.m_input_ctrlw_w.data(),
					MLP.m_hiddenw_wgth.data(), MLP.m_hiddenw_bias.data(), MLP.m_hiddenw_alpha,
					MLP.m_outputw_wgth.data(), MLP.m_outputw_bias.data(),
					MLP.m_mean_input.data(), MLP.m_stdev_input.data(),
					MLP.m_mean_label.data(), MLP.m_stdev_label.data(),
					MLP.m_utau_ref, MLP.m_output_denorm_utau2,
					MLP.m_output.data(), result.data(), false
					);

				//Calculate indices without ghost cells for storage of the tendencies
				int k_nogc = k - grid.m_kstart;
				int j_nogc = j - grid.m_jstart;
				int i_nogc = i - grid.m_istart;
				int k_1gc = k_nogc + 1;

				//Check whether a horizontal boundary is reached, and if so make use of horizontal periodic BCs.
				int i_nogc_upbound = 0;
				int i_nogc_downbound = 0;
				int j_nogc_upbound = 0;
				int j_nogc_downbound = 0;
				// upstream boundary
				if (i == (grid.m_istart))
				{
					i_nogc_upbound = grid.m_itot - 1;
				}
				else
				{
					i_nogc_upbound = i_nogc - 1;
				}
				if (j == (grid.m_jstart))
				{
					j_nogc_upbound = grid.m_jtot - 1;
				}
				else
				{
					j_nogc_upbound = j_nogc - 1;
				}
				// downstream boundary
				if (i == (grid.m_iend - 1))
				{
					i_nogc_downbound = 0;
				}
				else
				{
					i_nogc_downbound = i_nogc + 1;
				}
				if (j == (grid.m_jend - 1))
				{
					j_nogc_downbound = 0;
				}
				else
				{
					j_nogc_downbound = j_nogc + 1;
				}

				//Calculate tendencies using predictions from MLP
				//xu_upstream
				ut[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]         += -result[0] * dxi;
				ut[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc_upbound] +=  result[0] * dxi;

				//xu_downstream
				ut[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]           +=  result[1] * dxi;
				ut[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc_downbound] += -result[1] * dxi;

				//yu_upstream
				ut[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]         += -result[2] * dyi;
				ut[k_nogc*grid.m_ijtot + j_nogc_upbound * grid.m_itot + i_nogc] +=  result[2] * dyi;

				//yu_downstream
				ut[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]           +=  result[3] * dyi;
				ut[k_nogc*grid.m_ijtot + j_nogc_downbound * grid.m_itot + i_nogc] += -result[3] * dyi;

				//zu_upstream
				if (k != grid.m_kstart)
					// NOTES: 1) zu_upstream is in this way implicitly set to 0 at the bottom layer
					// 2) ghost cell is not assigned.
				{
					ut[(k_nogc-1)*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc] +=  result[4] * dzi[k_1gc-1];
					ut[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]     += -result[4] * dzi[k_1gc];
				}

				//zu_downstream
				if (k != (grid.m_kend - 1))
					// NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
					// 2) ghost cell is not assigned.
				{
					ut[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]     +=  result[5] * dzi[k_1gc];
					ut[(k_nogc+1)*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc] += -result[5] * dzi[k_1gc+1];
				}

				//xv_upstream
				vt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]         += -result[6] * dxi;
				vt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc_upbound] +=  result[6] * dxi;

				//xv_downstream
				vt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]           +=  result[7] * dxi;
				vt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc_downbound] += -result[7] * dxi;

				//yv_upstream
				vt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]         += -result[8] * dyi;
				vt[k_nogc*grid.m_ijtot + j_nogc_upbound * grid.m_itot + i_nogc] +=  result[8] * dyi;

				//yv_downstream
				vt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]           +=  result[9] * dyi;
				vt[k_nogc*grid.m_ijtot + j_nogc_downbound * grid.m_itot + i_nogc] += -result[9] * dyi;

				//zv_upstream
				if (k != grid.m_kstart)
					// NOTES: 1) zu_upstream is in this way implicitly set to 0 at the bottom layer
					// 2) ghost cell is not assigned.
				{
					vt[(k_nogc - 1)*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc] +=  result[10] * dzi[k_1gc - 1];
					vt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]       += -result[10] * dzi[k_1gc];
				}

				//zv_downstream
				if (k != (grid.m_kend - 1))
					// NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
					// 2) ghost cell is not assigned.
				{
					vt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]       +=  result[11] * dzi[k_1gc];
					vt[(k_nogc + 1)*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc] += -result[11] * dzi[k_1gc + 1];
				}

				if (k != grid.m_kstart) //Don't adjust wt for bottom layer, should stay 0
				{
					//xw_upstream
					wt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]         += -result[12] * dxi;
					wt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc_upbound] +=  result[12] * dxi;

					//xw_downstream
					wt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]           +=  result[13] * dxi;
					wt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc_downbound] += -result[13] * dxi;

					//yw_upstream
					wt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]         += -result[14] * dyi;
					wt[k_nogc*grid.m_ijtot + j_nogc_upbound * grid.m_itot + i_nogc] +=  result[14] * dyi;

					//yw_downstream
					wt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]           +=  result[15] * dyi;
					wt[k_nogc*grid.m_ijtot + j_nogc_downbound * grid.m_itot + i_nogc] += -result[15] * dyi;

					//zu_upstream
					if (k != (grid.m_kstart+1))
						//NOTE: Dont'adjust wt for bottom layer, should stay 0
					{
						wt[(k_nogc - 1)*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc] +=  result[16] * dzhi[k_1gc - 1];
					}
					wt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]           += -result[16] * dzhi[k_1gc];

					//zu_downstream
					wt[k_nogc*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc]           +=  result[17] * dzhi[k_1gc];
					if (k != (grid.m_kend - 1))
					// NOTE:although this does not change wt at the bottom layer, 
					// it is still not included for k=0 to keep consistency between the top and bottom of the domain.
					{
						wt[(k_nogc + 1)*grid.m_ijtot + j_nogc * grid.m_itot + i_nogc] += -result[17] * dzhi[k_1gc + 1];
					}
				}

				// Execute for each iteration in the first layer above the bottom layer, and for each iteration in the top layer, 
				// the MLP for a second grid cell to calculate 'missing' zw-values.
				if ((k == (grid.m_kend - 1)) || (k == (grid.m_kstart + 1)))
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
					select_box(u, MLP.m_input_ctrlu_u.data(), k, j, i_2grid, Network::boxsize, 0, 0, 0, 0, 0, 0, grid);
					select_box(v, MLP.m_input_ctrlu_v.data(), k, j, i_2grid, Network::boxsize, 0, 0, 1, 0, 0, 1, grid);
					select_box(w, MLP.m_input_ctrlu_w.data(), k, j, i_2grid, Network::boxsize, 1, 0, 0, 0, 0, 1, grid);
					select_box(u, MLP.m_input_ctrlv_u.data(), k, j, i_2grid, Network::boxsize, 0, 0, 0, 1, 1, 0, grid);
					select_box(v, MLP.m_input_ctrlv_v.data(), k, j, i_2grid, Network::boxsize, 0, 0, 0, 0, 0, 0, grid);
					select_box(w, MLP.m_input_ctrlv_w.data(), k, j, i_2grid, Network::boxsize, 1, 0, 0, 1, 0, 0, grid);
					select_box(u, MLP.m_input_ctrlw_u.data(), k, j, i_2grid, Network::boxsize, 0, 1, 0, 0, 1, 0, grid);
					select_box(v, MLP.m_input_ctrlw_v.data(), k, j, i_2grid, Network::boxsize, 0, 1, 1, 0, 0, 0, grid);
					select_box(w, MLP.m_input_ctrlw_w.data(), k, j, i_2grid, Network::boxsize, 0, 0, 0, 0, 0, 0, grid);

					//Execute MLP for selected second grid cell
					Inference(
						MLP.m_input_ctrlu_u.data(), MLP.m_input_ctrlu_v.data(), MLP.m_input_ctrlu_w.data(),
						MLP.m_hiddenu_wgth.data(), MLP.m_hiddenu_bias.data(), MLP.m_hiddenu_alpha,
						MLP.m_outputu_wgth.data(), MLP.m_outputu_bias.data(),
						MLP.m_input_ctrlv_u.data(), MLP.m_input_ctrlv_v.data(), MLP.m_input_ctrlv_w.data(),
						MLP.m_hiddenv_wgth.data(), MLP.m_hiddenv_bias.data(), MLP.m_hiddenv_alpha,
						MLP.m_outputv_wgth.data(), MLP.m_outputv_bias.data(),
						MLP.m_input_ctrlw_u.data(), MLP.m_input_ctrlw_v.data(), MLP.m_input_ctrlw_w.data(),
						MLP.m_hiddenw_wgth.data(), MLP.m_hiddenw_bias.data(), MLP.m_hiddenw_alpha,
						MLP.m_outputw_wgth.data(), MLP.m_outputw_bias.data(),
						MLP.m_mean_input.data(), MLP.m_stdev_input.data(),
						MLP.m_mean_label.data(), MLP.m_stdev_label.data(),
						MLP.m_utau_ref, MLP.m_output_denorm_utau2,
						MLP.m_output_zw.data(), result_zw.data(), true
					);

					//Calculate new indices for storage
					int k_nogc2 = k - grid.m_kstart;
					int j_nogc2 = j - grid.m_jstart;
					int i_nogc2 = i_2grid - grid.m_istart;
					int k_1gc2  = k_nogc2 + 1;

					//Store calculated tendencies
					//zw_upstream
					if (k == (grid.m_kstart + 1))
					{
						wt[k_nogc2 * grid.m_ijtot + j_nogc2 * grid.m_itot + i_nogc2] += -result_zw[0] * dzhi[k_1gc2];
					}
					//zw_downstream
					else
					{
						wt[k_nogc2 * grid.m_ijtot + j_nogc2 * grid.m_itot + i_nogc2] += result_zw[1] * dzhi[k_1gc2];
					}			
				}
			}
		}
	}
}