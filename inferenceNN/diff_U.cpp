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
	const Grid& gd
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
				box_var[k_box * ji_box + j_box * (boxsize - skip_firstx - skip_lastx) + i_box] = field_var[k_field * gd.m_ijcells + j_field * gd.m_icells + i_field];
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
	const Grid& gd,
	Network& MLP
)
{
	// Initialize std::vectors for storing results mlp
	std::vector<float> result(MLP.N_output, 0.0f);
	std::vector<float> result_zw(MLP.N_output_zw, 0.0f);

	//Calculate inverse height differences
	const float dxi = 1.f / gd.m_dx;
	const float dyi = 1.f / gd.m_dy;

	//Loop over field
	//NOTE1: offset factors included to ensure alternate sampling
	for (int k = gd.m_kstart; k < gd.m_kend; ++k)
	{
		int k_offset = k % 2;
		for (int j = gd.m_jstart; j < gd.m_jend; ++j)
		{
			int offset = static_cast<int>((j % 2) == k_offset); //Calculate offset in such a way that the alternation swaps for each vertical level.
			for (int i = gd.m_istart + offset; i < gd.m_iend; i += 2)
			{
				//Extract grid box flow fields
				select_box(u, MLP.m_input_ctrlu_u.data(), k, j, i, MLP.boxsize, 0, 0, 0, 0, 0, 0, gd);
				select_box(v, MLP.m_input_ctrlu_v.data(), k, j, i, MLP.boxsize, 0, 0, 1, 0, 0, 1, gd);
				select_box(w, MLP.m_input_ctrlu_w.data(), k, j, i, MLP.boxsize, 1, 0, 0, 0, 0, 1, gd);
				select_box(u, MLP.m_input_ctrlv_u.data(), k, j, i, MLP.boxsize, 0, 0, 0, 1, 1, 0, gd);
				select_box(v, MLP.m_input_ctrlv_v.data(), k, j, i, MLP.boxsize, 0, 0, 0, 0, 0, 0, gd);
				select_box(w, MLP.m_input_ctrlv_w.data(), k, j, i, MLP.boxsize, 1, 0, 0, 1, 0, 0, gd);
				select_box(u, MLP.m_input_ctrlw_u.data(), k, j, i, MLP.boxsize, 0, 1, 0, 0, 1, 0, gd);
				select_box(v, MLP.m_input_ctrlw_v.data(), k, j, i, MLP.boxsize, 0, 1, 1, 0, 0, 0, gd);
				select_box(w, MLP.m_input_ctrlw_w.data(), k, j, i, MLP.boxsize, 0, 0, 0, 0, 0, 0, gd);


				//Execute mlp once for selected grid box
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
					MLP.m_output.data(), result.data(), false
				);

				//Check whether a horizontal boundary is reached, and if so make use of horizontal periodic BCs.
				int i_upbound = 0;
				int i_downbound = 0;
				int j_upbound = 0;
				int j_downbound = 0;
				// upstream boundary
				if (i == (gd.m_istart))
				{
					i_upbound = gd.m_iend - 1;
				}
				else
				{
					i_upbound = i - 1;
				}
				if (j == (gd.m_jstart))
				{
					j_upbound = gd.m_jend - 1;
				}
				else
				{
					j_upbound = j - 1;
				}
				// downstream boundary
				if (i == (gd.m_iend - 1))
				{
					i_downbound = gd.m_istart;
				}
				else
				{
					i_downbound = i + 1;
				}
				if (j == (gd.m_jend - 1))
				{
					j_downbound = gd.m_jstart;
				}
				else
				{
					j_downbound = j + 1;
				}

				//Calculate tendencies using predictions from mlp
				//xu_upstream
				ut[k*gd.m_ijcells + j * gd.m_icells + i] += -result[0] * dxi;
				ut[k*gd.m_ijcells + j * gd.m_icells + i_upbound] += result[0] * dxi;

				//xu_downstream
				ut[k*gd.m_ijcells + j * gd.m_icells + i] += result[1] * dxi;
				ut[k*gd.m_ijcells + j * gd.m_icells + i_downbound] += -result[1] * dxi;

				//yu_upstream
				ut[k*gd.m_ijcells + j * gd.m_icells + i] += -result[2] * dyi;
				ut[k*gd.m_ijcells + j_upbound * gd.m_icells + i] += result[2] * dyi;

				//yu_downstream
				ut[k*gd.m_ijcells + j * gd.m_icells + i] += result[3] * dyi;
				ut[k*gd.m_ijcells + j_downbound * gd.m_icells + i] += -result[3] * dyi;

				//zu_upstream
				if (k != gd.m_kstart)
					// NOTES: 1) zu_upstream is in this way implicitly set to 0 at the bottom layer
					// 2) ghost cell is not assigned.
				{
					ut[(k - 1)*gd.m_ijcells + j * gd.m_icells + i] += result[4] * dzi[k - 1];
					ut[k*gd.m_ijcells + j * gd.m_icells + i] += -result[4] * dzi[k];
				}

				//zu_downstream
				if (k != (gd.m_kend - 1))
					// NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
					// 2) ghost cell is not assigned.
				{
					ut[k*gd.m_ijcells + j * gd.m_icells + i] += result[5] * dzi[k];
					ut[(k + 1)*gd.m_ijcells + j * gd.m_icells + i] += -result[5] * dzi[k + 1];
				}

				//xv_upstream
				vt[k*gd.m_ijcells + j * gd.m_icells + i] += -result[6] * dxi;
				vt[k*gd.m_ijcells + j * gd.m_icells + i_upbound] += result[6] * dxi;

				//xv_downstream
				vt[k*gd.m_ijcells + j * gd.m_icells + i] += result[7] * dxi;
				vt[k*gd.m_ijcells + j * gd.m_icells + i_downbound] += -result[7] * dxi;

				//yv_upstream
				vt[k*gd.m_ijcells + j * gd.m_icells + i] += -result[8] * dyi;
				vt[k*gd.m_ijcells + j_upbound * gd.m_icells + i] += result[8] * dyi;

				//yv_downstream
				vt[k*gd.m_ijcells + j * gd.m_icells + i] += result[9] * dyi;
				vt[k*gd.m_ijcells + j_downbound * gd.m_icells + i] += -result[9] * dyi;

				//zv_upstream
				if (k != gd.m_kstart)
					// NOTES: 1) zu_upstream is in this way implicitly set to 0 at the bottom layer
					// 2) ghost cell is not assigned.
				{
					vt[(k - 1)*gd.m_ijcells + j * gd.m_icells + i] += result[10] * dzi[k - 1];
					vt[k*gd.m_ijcells + j * gd.m_icells + i] += -result[10] * dzi[k];
				}

				//zv_downstream
				if (k != (gd.m_kend - 1))
					// NOTES: 1) zu_downstream is in this way implicitly set to 0 at the top layer
					// 2) ghost cell is not assigned.
				{
					vt[k*gd.m_ijcells + j * gd.m_icells + i] += result[11] * dzi[k];
					vt[(k + 1)*gd.m_ijcells + j * gd.m_icells + i] += -result[11] * dzi[k + 1];
				}

				if (k != gd.m_kstart) //Don't adjust wt for bottom layer, should stay 0
				{
					//xw_upstream
					wt[k*gd.m_ijcells + j * gd.m_icells + i] += -result[12] * dxi;
					wt[k*gd.m_ijcells + j * gd.m_icells + i_upbound] += result[12] * dxi;

					//xw_downstream
					wt[k*gd.m_ijcells + j * gd.m_icells + i] += result[13] * dxi;
					wt[k*gd.m_ijcells + j * gd.m_icells + i_downbound] += -result[13] * dxi;

					//yw_upstream
					wt[k*gd.m_ijcells + j * gd.m_icells + i] += -result[14] * dyi;
					wt[k*gd.m_ijcells + j_upbound * gd.m_icells + i] += result[14] * dyi;

					//yw_downstream
					wt[k*gd.m_ijcells + j * gd.m_icells + i] += result[15] * dyi;
					wt[k*gd.m_ijcells + j_downbound * gd.m_icells + i] += -result[15] * dyi;

					//zu_upstream
					if (k != (gd.m_kstart + 1))
						//NOTE: Dont'adjust wt for bottom layer, should stay 0
					{
						wt[(k - 1)*gd.m_ijcells + j * gd.m_icells + i] += result[16] * dzhi[k - 1];
					}
					wt[k*gd.m_ijcells + j * gd.m_icells + i] += -result[16] * dzhi[k];

					//zu_downstream
					wt[k*gd.m_ijcells + j * gd.m_icells + i] += result[17] * dzhi[k];
					if (k != (gd.m_kend - 1))
						// NOTE:although this does not change wt at the bottom layer, 
						// it is still not included for k=0 to keep consistency between the top and bottom of the domain.
					{
						wt[(k + 1)*gd.m_ijcells + j * gd.m_icells + i] += -result[17] * dzhi[k + 1];
					}
				}

				// Execute for each iteration in the first layer above the bottom layer, and for each iteration in the top layer, 
				// the mlp for a second grid cell to calculate 'missing' zw-values.
				if ((k == (gd.m_kend - 1)) || (k == (gd.m_kstart + 1)))
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
					select_box(u, MLP.m_input_ctrlu_u.data(), k, j, i_2grid, MLP.boxsize, 0, 0, 0, 0, 0, 0, gd);
					select_box(v, MLP.m_input_ctrlu_v.data(), k, j, i_2grid, MLP.boxsize, 0, 0, 1, 0, 0, 1, gd);
					select_box(w, MLP.m_input_ctrlu_w.data(), k, j, i_2grid, MLP.boxsize, 1, 0, 0, 0, 0, 1, gd);
					select_box(u, MLP.m_input_ctrlv_u.data(), k, j, i_2grid, MLP.boxsize, 0, 0, 0, 1, 1, 0, gd);
					select_box(v, MLP.m_input_ctrlv_v.data(), k, j, i_2grid, MLP.boxsize, 0, 0, 0, 0, 0, 0, gd);
					select_box(w, MLP.m_input_ctrlv_w.data(), k, j, i_2grid, MLP.boxsize, 1, 0, 0, 1, 0, 0, gd);
					select_box(u, MLP.m_input_ctrlw_u.data(), k, j, i_2grid, MLP.boxsize, 0, 1, 0, 0, 1, 0, gd);
					select_box(v, MLP.m_input_ctrlw_v.data(), k, j, i_2grid, MLP.boxsize, 0, 1, 1, 0, 0, 0, gd);
					select_box(w, MLP.m_input_ctrlw_w.data(), k, j, i_2grid, MLP.boxsize, 0, 0, 0, 0, 0, 0, gd);

					//Execute mlp for selected second grid cell
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

					//Store calculated tendencies
					//zw_upstream
					if (k == (gd.m_kstart + 1))
					{
						wt[k * gd.m_ijcells + j * gd.m_icells + i_2grid] += -result_zw[0] * dzhi[k];
					}
					//zw_downstream
					else
					{
						wt[k * gd.m_ijcells + j * gd.m_icells + i_2grid] += result_zw[1] * dzhi[k];
					}
				}
			}
		}
	}
}