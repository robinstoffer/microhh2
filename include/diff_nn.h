/*
 * MicroHH
 * Copyright (c) 2011-2017 Chiel van Heerwaarden
 * Copyright (c) 2011-2017 Thijs Heus
 * Copyright (c) 2014-2017 Bart van Stratum
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

#ifndef DIFF_NN_H
#define DIFF_NN_H

#include <vector>
#include <string>
#include "diff.h"
#include "boundary_cyclic.h"
#include "field3d_operators.h"

template<typename> class Stats;

template<typename TF>
class Diff_NN : public Diff<TF>
{
    public:
        Diff_NN(Master&, Grid<TF>&, Fields<TF>&, Boundary<TF>&, Input&);
        ~Diff_NN();

        Diffusion_type get_switch() const;
        unsigned long get_time_limit(unsigned long, double);
        double get_dn(double);

        void create(Stats<TF>&);
        void init();
        void exec(Stats<TF>&);
        void exec_viscosity(Thermo<TF>&);
        void diff_flux(Field3d<TF>&, const Field3d<TF>&);
        void exec_stats(Stats<TF>&);

        #ifdef USECUDA
        void prepare_device(Boundary<TF>&);
        void clear_device();
        #endif

	
	void select_box(
		const TF* restrict const field_var,
		float* restrict const box_var,
		const int k_center,
		const int j_center,
		const int i_center,
		const int boxsize,
		const int skip_firstx,
		const int skip_lastx,
		const int skip_firsty,
		const int skip_lasty,
		const int skip_firstz,
		const int skip_lastz
	);
	
	void diff_U(
		const TF* restrict const u,
		const TF* restrict const v,
		const TF* restrict const w,
		TF* restrict const ut,
		TF* restrict const vt,
		TF* restrict const wt
	);

	void hidden_layer1(
		const float* restrict const weights,
		const float* restrict const bias,
		const float* restrict const input,
		float* restrict const layer_out,
		const float alpha
	);
	
	void output_layer(
		const float* restrict const weights,
		const float* restrict const bias,
		const float* restrict const layer_in,
		float* restrict const layer_out
		);
	
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
		const bool zw_flag);
        
	void file_reader(
		float* const weights,
		const std::string& filename,
		const int N);

	// Size variables
	static constexpr int Nbatch = 1; //assume fixed batch size of 1
	static constexpr int boxsize = 5; //number of grid cells that the selected grid boxes extend in each spatial direction (i.e. 5 grid cells in case of a 5*5*5 grid box)
	static constexpr int N_inputvar = 3; //number of input variables
	static constexpr int N_input = 125; // =(5*5*5), size of 1 sample of 1 variable
	static constexpr int N_input_adjusted = 80; //=(4*5*4), adjusted size of 1 sample of 1 variable
	static constexpr int N_input_tot = 375; //=3*(5*5*5)
	static constexpr int N_input_tot_adjusted = 285; //=2*(4*5*4)+1*(5*5*5), adjusted size of 1 sample of all variables
	static constexpr int N_hidden = 64; // number of neurons in hidden layer
	static constexpr int N_output = 18; // number of output transport components
	static constexpr int N_output_zw = 2; // number of output transport components in case only zw is evaluated
	static constexpr int N_output_control = 6; // number of output transport components per control volume

	// Network variables
	float m_utau_ref;
	std::vector<float> m_hiddenu_wgth;
	std::vector<float> m_hiddenv_wgth;
	std::vector<float> m_hiddenw_wgth;
	std::vector<float> m_outputu_wgth;
	std::vector<float> m_outputv_wgth;
	std::vector<float> m_outputw_wgth;
	std::vector<float> m_hiddenu_bias;
	std::vector<float> m_hiddenv_bias;
	std::vector<float> m_hiddenw_bias;
	std::vector<float> m_outputu_bias;
	std::vector<float> m_outputv_bias;
	std::vector<float> m_outputw_bias;
	float m_hiddenu_alpha;
	float m_hiddenv_alpha;
	float m_hiddenw_alpha;
	std::vector<float> m_input_ctrlu_u;
	std::vector<float> m_input_ctrlu_v;
	std::vector<float> m_input_ctrlu_w;
	std::vector<float> m_input_ctrlv_u;
	std::vector<float> m_input_ctrlv_v;
	std::vector<float> m_input_ctrlv_w;
	std::vector<float> m_input_ctrlw_u;
	std::vector<float> m_input_ctrlw_v;
	std::vector<float> m_input_ctrlw_w;
	std::vector<float> m_output;
	std::vector<float> m_output_zw;
	float m_output_denorm_utau2;
	std::vector<float> m_mean_input;
	std::vector<float> m_stdev_input;
	std::vector<float> m_mean_label;
	std::vector<float> m_stdev_label;

    private:
        using Diff<TF>::master;
        using Diff<TF>::grid;
        using Diff<TF>::fields;
        using Diff<TF>::boundary;
        Boundary_cyclic<TF> boundary_cyclic;
        Field3d_operators<TF> field3d_operators;

        using Diff<TF>::tPr;

        const Diffusion_type swdiff = Diffusion_type::Diff_smag2;

        void create_stats(Stats<TF>&);

        TF* mlen_g;

        double dnmax;
        double dnmul;

        double cs;

        const std::string tend_name = "diff";
        const std::string tend_longname = "Diffusion";
};
#endif
