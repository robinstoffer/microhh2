// Header file for network class
#include <math.h>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <iostream>
#ifndef NETWORK_H // Implement header guard
#define NETWORK_H
#define restrict __restrict

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

class Network
{
    public:
		
        Network(std::string var_filepath); // constructor
		
		~Network(); // destructor

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


};
#endif
