//#include <math.h>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <iostream>
extern "C"
{
	//#include <cblas.h>
	#include <mkl.h>
}
#include "Network.h"
#define restrict __restrict

//using namespace Blaze;

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

void Network::file_reader(
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

Network::Network(std::string var_filepath)
{
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
	std::vector<float> hiddenu_wgth_notr(Network::N_hidden*Network::N_input_tot_adjusted); // Store tranposed variants of weights in class, see below. These matrices are temporary.
	std::vector<float> hiddenv_wgth_notr(Network::N_hidden*Network::N_input_tot_adjusted);
	std::vector<float> hiddenw_wgth_notr(Network::N_hidden*Network::N_input_tot_adjusted);
	std::vector<float> outputu_wgth_notr(Network::N_output_control*Network::N_hidden);
	std::vector<float> outputv_wgth_notr(Network::N_output_control*Network::N_hidden);
	std::vector<float> outputw_wgth_notr(Network::N_output_control*Network::N_hidden);
	m_hiddenu_wgth.resize(Network::N_hidden*Network::N_input_tot_adjusted);
	m_hiddenv_wgth.resize(Network::N_hidden*Network::N_input_tot_adjusted);
	m_hiddenw_wgth.resize(Network::N_hidden*Network::N_input_tot_adjusted);
	m_outputu_wgth.resize(Network::N_output_control*Network::N_hidden);
	m_outputv_wgth.resize(Network::N_output_control*Network::N_hidden);
	m_outputw_wgth.resize(Network::N_output_control*Network::N_hidden);
	m_hiddenu_bias.resize(Network::N_hidden);
	m_hiddenv_bias.resize(Network::N_hidden);
	m_hiddenw_bias.resize(Network::N_hidden);
	m_outputu_bias.resize(Network::N_output_control);
	m_outputv_bias.resize(Network::N_output_control);
	m_outputw_bias.resize(Network::N_output_control);
	m_mean_input.resize(Network::N_inputvar);
	m_stdev_input.resize(Network::N_inputvar);
	m_mean_label.resize(Network::N_output);
	m_stdev_label.resize(Network::N_output);
	m_hiddenu_alpha = 0.f;
	m_hiddenv_alpha = 0.f;
	m_hiddenw_alpha = 0.f;
	m_utau_ref = 0.f;
	m_output_denorm_utau2 = 0.f;

	file_reader(hiddenu_wgth_notr.data(),hiddenu_wgth_str,Network::N_hidden*Network::N_input_tot_adjusted);
	file_reader(hiddenv_wgth_notr.data(),hiddenv_wgth_str,Network::N_hidden*Network::N_input_tot_adjusted);
	file_reader(hiddenw_wgth_notr.data(),hiddenw_wgth_str,Network::N_hidden*Network::N_input_tot_adjusted);
	file_reader(outputu_wgth_notr.data(),outputu_wgth_str,Network::N_output*Network::N_hidden);
	file_reader(outputv_wgth_notr.data(),outputv_wgth_str,Network::N_output*Network::N_hidden);
	file_reader(outputw_wgth_notr.data(),outputw_wgth_str,Network::N_output*Network::N_hidden);
	file_reader(m_hiddenu_bias.data(), hiddenu_bias_str, Network::N_hidden);
	file_reader(m_hiddenv_bias.data(), hiddenv_bias_str, Network::N_hidden);
	file_reader(m_hiddenw_bias.data(), hiddenw_bias_str, Network::N_hidden);
	file_reader(m_outputu_bias.data(), outputu_bias_str, Network::N_output);
	file_reader(m_outputv_bias.data(), outputv_bias_str, Network::N_output);
	file_reader(m_outputw_bias.data(), outputw_bias_str, Network::N_output);
	file_reader(m_mean_input.data(),mean_input_str,Network::N_inputvar);
	file_reader(m_stdev_input.data(),stdev_input_str,Network::N_inputvar);
	file_reader(m_mean_label.data(),mean_label_str,Network::N_output);
	file_reader(m_stdev_label.data(),stdev_label_str,Network::N_output);
	file_reader(&m_hiddenu_alpha,hiddenu_alpha_str,1);
	file_reader(&m_hiddenv_alpha,hiddenv_alpha_str,1);
	file_reader(&m_hiddenw_alpha,hiddenw_alpha_str,1);
	file_reader(&m_utau_ref, utau_ref_str, 1);
	file_reader(&m_output_denorm_utau2,output_denorm_utau2_str,1);

	// Take transpose of weights and store those in the class
	for (int hiddenidx = 0; hiddenidx < Network::N_hidden; ++hiddenidx)
	{
		for (int inpidx = 0; inpidx < Network::N_input_tot_adjusted; ++inpidx)
		{
			int idx_tr = inpidx + hiddenidx * Network::N_input_tot_adjusted;
			int idx_notr = inpidx * Network::N_hidden + hiddenidx;
			m_hiddenu_wgth[idx_tr] = hiddenu_wgth_notr[idx_notr];
			m_hiddenv_wgth[idx_tr] = hiddenv_wgth_notr[idx_notr];
			m_hiddenw_wgth[idx_tr] = hiddenw_wgth_notr[idx_notr];
		}
	}
	for (int outputidx = 0; outputidx < Network::N_output_control; ++outputidx)
	{
		for (int hiddenidx = 0; hiddenidx < Network::N_hidden; ++hiddenidx)
		{
			int idx_tr = hiddenidx + outputidx * Network::N_hidden;
			int idx_notr = hiddenidx * Network::N_output_control + outputidx;
			m_outputu_wgth[idx_tr] = outputu_wgth_notr[idx_notr];
			m_outputv_wgth[idx_tr] = outputv_wgth_notr[idx_notr];
			m_outputw_wgth[idx_tr] = outputw_wgth_notr[idx_notr];
		}
	}

	// Define dynamic arrays hidden/ouptut layers and initialize them with zeros
	m_input_ctrlu_u.resize(Network::N_input,0.0f);
	m_input_ctrlu_v.resize(Network::N_input_adjusted,0.0f);
	m_input_ctrlu_w.resize(Network::N_input_adjusted,0.0f);
	m_input_ctrlv_u.resize(Network::N_input_adjusted, 0.0f);
	m_input_ctrlv_v.resize(Network::N_input, 0.0f);
	m_input_ctrlv_w.resize(Network::N_input_adjusted, 0.0f);
	m_input_ctrlw_u.resize(Network::N_input_adjusted, 0.0f);
	m_input_ctrlw_v.resize(Network::N_input_adjusted, 0.0f);
	m_input_ctrlw_w.resize(Network::N_input, 0.0f);
	m_output.resize(Network::N_output,0.0f);
	m_output_zw.resize(Network::N_output_zw,0.0f);
}

Network::~Network()
{
}