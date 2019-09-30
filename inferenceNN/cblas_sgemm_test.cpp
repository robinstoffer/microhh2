// cblas_sgemm.cpp : 
// This script can be used to test the effectiveness in the hidden layer of cblas_sgemm ...
// for a chosen batch size vs cblas_sgemv for each sample separately, given a single flow field.

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
extern "C"
{
#include <mkl.h>
}

int main()
{
	std::cout << "Start executing \n";
	// Initialize needed variables
	constexpr int batch_size_field = 152064; //((48*96*64 + 2*96*48)/2)
	//constexpr int batch_size_field = 294912;
	constexpr int N_input_tot_adjusted = 285;
	//constexpr int N_input_tot_adjusted = 375;
	//constexpr int N_hidden = 1024;
	constexpr int N_hidden = 64;
	constexpr int N_output_control = 6;
	std::vector<float> inputs(batch_size_field * N_input_tot_adjusted, 1.0f);
	std::vector<float> weights(N_hidden * N_input_tot_adjusted, 1.0f);
	std::vector<float> outputs(batch_size_field * N_hidden, 0.0f);
	
	/*// Fill inputs and weights with random values between 0 and 1
	std::srand(1); // use each time a random seed of 1
	for (int inpidx = 0; inpidx < (batch_size_field * N_input_tot_adjusted); ++inpidx)
	{
		inputs[inpidx] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX); // Generate random number between 0 and 1
	}
	for (int weightsidx = 0; weightsidx < (N_hidden * N_input_tot_adjusted); ++weightsidx)
	{
		weights[weightsidx] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX); // Generate random number between 0 and 1
	}*/

	// Do one big matrix-matrix multiplication
	//Time code
	std::chrono::steady_clock::time_point begin1 = std::chrono::steady_clock::now();

	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, batch_size_field, N_hidden, N_input_tot_adjusted, 
		1., inputs.data(), N_input_tot_adjusted, weights.data(), N_hidden, 0, outputs.data(), N_hidden);

	//End timing & print time difference
	std::chrono::steady_clock::time_point end1 = std::chrono::steady_clock::now();
	std::cout << "Time difference large MM multiplication = " << (std::chrono::duration_cast<std::chrono::microseconds>(end1 - begin1).count()) / 1000000. << "[seconds]" << std::endl;

	// Do only matrix-vector multiplications, loop over batch
	// Initialize samples in a single batch once, 
	// such that the performance difference with the code above is only due to the usage of gemm instead of gemv.
	std::vector<float> inputs_singlesample(N_input_tot_adjusted, 1.0f); 
	std::vector<float> weights_singlesample(N_hidden*N_input_tot_adjusted, 1.0f);
	std::vector<float> outputs_singlesample(N_hidden, 1.0f);
	//Time code
	std::chrono::steady_clock::time_point begin2 = std::chrono::steady_clock::now();

	for (int batchidx = 0; batchidx < batch_size_field; ++batchidx)
	{
		cblas_sgemv(CblasRowMajor, CblasNoTrans, N_hidden, N_input_tot_adjusted,
			1., weights_singlesample.data(), N_input_tot_adjusted, inputs_singlesample.data(), 1, 0, outputs_singlesample.data(), 1);
	}

	//End timing & print time difference
	std::chrono::steady_clock::time_point end2 = std::chrono::steady_clock::now();
	std::cout << "Time difference small MV multiplications = " << (std::chrono::duration_cast<std::chrono::microseconds>(end2 - begin2).count()) / 1000000. << "[seconds]" << std::endl;

	// Do matrix-matrix multiplications for smaller (sub-)batches, loop over total batch
	constexpr int small_batch = 48; //Single row in LES-simulation, even larger computational gains at smaller batches (e.g. 12) or at a lower memory cost (e.g. 8).
	//constexpr int small_batch = 12;
	// Initialize samples in a single (sub-)batch once, 
	// such that the performance difference with the code above is only due to the usage of gemm instead of gemv.
	std::vector<float> inputs_singlebatch(small_batch*N_input_tot_adjusted, 1.0f);
	std::vector<float> weights_singlebatch(N_hidden*N_input_tot_adjusted, 1.0f);
	std::vector<float> outputs_singlebatch(small_batch*N_hidden, 1.0f);
	//Time code
	std::chrono::steady_clock::time_point begin3 = std::chrono::steady_clock::now();

	for (int batchidx = 0; batchidx * small_batch < batch_size_field; ++batchidx)
	{
		cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, small_batch, N_hidden, N_input_tot_adjusted,
			1., inputs_singlebatch.data(), N_input_tot_adjusted, weights_singlebatch.data(), 
			N_hidden, 0, outputs_singlebatch.data(), N_hidden);
	}

	//End timing & print time difference
	std::chrono::steady_clock::time_point end3 = std::chrono::steady_clock::now();
	std::cout << "Time difference sub-batch MM multiplications = " << (std::chrono::duration_cast<std::chrono::microseconds>(end3 - begin3).count()) / 1000000. << "[seconds]" << std::endl;
	
}