// main.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "Network.h"
#include "Grid.h"
#include "diff_U.h"
#include <iostream>
#include <string>
#include <chrono>
#include <vector>

int main()
{
	// For now, hard-code command-line inputs user.
	
	// Directory where the parameters of the MLP are stored.
	// NOTE: Set this to explicitly to the path where the variables of the MLP are stored!
	
	//Windows
	//std::string var_filepath = "M:\\My Documents\\Machine learning projects\\SURFsara project\\Scripts\\Neural network\\Variables_MLP13\\";
	//Linux
	//std::string var_filepath = "/home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP13/";
	std::string var_filepath = "Variables_MLP13/";


	//Instantiate grid object, which in this test script just contains hard-coded variables
	Grid gd;

	//For simplicitly, hard-code height differences
	std::vector<float> dzi(gd.m_kcells, 0);
	std::vector<float> dzhi(gd.m_kcells, 0);
	for (int k = 0; k < gd.m_kcells; ++k)
	{
		dzi[k] = 32.f;
		dzhi[k] = 32.f;
	}

	// initialize dynamically allocated arrays for velocity fields and tendencies.
	std::vector<float> u(gd.m_kcells*gd.m_jcells*gd.m_icells, 0.0f);
	//NOTE: on purpose one smaller in xh-direction than stored in nc-file, compensate for this when reading flow fields from nc-file.
	std::vector<float> v(gd.m_kcells*gd.m_jcells*gd.m_icells, 0.0f);
	//NOTE: on purpose one smaller in yh-direction than stored in nc-file
	std::vector<float> w(gd.m_khcells*gd.m_jcells*gd.m_icells, 0.0f);
	std::vector<float> ut(gd.m_kcells*gd.m_jcells*gd.m_icells, 0.0f);
	std::vector<float> vt(gd.m_kcells*gd.m_jcells*gd.m_icells, 0.0f);
	std::vector<float> wt(gd.m_khcells*gd.m_jcells*gd.m_icells, 0.0f);

	
	//Instantiate Network class to make predictions
	Network MLP(var_filepath);

	// Define time steps for inference
	int tstart = 0;
	int tend = 50;

	//Start time loop, loop over all flow fields
	int counter = 0;
	for (int t = tstart; t < tend; ++t)
		//for (int t = 0; t < 1; ++t) // FOR TESTING PURPOSES ONLY!
	{
		// Create random velocity fields at each time step
		std::srand(1); // use each time a random seed of 1
		for (int i = 0; i < (gd.m_kcells*gd.m_jcells*gd.m_icells); ++i)
		{
			u[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX); // Generate random number between 0 and 1
			v[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX); // Generate random number between 0 and 1
		}
		for (int i = 0; i < (gd.m_khcells*gd.m_jcells*gd.m_icells); ++i)
		{
			w[i] = static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX); // Generate random number between 0 and 1
		}

		//Time code
		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		//Call diff_U
		diff_U(u.data(), v.data(), w.data(), dzi.data(), dzhi.data(), ut.data(), vt.data(), wt.data(), gd, MLP);
		//End timing & print time difference
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		std::cout << "Time difference = " << (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) / 1000000. << "[seconds]" << std::endl;

		//Add 1 to counter
		counter += 1;

		//Set tendencies back to 0 for next iteration
		for (int i = 0; i < gd.m_kcells*gd.m_jcells*gd.m_icells; ++i)
		{
			ut[i] = 0;
			vt[i] = 0;
		}
		for (int i = 0; i < gd.m_khcells*gd.m_jcells*gd.m_icells; ++i)
		{
			wt[i] = 0;
		}
	}
	return 0;
}