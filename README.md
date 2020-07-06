MicroHH (v2.0 with ANN SGS model from Stoffer et al. (2020))
-------
[![Travis](https://api.travis-ci.org/microhh/microhh.svg?branch=master)](https://travis-ci.org/microhh/microhh) [![Documentation Status](https://readthedocs.org/projects/microhh/badge/?version=latest)](https://microhh.readthedocs.io/en/latest/?badge=latest)

MicroHH (v2.0) is a computational fluid dynamics code made for Direct Numerical Simulation (DNS) and Large-Eddy Simulation of turbulent flows in the atmospheric boundary layer. The code is written in C++.

The base code of MicroHH (v2.0) is hosted on GitHub (http://github.com/microhh2). Here, the latest version of the base code can be found, as well as all releases. A detailed description of the base code can be found in [Van Heerwaarden et al. (2017)](http://www.geosci-model-dev-discuss.net/gmd-2017-41/#discussion). In case you decide to use MicroHH for your own research, the developers of MicroHH would appreciate to be notified and kindly request to cite their reference paper. The version (1.0) described in the reference paper has been assigned a DOI via [Zenodo](https://zenodo.org).

[![DOI](https://zenodo.org/badge/14754940.svg)](https://zenodo.org/badge/latestdoi/14754940)

Additionally, this repository contains the ANN SGS model described in detail by Stoffer et al. (2020). It is fully integrated in the MicroHH (v2.0) code. All the scripts needed to train the ANN are included a well.

Requirements + software libraries used by Stoffer et al. (2020)
------------
In order to compile MicroHH and use all the scripts related to the ANN SGS model, you need several additional software libraries. We list them below, together with the versions used by Stoffer et al. (2020):
* C++ compiler (Used: Intel compiler version 18.0 update 3 for Linux).
* Intel MKL (Used: version 18.0 update 3 for Linux).
    Note: the Intel MKL library is only needed for the calls to the `cblas_sgemv` function in `diff_nn.cxx`. It is therefore possible to use other CBLAS libraries. If you want       this, do change the header file named in `diff_nn.cxx:46` to the one corresponding with the desired CBLAS library. Furthermore, the configuration files mentioned below (with     extension `.cmake`) have to be changed accordingly as well.
* FFTW3 libraries (Used: version 3.3.8).
* NetCDF4-C library (Used: version 4.6.1).
* CMake (Used: version 3.12.1).
* MPI2/3 implementation (Optional for MPI support; used: Intel MPI library version update 3.0 for Linux).
* CUDA (Optional for GPU support; not used).
* Python3 (Used: version 3.6.6) with the following libraries:
    * Tensorflow v1 (not version >=2.0; Used: version 1.12.0).
    * Numpy (Used: version 1.19.0).
    * Scipy (Used: version 1.2.0).
    * Matplotlib (Used: version 3.0.2).
    * NetCDF4 (Used: version 1.14.2).
    * Scikit-learn (Used: version 3.0.2).

Note: in the `config` directory, for Cartesius a bash script `install_tensorflow_cpu_cartesius.sh` is provided that creates a virtual environment with all the required Python libraries.

Compilation of the code
-----------------------
First, enter the config directory: 

    cd config

Here, you find two `.cmake` files with settings for the Dutch national supercomputer Cartesius (https://userinfo.surfsara.nl/systems/cartesius) and ubunty systems (tested for version 18.04.4 LTS). All the results shown in Stoffer et al. (2020) have been created on Cartesius. If you want to run the code on an ubunty system, check that the specified library locations are correct for your specific system and update if necessary. If you want to run the code on another system, create a new cmake file with the correct compiler settings and the proper location for all libraries.

Subsequently, copy the file corresponding with your system to default.cmake. Let us assume your system is Ubuntu:

    cp ubuntu.cmake default.cmake

Then, go back to the main directory and create a subdirectory with an arbitrary name in which you will compile the code. Let us assume this directory is called "build":

    cd ..  
    mkdir build  
    cd build   

From this directory, run cmake with the suffix .. to point to the parent directory where the CMakeLists.txt is found. This builds the model without Message Passing Interface (MPI) and CUDA support.

    cmake ..

In case you prefer to enable either MPI or CUDA, run INSTEAD of the previous command:
    
    cmake .. -DUSEMPI=TRUE

or

    cmake .. -DUSECUDA=TRUE

(Note that once the build has been configured and you wish to change the `USECUDA` or `USEMPI` setting, you must delete the build directory or create an additional empty directory from which `cmake` is run. For the moser600 DNS test case used in Stoffer et al. (2020), it is recommended to enable MPI to keep the total simulation time feasible.)

With the previous command you have triggered the build system and created the make files, if the `default.cmake` file contains the correct settings. Now, you can start the compilation of the code and create the microhh executable with:

    make -j

Your directory should contain a file named `microhh` now. This is the main executable.

Running the moser600 DNS test case used by Stoffer et al. (2020)
-----------------------
To start one of the included test cases, go back to the main directory and  open the directory `cases`. Here, the high-resolution DNS `moser600` case and the corresponding LES case with ANN SGS model `moser600_lesNNrestart` have been included. We start with the DNS `moser600` case, the high-resolution direct numerical simulation of turbulent channel flow used by Stoffer et al. (2020).

    cd cases/moser600

First, we have to create the vertical profiles for our prognostic variables:

    python moser600_input.py

Then, we have to copy or link the `microhh` executable to the current directory. Here we assume the executable is in the build directory that we have created before.

    cp ../../build/microhh .

Now, we can start `microhh` in initialization mode to create the initial fields:

    ./microhh init moser600

If everything works out properly, a series of files has been created. The model can be started now following:

    ./microhh run moser600

This will take, depending on the run settings in the file 'moser600.ini', quite some time. It is therefore recommended to run the case with MPI enabled on a number of nodes. For the Dutch national supercomputer Cartesius, the job script we used has been provided as an example ('job_moser600_MPI'). In case you want to use that one, please note that this job scripts requires the output from the `install_tensorflow_cpu_cartesius.sh` script.

After the simulation is finished, a statistics file called `moser600.default.0000000.nc` is created. You can open this file with a plotting tool like `ncview`, or plot the results against the reference data of Moser et al. (1999) with the three scripts provided (`moser600budget.py`, `moser600spectra.py`, `moser600stats.py`). Depending on where you have stored this reference data, you may have to change some of the paths specified in the scripts.

Generating the training data
-----------------------
After running the DNS case described above, you should have binary velocity fields stored in the `moser600` directory together with all the other corresponding settings/output files. Subsequently, you can generate the training data by running `main_training.py`, which produces with its current settings 1) a netCDF-file with the training data called `training_data.nc` and 2) binary TFRecord-files that contain ~10.000 samples per file (generated from `training_data.nc`). The TFRecord files will serve as direct input to the ANN training scripts. For Cartesius, the job script we used has been provided as an example (which again needs the output from the `install_tensorflow_cpu_cartesius.sh` script).
