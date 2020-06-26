#!/bin/bash
##Extra commands needed to compile scripts in Ubuntu:
##Go to directory
cd /home/robin/microhh2/inferenceNN
##Set Intel MPI environment variables
##bash /home/robin/intel/impi/2019.5.281/intel64/bin/mpivars.sh


##Compile program
##test inference
#g++ -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid_test.cpp main_test.cpp Network.h Network.cpp -I${MKL_ROOT}/include -L${MKL_ROOT}/lib/x86_64-linux-gnu -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 -Ofast -march=native

##Compile program
##actual inference
export MKL_ROOT=/usr
g++ -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid.cpp main.cpp Network.h Network.cpp -I${MKL_ROOT}/include -L${MKL_ROOT}/lib/x86_64-linux-gnu -lnetcdf -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 -O0 -g -march=native -Ofast

##Run program
export MKL_ROOT=/usr
export LD_LIBRARY_PATH=${MKL_ROOT}/lib/x86_64-linux-gnu
##gdb MLP
./MLP
