#Extra commands needed to compile scripts in Ubuntu:
#Compile program
export MKL_ROOT=/home/robin/intel/compilers_and_libraries_2019.5.281/linux/mk
g++ -Wall -o MLP diff_U.h diff_U.cpp Grid.h Grid_test.cpp main_test.cpp Network.h Network.cpp -I${MKL_ROOT}/include -L${MKL_ROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -std=c++14 -Ofast -march=native
#Run program
export LD_LIBRARY_PATH==${MKL_ROOT}/lib/intel64
./MLP
