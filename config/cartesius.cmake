# Cartesius
# Before using this config file, execute the following commands first:
#####
# Intel Compiler:
# module purge
# module load 2019
# module load CMake
# module load intel/2018b
# module load netCDF/4.6.1-intel-2018b
# module load FFTW/3.3.8-intel-2018b
#####
#NOTE: not tested with GNU compiler, in that case the script below has to be changed to include the MKL library correctly.

# Use "lfs setstripe -c 50" in empty directories if large files need to be
# written with MPI-IO. It hugely increases the IO performance.

if(USEMPI)
    set(ENV{CC}  mpiicc ) # C compiler for parallel build
    set(ENV{CXX} mpiicpc) # C++ compiler for parallel build
    set(USEICC TRUE)
else()
    set(ENV{CC}  icc ) # C compiler for parallel build
    set(ENV{CXX} icpc) # C++ compiler for serial build
    set(USEICC TRUE)
endif()

if(USECUDA)
    set(ENV{CC}  gcc) # C compiler for serial build
    set(ENV{CXX} g++) # C++ compiler for serial build
    set(USEICC FALSE)
endif()

if(USEICC)
    set(USER_CXX_FLAGS "-std=c++14 -restrict")
    #set(USER_CXX_FLAGS "-std=c++14 -restrict -I$ENV{MKLROOT}/include")
    set(USER_CXX_FLAGS_RELEASE "-Ofast -xAVX -axCORE-AVX-I,CORE-AVX2,CORE-AVX512")
    set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")
    add_definitions(-DRESTRICTKEYWORD=restrict)
    set(USER_EXE_LINKER_FLAGS "-lfftw3 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core") #NOTE: fftw3 libraries are on purpose explicitly linked first. This is to prioritize the fftw3 libraries over the intel MKL fftw libraries: the intel MKL fftw libraries do not support all fftw3 functions used in the MicroHH code (e.g. the function fftw_export_wisdom_to_filename).
else()
    set(USER_CXX_FLAGS "-std=c++14 -fopenmp")
    set(USER_CXX_FLAGS_RELEASE "-Ofast -march=ivybridge") # -march optimized for the CPU present in Cartesius GPU nodes
    add_definitions(-DRESTRICTKEYWORD=__restrict__)
endif()

set(FFTW_LIB        "fftw3")
set(FFTWF_LIB       "fftw3f")
#set(MKL_INCLUDE_DIR "$ENV{MKLROOT}/include")
set(MKL_LIB1        "$ENV{MKLROOT}/lib/intel64/libmkl_intel_lp64.so")
set(MKL_LIB2        "$ENV{MKLROOT}/lib/intel64/libmkl_sequential.so")
set(MKL_LIB3        "$ENV{MKLROOT}/lib/intel64/libmkl_core.so")
set(NETCDF_LIB_C    "netcdf")
set(IRC_LIB         "irc")
set(IRC_LIB         "")
set(HDF5_LIB        "hdf5")
set(SZIP_LIB        "sz")

#set(LIBS ${FFTW_LIB} ${FFTWF_LIB} ${NETCDF_LIB_C} ${HDF5_LIB} ${SZIP_LIB} ${IRC_LIB} m z curl)
#set(LIBS ${NETCDF_LIB_C} ${HDF5_LIB} ${SZIP_LIB} ${IRC_LIB} m z curl)
set(LIBS ${FFTW_LIB} ${FFTWF_LIB} ${MKL_LIB1} ${MKL_LIB2} ${MKL_LIB3} ${NETCDF_LIB_C} ${HDF5_LIB} ${SZIP_LIB} ${IRC_LIB} m z curl)

#set(INCLUDE_DIRS ${MKL_INCLUDE_DIR})

if(USECUDA)
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    set(LIBS ${LIBS} -rdynamic $ENV{EBROOTCUDA}/lib64/libcufft.so)
    set(USER_CUDA_NVCC_FLAGS "-arch=sm_35")
    list(APPEND CUDA_NVCC_FLAGS "-std=c++14")
    list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")
endif()
