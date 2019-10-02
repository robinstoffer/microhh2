# Ubuntu 16.04
if(USEMPI) 
  set(ENV{CC}  mpicc ) # C compiler for parallel build
  set(ENV{CXX} mpicxx) # C++ compiler for parallel build
else()
  set(ENV{CC}  gcc) # C compiler for serial build
  set(ENV{CXX} g++) # C++ compiler for serial build
endif()
set(ENV{MKL_ROOT} "/home/robin/intel/compilers_and_libraries_2019.5.281/linux/mkl")

set(USER_CXX_FLAGS "-std=c++14")
set(USER_CXX_FLAGS_RELEASE "-Ofast -DNDEBUG -mtune=native -march=native")
set(USER_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wno-unknown-pragmas")

set(FFTW_INCLUDE_DIR   "/usr/include")
set(FFTW_LIB           "/usr/lib/x86_64-linux-gnu/libfftw3.so")
set(FFTWF_LIB          "/usr/lib/x86_64-linux-gnu/libfftw3f.so")
set(MKL_INCLUDE_DIR    "/home/robin/intel/compilers_and_libraries_2019.5.281/linux/mkl/include")
set(MKL_LIB1           "/home/robin/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/libmkl_intel_lp64.so")
set(MKL_LIB2           "/home/robin/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/libmkl_sequential.so")
set(MKL_LIB3           "/home/robin/intel/compilers_and_libraries_2019.5.281/linux/mkl/lib/intel64/libmkl_core.so")
set(NETCDF_INCLUDE_DIR "/usr/include")
set(NETCDF_LIB_C       "/usr/lib/x86_64-linux-gnu/libnetcdf.so")
set(NETCDF_LIB_CPP     "/usr/lib/x86_64-linux-gnu/libnetcdf_c++4.so")
set(HDF5_LIB_1         "/usr/lib/x86_64-linux-gnu/libhdf5_serial.so")
set(HDF5_LIB_2         "/usr/lib/x86_64-linux-gnu/libhdf5_serial_hl.so")
set(SZIP_LIB           "")
set(LIBS ${FFTW_LIB} ${FFTWF_LIB} ${MKL_LIB1} ${MKL_LIB2} ${MKL_LIB3} ${NETCDF_LIB_CPP} ${NETCDF_LIB_C} ${HDF5_LIB_2} ${HDF5_LIB_1} ${SZIP_LIB} m z curl)
set(INCLUDE_DIRS ${FFTW_INCLUDE_DIR} ${MKL_INCLUDE_DIR} ${NETCDF_INCLUDE_DIR})

add_definitions(-DRESTRICTKEYWORD=__restrict__)
