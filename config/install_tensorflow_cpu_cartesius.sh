#!/bin/bash
# Bash script for Cartesius to create a virtual environment with Tensorflow (v1.12.0) and other required Python libraries
# Loading modules
module purge
module load 2019
module load Python/3.6.6-intel-2018b
module load intel/2018b

echo "Loaded modules are:"
module list

VIRTENV=firstMLP_intel_CPU
VIRTENV_ROOT=~/virtualenv

# Creating virtual env
echo "Creating virtual environment $VIRTENV_ROOT/$VIRTENV"
virtualenv $VIRTENV_ROOT/$VIRTENV

# Sourcing virtual env
echo "Sourching virtual environment $VIRTENV_ROOT/$VIRTENV"
source $VIRTENV_ROOT/$VIRTENV/bin/activate

# Check current python packages
echo "Current Python packages"
pip3 list

# Export LD_LIBRARY_PATH
export MPICC=mpicc
export MPICXX=mpicxx

# Install standard libraries
pip3 install numpy==1.15.4
pip3 install scipy==1.2.0
pip3 install netCDF4==1.4.2
pip3 install pandas==1.0.3
pip3 install matplotlib==3.0.2
pip3 install scikit-learn==0.23.1

# Tensorflow
echo "Installing Tensorflow"
pip3 install tensorflow==1.12.0 --no-cache-dir
pip3 install tensorboard==1.12.0 --no-cache-dir

## Getting this error when installing horovod, unless we comment a line: https://github.com/protocolbuffers/protobuf/issues/4069
#sed -i 's/static_assert(std::is_pod<AuxillaryParseTableField>::value, "");/\/\/static_assert(std::is_pod<AuxillaryParseTableField>::value, "");/g' $VIRTENV_ROOT/$VIRTENV/lib/python3.6/site-packages/tensorflow/include/google/protobuf/generated_message_table_driven.h
#
## Horovod
#echo "Installing Horovod"
#pip install horovod --no-cache-dir
#
## netCDF
#echo "Installing netCDF for python"
#pip3 install netCDF4 --no-cache-dir
#
## Scipy
#echo "Installing scipy for python"
#pip3 install scipy --no-cache-dir
#
## Matplotlib
## Note: Tkinter is not supported with this installation
## Switch backend to Agg after importing matplotlib in your python session, then plot
#echo "Installing matplotlib"
#pip3 install matplotlib --no-cache-dir
