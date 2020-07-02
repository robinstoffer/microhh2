#Main to test and run func_generate_training.py, sample_training_data_tfrecord.py, microhh_tools_robinst.py, grid_objects_training.py, downsampling_training.py.
#Author: Robin Stoffer (robin.stoffer@wur.nl)

#Load modules
import numpy as np
#import netCDF4 as nc
#import struct as st
#import glob
#import re
#import matplotlib as mpl
#mpl.use('Agg') #Prevent that Matplotlib uses Tk, which is not configured for the Python version I am using
#import matplotlib.pyplot as plt
#import scipy.interpolate

#Load scripts to be tested
#from grid_objects_training import Finegrid, Coarsegrid
#from grid_objects_training_hor import Finegrid, Coarsegrid
from func_generate_training import generate_training_data
#from func_generate_training_hor import generate_training_data_hor
from sample_training_data_tfrecord import generate_samples
from func_boxfilter import boxfilter_dns

##Do testing
#coordx1 = np.array([0.25,0.75])
#xsize1 = 1.0
#coordy1 = np.array([0.25,0.75])
#ysize1 = 1.0
#coordz1 = np.array([0.25,0.75,1.25])
#zsize1 = 1.5
#
#coordx2 = np.array([0.5,1.5,2.5]) #NOTE: Some small rounding errors in 15th decimal
#xsize2 = 3.0
#coordy2 = np.array([1.0,3.0,5.0])
#ysize2 = 6.0
#coordz2 = np.array([0.05,0.20,0.60])
#zsize2 = 1.0
#
#coordx3 = np.array([1.0,3.0,5.0,7.0,9.0]) #NOTE: Some small rounding errors in 13th decimal
#xsize3 = 10.0
#coordy3 = np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5])
#ysize3 = 9.0
#coordz3 = np.array([0.9,0.93,0.9999])
#zsize3 = 1.0
#
#coordx4 = np.array([0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75]) #NOTE: small rounding errors in 13th decimal
#xsize4 = 4.0
#coordy4 = np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5])
#ysize4 = 10.0
#coordz4 = np.array([0.01,0.05,0.1,13.7,13.8,13.85,13.9,13.95])
##coordz4 = np.array([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
#zsize4 = 14.0
##zsize4 = 16.0
#
#coordx5 = np.array([0.25,0.75,1.25,1.75,2.25,2.75,3.25,3.75,4.25,4.75,5.25,5.75,6.25,6.75,7.25,7.75,8.25,8.75,9.25,9.75]) #NOTE: small rounding errors in 12th decimal, in case in 9th decimal
#xsize5 = 10.0
#coordy5 = np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5,18.5,19.5])
#ysize5 = 20.0
#coordz5 = np.array([0.01,0.05,0.1,0.2,0.4,0.6,0.8,1.0,2.0,3.0,4.0,5.0,5.1,5.15,5.2,5.4,6.0,9.0,9.5,10.0,11.5,12.3,13.0,13.5,13.7,13.8,13.85,13.9,13.95])
#zsize5 = 14.0
#
#
#coordx = coordx3
#xsize  =  xsize3
#coordy = coordy3
#ysize  =  ysize3
#coordz = coordz3
#zsize  =  zsize3
##
#finegrid = Finegrid(read_grid_flag = False, fourth_order = False, coordx = coordx, xsize = xsize, coordy = coordy, ysize = ysize, coordz = coordz, zsize = zsize, periodic_bc = (False, True, True), zero_w_topbottom = True)
##finegrid = Finegrid()
##finegrid = Finegrid(read_grid_flag = False, fourth_order = False, coordx = np.array([0.25, 0.75]), xsize = 1.0, coordy = np.array([0.25,0.75]), ysize = 1.0, coordz = np.array([0.5,0.75]), zsize = 1.0, periodic_bc = (False, True, True), no_slip = True)
#
#output_shape = (coordz.shape[0], coordy.shape[0], coordx.shape[0])
#start_value = 0
#end_value = start_value + output_shape[0]*output_shape[1]*output_shape[2]
#output_array = np.reshape(np.arange(start_value,end_value), output_shape)
##output_array = np.ones(output_shape)
##output_0level = np.zeros((output_shape[1], output_shape[2]))
#output_1level = np.ones((output_shape[1], output_shape[2]))
##output_array = np.stack([output_1level, 2*output_1level, 3*output_1level], axis = 0)
##output_array = np.stack([output_1level, 2*output_1level, 3*output_1level, 4*output_1level, 5*output_1level, 6*output_1level, 2*output_1level, 4*output_1level, 5*output_1level, 4*output_1level,3*output_1level,2*output_1level,1*output_1level,6*output_1level,1*output_1level,6*output_1level,1*output_1level,6*output_1level,5*output_1level,4*output_1level,3*output_1level,2*output_1level,1*output_1level,5*output_1level,2*output_1level,4*output_1level,1*output_1level,3*output_1level,5*output_1level])
#
#finegrid.create_variables('u', output_array, bool_edge_gridcell = (False, False, True))
#finegrid.create_variables('v', output_array, bool_edge_gridcell = (False, True, False))
#finegrid.create_variables('w', output_array, bool_edge_gridcell = (True, False, False))
#finegrid.create_variables('p', output_array, bool_edge_gridcell = (False, False, False))
##
##coarsegrid = Coarsegrid((28,19,19), finegrid, igc = 2, jgc = 2, kgc = 2)
#coarsegrid = Coarsegrid((3,3), finegrid, igc = 2, jgc = 2, kgc = 2)
#coarsegrid.downsample('u')
#coarsegrid.downsample('v')
#coarsegrid.downsample('w')
#coarsegrid.downsample('p')
##
#volume_finegrid_u = finegrid.volume_integral('u')
#volume_coarsegrid_u = coarsegrid.volume_integral('u')
#print('Volume integral u finegrid: ' + str(volume_finegrid_u))
#print('Volume integral u coarsegrid: ' + str(volume_coarsegrid_u))
#
#volume_finegrid_v = finegrid.volume_integral('v')
#volume_coarsegrid_v = coarsegrid.volume_integral('v')
#print('Volume integral v finegrid: ' + str(volume_finegrid_v))
#print('Volume integral v coarsegrid: ' + str(volume_coarsegrid_v))
#
#volume_finegrid_w = finegrid.volume_integral('w')
#volume_coarsegrid_w = coarsegrid.volume_integral('w')
#print('Volume integral w finegrid: ' + str(volume_finegrid_w))
#print('Volume integral w coarsegrid: ' + str(volume_coarsegrid_w))
#print('NOTE: volume integral w is for coarsegrid not correct anymore because of implemented BC at the top/bottom (i.e. that w = 0). It does however add up when no downsampling is applied in the vertical direction.')
#
#volume_finegrid_p = finegrid.volume_integral('p')
#volume_coarsegrid_p = coarsegrid.volume_integral('p')
#print('Volume integral p finegrid: ' + str(volume_finegrid_p))
#print('Volume integral p coarsegrid: ' + str(volume_coarsegrid_p))

input_directory = '/projects/1/flowsim/simulation1/'
#output_directory = '/projects/1/flowsim/simulation1/coarsehor/'
#output_directory = '/projects/1/flowsim/simulation1/lesscoarse/'
#output_directory = '/projects/1/flowsim/simulation1/lesscoarse_horfields/'
output_directory = '/projects/1/flowsim/simulation1/lesscoarse_prefsampling/'
settings_filepath = '/projects/1/flowsim/simulation1/moser600.ini'
grid_filepath = '/projects/1/flowsim/simulation1/grid.0000000'
#input_directory = '/home/robinst/microhh/cases/moser600/simulation2_new/'
#output_directory = '/home/robinst/microhh/cases/moser600/git_repository/Training data'
#settings_filepath = '/home/robinst/microhh/cases/moser600/simulation2_new/moser600.ini'
#grid_filepath = '/home/robinst/microhh/cases/moser600/simulation2_new/grid.0000000'
name_training_file = 'training_data.nc'
name_boxfilter_file = 'dns_boxfilter.nc'
training_filepath = output_directory + name_training_file
sampling_filepath = output_directory + 'samples_training.nc'
means_stdev_filepath = output_directory + 'means_stdevs_allfields.nc'

#NOTE1:Original downsampling (to downsample from 100m to 4m in horizontal directions): (64,16,32)
#NOTE2: Resolution of high-resolution simulations is (256,384,768)
#generate_training_data((64,48,96), input_directory, output_directory, reynolds_number_tau = 590, size_samples = 5, testing = False, periodic_bc = (False,True,True), zero_w_topbottom = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath, name_output_file = name_training_file)
#generate_training_data_hor((48,96), input_directory, output_directory, reynolds_number_tau = 590, size_samples = 5, testing = False, periodic_bc = (False,True,True), zero_w_topbottom = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath, name_output_file = name_training_file)
generate_samples(output_directory, training_filepath = training_filepath, samples_filepath = sampling_filepath, means_stdev_filepath = means_stdev_filepath, create_binary = True, create_netcdf = False, store_means_stdevs = True)
#boxfilter_dns(input_directory, output_directory, reynolds_number_tau = 590, filter_widths = (0.03125,0.06544984695,0.06544984695), testing = False, periodic_bc = (False,True,True), zero_w_topbottom = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath, name_output_file = name_boxfilter_file)
#boxfilter_dns(input_directory, output_directory, reynolds_number_tau = 590, filter_widths = (1.0,1.0,1.0), testing = True, periodic_bc = (False,True,True), zero_w_topbottom = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath, name_output_file = name_boxfilter_file)
