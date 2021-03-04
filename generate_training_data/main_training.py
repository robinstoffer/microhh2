#Main script to generate training data, makes use of func_generate_training.py, sample_training_data_tfrecord.py, grid_objects_training.py, downsampling_training.py.
#NOTE: requires simulation output (i.e. binary velocity fields) of the selected DNS channel flow test case! In this script, it is assumed that this output is stored in the moser600 case directory.

import multiprocessing as mp

#Load scripts with functions needed for generating training data and samples
from func_generate_training import generate_training_data
from sample_training_data_tfrecord import generate_samples

if __name__ == "__main__":

    input_directory = '/projects/1/flowsim/gmd_preprint_moser600_dns_run/'
    settings_filepath = '/projects/1/flowsim/gmd_preprint_moser600_dns_run/moser600.ini'
    grid_filepath = '/projects/1/flowsim/gmd_preprint_moser600_dns_run/grid.0000000'
    name_training_file = 'training_data.nc'
    name_boxfilter_file = 'dns_boxfilter.nc'
     
    #Generate training data and samples
    #8dx
    output_directory = '/projects/1/flowsim/gmd_training_data/8dx4dz/'
    training_filepath = output_directory + name_training_file
    sampling_filepath = output_directory + 'samples_training.nc'
    means_stdev_filepath = output_directory + 'means_stdevs_allfields.nc'
    
    generate_training_data((64,48,96), input_directory, output_directory, size_samples = 5, testing = False, periodic_bc = (False,True,True), zero_w_topbottom = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath, name_output_file = name_training_file)
    ##Generate samples stored in tfrecord-files
    #generate_samples(output_directory, training_filepath = training_filepath, samples_filepath = sampling_filepath, means_stdev_filepath = means_stdev_filepath, create_tfrecord = True, store_means_stdevs = True)
    
    #4dx
    output_directory = '/projects/1/flowsim/gmd_training_data/4dx4dz/'
    training_filepath = output_directory + name_training_file
    sampling_filepath = output_directory + 'samples_training.nc'
    means_stdev_filepath = output_directory + 'means_stdevs_allfields.nc'
    #
    generate_training_data((64,96,192), input_directory, output_directory, size_samples = 5, testing = False, periodic_bc = (False,True,True), zero_w_topbottom = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath, name_output_file = name_training_file)
    ##Generate samples stored in tfrecord-files
    #generate_samples(output_directory, training_filepath = training_filepath, samples_filepath = sampling_filepath, means_stdev_filepath = means_stdev_filepath, create_tfrecord = True, store_means_stdevs = True)
    
    #12dx
    output_directory = '/projects/1/flowsim/gmd_training_data/12dx4dz/'
    training_filepath = output_directory + name_training_file
    sampling_filepath = output_directory + 'samples_training.nc'
    means_stdev_filepath = output_directory + 'means_stdevs_allfields.nc'
    
    generate_training_data((64,32,64), input_directory, output_directory, size_samples = 5, testing = False, periodic_bc = (False,True,True), zero_w_topbottom = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath, name_output_file = name_training_file)
    #
    ##Generate samples stored in tfrecord-files
    #generate_samples(output_directory, training_filepath = training_filepath, samples_filepath = sampling_filepath, means_stdev_filepath = means_stdev_filepath, create_tfrecord = True, store_means_stdevs = True)
    
    ##test
    #output_directory = '/projects/1/flowsim/gmd_training_data/test/'
    #training_filepath = output_directory + name_training_file
    #sampling_filepath = output_directory + 'samples_training.nc'
    #means_stdev_filepath = output_directory + 'means_stdevs_allfields.nc'
    #generate_training_data((4,4,4), input_directory, output_directory, size_samples = 5, testing = True, periodic_bc = (False,True,True), zero_w_topbottom = True, settings_filepath = settings_filepath, grid_filepath = grid_filepath, name_output_file = name_training_file)
    #
    ##Generate samples stored in tfrecord-files
    #generate_samples(output_directory, training_filepath = training_filepath, samples_filepath = sampling_filepath, means_stdev_filepath = means_stdev_filepath, create_tfrecord = True, store_means_stdevs = True)

