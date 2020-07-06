#Script that contains MLP for turbulent channel flow case without distributed learning, making use of the tf.Estimator and tf.Dataset API.
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import numpy as np
import netCDF4 as nc
import tensorflow as tf
#import horovod.tensorflow as hvd
import random
import os
import subprocess
import glob
import argparse
import matplotlib
matplotlib.use('agg')
from tensorflow.python import debug as tf_debug

#Set logging info
tf.logging.set_verbosity(tf.logging.INFO)

##Enable eager execution
#tf.enable_eager_execution()

#Amount of cores on node
ncores = int(subprocess.check_output(["nproc", "--all"]))

# Instantiate the parser
parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--checkpoint_dir', type=str, default='/projects/1/flowsim/simulation1/CNN_checkpoints',
                    help='directory where checkpoints are stored')
parser.add_argument('--input_dir', type=str, default='/projects/1/flowsim/simulation1/',
                    help='directory where tfrecord files are located')
parser.add_argument('--stored_means_stdevs_filepath', type=str, default='/projects/1/flowsim/simulation1/means_stdevs_allfields.nc', \
        help='filepath for stored means and standard deviations of input variables, which should refer to a nc-file created as part of the training data')
parser.add_argument('--training_filepath', type=str, default='/projects/1/flowsim/simulation1/training_data.nc', \
        help='filepath for stored training file which should contain the friction velocity and be in netCDF-format.')
parser.add_argument('--gradients', default=None, \
        action='store_true', \
        help='Wind velocity gradients are used as input for the NN when this is true, otherwhise absolute wind velocities are used.')
parser.add_argument('--benchmark', dest='benchmark', default=None, \
        action='store_true', \
        help='Do fullrun when benchmark is false, which includes producing and storing of preditions. Furthermore, in a fullrun more variables are stored to facilitate reconstruction of the corresponding transport fields. When the benchmark flag is true, the scripts ends immidiately after calculating the validation loss to facilitate benchmark tests.')
parser.add_argument('--debug', default=None, \
        action='store_true', \
        help='Run script in debug mode to inspect tensor values while the Estimator is in training mode.')
parser.add_argument('--n_hidden', type=int, default=64, \
        help='Number of neurons in hidden layer')
parser.add_argument('--intra_op_parallelism_threads', type=int, default=ncores-1, \
        help='intra_op_parallelism_threads')
parser.add_argument('--inter_op_parallelism_threads', type=int, default=1, \
        help='inter_op_parallelism_threads')
parser.add_argument('--num_steps', type=int, default=10000, \
        help='Number of steps, i.e. number of batches times number of epochs')
#parser.add_argument('--batch_size', type=int, default=100, \
#        help='Number of samples selected in each batch')
parser.add_argument('--profile_steps', type=int, default=10000, \
        help='Every nth step, a profile measurement is performed that is stored in a JSON-file.')
parser.add_argument('--summary_steps', type=int, default=100, \
        help='Every nth step, a summary is written for Tensorboard visualization')
parser.add_argument('--checkpoint_steps', type=int, default=10000, \
        help='Every nth step, a checkpoint of the model is written')
args = parser.parse_args()

##Define function for making variables dimensionless and standardized.
##NOTE: uses global variable utau_ref
#def _standardization(variable, mean, standard_dev):
#    variable = variable / utau_ref
#    standardized_variable = (variable - mean) / standard_dev
#    return standardized_variable

#Define parse function for tfrecord files, which gives for each component in the example_proto 
#the output in format (dict(features),tensor(labels)) and normalizes according to specified means and variances.
def _parse_function(example_proto):

    if args.gradients is None:

        if args.benchmark is None:
            keys_to_features = {
                'uc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'vc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'wc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                #'pc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'unres_tau_xu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'x_sample_size':tf.FixedLenFeature([],tf.int64),
                'y_sample_size':tf.FixedLenFeature([],tf.int64),
                'z_sample_size':tf.FixedLenFeature([],tf.int64),
                'flag_topwall_sample':tf.FixedLenFeature([],tf.int64),
                'flag_bottomwall_sample':tf.FixedLenFeature([],tf.int64),
                'tstep_sample':tf.FixedLenFeature([],tf.int64),
                'xloc_sample':tf.FixedLenFeature([],tf.float32),
                'xhloc_sample':tf.FixedLenFeature([],tf.float32),
                'yloc_sample':tf.FixedLenFeature([],tf.float32),
                'yhloc_sample':tf.FixedLenFeature([],tf.float32),
                'zloc_sample':tf.FixedLenFeature([],tf.float32),
                'zhloc_sample':tf.FixedLenFeature([],tf.float32)
            }

        else:
            keys_to_features = {
                'uc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'vc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'wc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                #'pc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
                'unres_tau_xu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'flag_topwall_sample':tf.FixedLenFeature([],tf.int64),
                'flag_bottomwall_sample':tf.FixedLenFeature([],tf.int64)
            }

            
        parsed_features = tf.parse_single_example(example_proto, keys_to_features) #Uncomment when .batch() applied after .map()
        #parsed_features = tf.parse_example(example_proto, keys_to_features)
        #parsed_features['uc_sample'] = _standardization(parsed_features['uc_sample'], means['uc'], stdevs['uc'])
        #parsed_features['vc_sample'] = _standardization(parsed_features['vc_sample'], means['vc'], stdevs['vc'])
        #parsed_features['wc_sample'] = _standardization(parsed_features['wc_sample'], means['wc'], stdevs['wc'])
        #parsed_features['pc_sample'] = _standardization(parsed_features['pc_sample'], means['pc'], stdevs['pc'])

    else:

        if args.benchmark is None:
            keys_to_features = {
                'ugradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'ugrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'ugradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                #'pgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                #'pgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                #'pgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'unres_tau_xu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'x_sample_size':tf.FixedLenFeature([],tf.int64),
                'y_sample_size':tf.FixedLenFeature([],tf.int64),
                'z_sample_size':tf.FixedLenFeature([],tf.int64),
                'tstep_sample':tf.FixedLenFeature([],tf.int64),
                'flag_topwall_sample':tf.FixedLenFeature([],tf.int64),
                'flag_bottomwall_sample':tf.FixedLenFeature([],tf.int64),
                'xloc_sample':tf.FixedLenFeature([],tf.float32),
                'xhloc_sample':tf.FixedLenFeature([],tf.float32),
                'yloc_sample':tf.FixedLenFeature([],tf.float32),
                'yhloc_sample':tf.FixedLenFeature([],tf.float32),
                'zloc_sample':tf.FixedLenFeature([],tf.float32),
                'zhloc_sample':tf.FixedLenFeature([],tf.float32)
            }
        else:
            keys_to_features = {
                'ugradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'ugrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'ugradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'vgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'wgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                #'pgradx_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                #'pgrady_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                #'pgradz_sample':tf.FixedLenFeature([3*3*3],tf.float32),
                'unres_tau_xu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample_upstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zu_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zv_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_xw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_yw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'unres_tau_zw_sample_downstream' :tf.FixedLenFeature([],tf.float32),
                'flag_topwall_sample':tf.FixedLenFeature([],tf.int64),
                'flag_bottomwall_sample':tf.FixedLenFeature([],tf.int64)
            }

        parsed_features = tf.parse_single_example(example_proto, keys_to_features) #Uncomment when .batch() is applied after .map()
        #parsed_features = tf.parse_example(example_proto, keys_to_features)
    
    #Extract labels from the features dictionary, and stack them in a new labels array.
    labels = {}
    labels['unres_tau_xu_upstream'] =  parsed_features.pop('unres_tau_xu_sample_upstream')
    labels['unres_tau_yu_upstream'] =  parsed_features.pop('unres_tau_yu_sample_upstream')
    labels['unres_tau_zu_upstream'] =  parsed_features.pop('unres_tau_zu_sample_upstream')
    labels['unres_tau_xv_upstream'] =  parsed_features.pop('unres_tau_xv_sample_upstream')
    labels['unres_tau_yv_upstream'] =  parsed_features.pop('unres_tau_yv_sample_upstream')
    labels['unres_tau_zv_upstream'] =  parsed_features.pop('unres_tau_zv_sample_upstream')
    labels['unres_tau_xw_upstream'] =  parsed_features.pop('unres_tau_xw_sample_upstream')
    labels['unres_tau_yw_upstream'] =  parsed_features.pop('unres_tau_yw_sample_upstream')
    labels['unres_tau_zw_upstream'] =  parsed_features.pop('unres_tau_zw_sample_upstream')
    labels['unres_tau_xu_downstream'] =  parsed_features.pop('unres_tau_xu_sample_downstream')
    labels['unres_tau_yu_downstream'] =  parsed_features.pop('unres_tau_yu_sample_downstream')
    labels['unres_tau_zu_downstream'] =  parsed_features.pop('unres_tau_zu_sample_downstream')
    labels['unres_tau_xv_downstream'] =  parsed_features.pop('unres_tau_xv_sample_downstream')
    labels['unres_tau_yv_downstream'] =  parsed_features.pop('unres_tau_yv_sample_downstream')
    labels['unres_tau_zv_downstream'] =  parsed_features.pop('unres_tau_zv_sample_downstream')
    labels['unres_tau_xw_downstream'] =  parsed_features.pop('unres_tau_xw_sample_downstream')
    labels['unres_tau_yw_downstream'] =  parsed_features.pop('unres_tau_yw_sample_downstream')
    labels['unres_tau_zw_downstream'] =  parsed_features.pop('unres_tau_zw_sample_downstream')

    #Uncomment when parse_single_example is used
    labels = tf.stack([ 
        labels['unres_tau_xu_upstream'], labels['unres_tau_xu_downstream'], 
        labels['unres_tau_yu_upstream'], labels['unres_tau_yu_downstream'],
        labels['unres_tau_zu_upstream'], labels['unres_tau_zu_downstream'],
        labels['unres_tau_xv_upstream'], labels['unres_tau_xv_downstream'],
        labels['unres_tau_yv_upstream'], labels['unres_tau_yv_downstream'],
        labels['unres_tau_zv_upstream'], labels['unres_tau_zv_downstream'],
        labels['unres_tau_xw_upstream'], labels['unres_tau_xw_downstream'],
        labels['unres_tau_yw_upstream'], labels['unres_tau_yw_downstream'],
        labels['unres_tau_zw_upstream'], labels['unres_tau_zw_downstream']
        ], axis=0)

    ##Uncomment when parse_example is used
    #labels = tf.stack([ 
    #    labels['unres_tau_xu_upstream'], labels['unres_tau_xu_downstream'], 
    #    labels['unres_tau_yu_upstream'], labels['unres_tau_yu_downstream'],
    #    labels['unres_tau_zu_upstream'], labels['unres_tau_zu_downstream'],
    #    labels['unres_tau_xv_upstream'], labels['unres_tau_xv_downstream'],
    #    labels['unres_tau_yv_upstream'], labels['unres_tau_yv_downstream'],
    #    labels['unres_tau_zv_upstream'], labels['unres_tau_zv_downstream'],
    #    labels['unres_tau_xw_upstream'], labels['unres_tau_xw_downstream'],
    #    labels['unres_tau_yw_upstream'], labels['unres_tau_yw_downstream'],
    #    labels['unres_tau_zw_upstream'], labels['unres_tau_zw_downstream']
    #    ], axis=1)
    return parsed_features,labels


#Define training input function
def train_input_fn(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    #dataset = dataset.batch(batch_size) #Uncomment when each batch correponds to one tfrecord file
    #dataset = dataset.shuffle(len(filenames)) #comment this line when cache() is done after map()
    dataset = dataset.map(lambda line:_parse_function(line), num_parallel_calls=ncores) #Parallelize map transformation using the total amount of CPU cores available.
    dataset = dataset.cache() #NOTE: The unavoidable consequence of using cache() before shuffle is that during all epochs the order of the flow fields is approximately the same (which can be alleviated by choosing a large buffer size, but that costs quite some computational effort). However, using shuffle before cache() will strongly increase the computational effort since memory becomes saturated.    
    #dataset = dataset.shuffle(buffer_size=10000) #Defaults to reshuffling each time the dataset is iterated over, gives problems if each batch should correspond to one vertical level(shuffles also between individual tfrecords!)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()
    dataset.prefetch(1)

    return dataset

#Define evaluation function
def eval_input_fn(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    #dataset = dataset.batch(batch_size) #Uncomment when each batch correponds to one tfrecord file
    #dataset = dataset.shuffle(len(filenames)) #comment this line when cache() is done after map()
    dataset = dataset.map(lambda line:_parse_function(line), num_parallel_calls=ncores)
    dataset = dataset.cache()
    dataset = dataset.batch(batch_size, drop_remainder=False)
    #dataset = dataset.batch(batch_size)
    dataset.prefetch(1)

    return dataset    

#Define function for splitting the training and validation set
#NOTE: this split is on purpose not random. Always the flow fields corresponding to the last time steps are selected (noting that each tfrecord file contains all the samples of one flow field), such that the flow fields used for validation are as independent as possible from the fields used for training.
def split_train_val(time_steps, val_ratio):
    shuffled_steps = np.flip(time_steps)
    val_set_size = max(int(len(time_steps) * val_ratio),1) #max(..) makes sure that always at least 1 file is selected for validation
    val_steps = shuffled_steps[:val_set_size]
    train_steps = shuffled_steps[val_set_size:]
    return train_steps,val_steps


#Define function that builds a separate MLP.
def create_MLP(inputs, name_MLP, params):
    '''Function to build a MLP with specified inputs and labels (which are needed to build dedicated training ops). Inputs should be a list of tf.Tensors containing the individual variables.\\
            NOTE: this function accesses the global variable num_labels.'''

    with tf.name_scope('MLP_'+name_MLP):

        #Define input layer
        input_layer = tf.concat(inputs, axis=1, name = 'input_layer_'+name_MLP)

        #Define hidden and output layers
        dense1_layerdef  = tf.layers.Dense(units=params["n_dense1"], name="dense1_"+name_MLP, \
                activation=params["activation_function"], kernel_initializer=params["kernel_initializer"])
        dense1 = dense1_layerdef.apply(input_layer)
        output_layerdef = tf.layers.Dense(units=num_labels, name="output_layer_"+name_MLP, \
                activation=None, kernel_initializer=params["kernel_initializer"])
        output_layer = output_layerdef.apply(dense1)
        #output_layer_mask = tf.math.multiply(output_layer, mask)
        #Visualize activations hidden layer in TensorBoard
        tf.summary.histogram('activations_hidden_layer1'+name_MLP, dense1)
        tf.summary.scalar('fraction_of_zeros_in_activations_hidden_layer1'+name_MLP, tf.nn.zero_fraction(dense1))

        #Visualize layers in TensorBoard
        tf.summary.histogram('input_layer_'+name_MLP, input_layer)
        tf.summary.histogram('hidden_layer_'+name_MLP, dense1)
        tf.summary.scalar('fraction_of_zeros_in_activations_hidden_layer1'+name_MLP, tf.nn.zero_fraction(dense1))
        tf.summary.histogram('output_layer_'+name_MLP, output_layer)
    return output_layer

#Define model function for MLP estimator
def model_fn(features, labels, mode, params):
    '''Model function which calls create_MLP multiple times to build MLPs that each predict some of the labels. These separate MLPs are trained separately, but combined in validation and inference mode. \\
            NOTE: this function accesses the global variables args.gradients, means_dict_avgt, stdevs_dict_avgt, utau_ref, and iter_per_epoch.'''

    #Define tf.constants for storing the means and stdevs of the input variables & labels, which is needed for the normalisation and subsequent denormalisation in this graph
    #NOTE: the means and stdevs for the '_upstream' and '_downstream' labels are the same. This is why each mean and stdev is repeated twice.

    if args.gradients is None:         
        
        means_inputs = tf.constant([[
            means_dict_avgt['uc'],
            means_dict_avgt['vc'],
            means_dict_avgt['wc']]])
            #means_dict_avgt['pc']]])
        
        stdevs_inputs = tf.constant([[
            stdevs_dict_avgt['uc'],
            stdevs_dict_avgt['vc'],
            stdevs_dict_avgt['wc']]])
            #stdevs_dict_avgt['pc']]])
    
    else:

        means_inputs = tf.constant([[
            means_dict_avgt['ugradx'],
            means_dict_avgt['ugrady'],
            means_dict_avgt['ugradz'],
            means_dict_avgt['vgradx'],
            means_dict_avgt['vgrady'],
            means_dict_avgt['vgradz'],
            means_dict_avgt['wgradx'],
            means_dict_avgt['wgrady'],
            means_dict_avgt['wgradz']]])
            #means_dict_avgt['pgradx'],
            #means_dict_avgt['pgrady'],
            #means_dict_avgt['pgradz']]])
        
        stdevs_inputs = tf.constant([[
            stdevs_dict_avgt['ugradx'],
            stdevs_dict_avgt['ugrady'],
            stdevs_dict_avgt['ugradz'],
            stdevs_dict_avgt['vgradx'],
            stdevs_dict_avgt['vgrady'],
            stdevs_dict_avgt['vgradz'],
            stdevs_dict_avgt['wgradx'],
            stdevs_dict_avgt['wgrady'],
            stdevs_dict_avgt['wgradz']]])
            #stdevs_dict_avgt['pgradx'],
            #stdevs_dict_avgt['pgrady'],
            #stdevs_dict_avgt['pgradz']]])
        
    means_labels = tf.constant([[ 
        means_dict_avgt['unres_tau_xu_sample'],
        means_dict_avgt['unres_tau_xu_sample'],
        means_dict_avgt['unres_tau_yu_sample'],
        means_dict_avgt['unres_tau_yu_sample'],
        means_dict_avgt['unres_tau_zu_sample'],
        means_dict_avgt['unres_tau_zu_sample'],
        means_dict_avgt['unres_tau_xv_sample'],
        means_dict_avgt['unres_tau_xv_sample'],
        means_dict_avgt['unres_tau_yv_sample'],
        means_dict_avgt['unres_tau_yv_sample'],
        means_dict_avgt['unres_tau_zv_sample'],
        means_dict_avgt['unres_tau_zv_sample'],
        means_dict_avgt['unres_tau_xw_sample'],
        means_dict_avgt['unres_tau_xw_sample'],
        means_dict_avgt['unres_tau_yw_sample'],
        means_dict_avgt['unres_tau_yw_sample'],
        means_dict_avgt['unres_tau_zw_sample'],
        means_dict_avgt['unres_tau_zw_sample']]])
    
    stdevs_labels = tf.constant([[ 
        stdevs_dict_avgt['unres_tau_xu_sample'],
        stdevs_dict_avgt['unres_tau_xu_sample'],
        stdevs_dict_avgt['unres_tau_yu_sample'],
        stdevs_dict_avgt['unres_tau_yu_sample'],
        stdevs_dict_avgt['unres_tau_zu_sample'],
        stdevs_dict_avgt['unres_tau_zu_sample'],
        stdevs_dict_avgt['unres_tau_xv_sample'],
        stdevs_dict_avgt['unres_tau_xv_sample'],
        stdevs_dict_avgt['unres_tau_yv_sample'],
        stdevs_dict_avgt['unres_tau_yv_sample'],
        stdevs_dict_avgt['unres_tau_zv_sample'],
        stdevs_dict_avgt['unres_tau_zv_sample'],
        stdevs_dict_avgt['unres_tau_xw_sample'],
        stdevs_dict_avgt['unres_tau_xw_sample'],
        stdevs_dict_avgt['unres_tau_yw_sample'],
        stdevs_dict_avgt['unres_tau_yw_sample'],
        stdevs_dict_avgt['unres_tau_zw_sample'],
        stdevs_dict_avgt['unres_tau_zw_sample']]])

    #a1 = tf.print("means_labels: ", means_labels, output_stream=tf.logging.info, summarize=-1)
    #a2 = tf.print("stdev_labels: ", stdevs_labels, output_stream=tf.logging.info, summarize=-1)
    
    #Define identity ops for input variables, which can be used to set-up a frozen graph for inference.
    if args.gradients is None:
        input_u      = tf.identity(features['uc_sample'], name = 'input_u')
        #a1 = tf.print("input_u: ", input_u, output_stream=tf.logging.info, summarize=-1)
        #a5 = tf.print("input_u_shape: ", tf.shape(input_u), output_stream=tf.logging.info, summarize=-1)
        input_v      = tf.identity(features['vc_sample'], name = 'input_v')
        #a2 = tf.print("input_v: ", input_u, output_stream=tf.logging.info, summarize=-1)
        #a6 = tf.print("input_v_shape: ", tf.shape(input_v), output_stream=tf.logging.info, summarize=-1)
        input_w      = tf.identity(features['wc_sample'], name = 'input_w')
        #a3 = tf.print("input_w: ", input_u, output_stream=tf.logging.info, summarize=-1)
        #a7 = tf.print("input_w_shape: ", tf.shape(input_w), output_stream=tf.logging.info, summarize=-1)
        #input_p      = tf.identity(features['pc_sample'], name = 'input_p')
        #input_utau_ref = tf.identity(utau_ref, name = 'input_utau_ref') #Allow to feed utau_ref during inference, which likely helps to achieve Re independent results.
        input_utau_ref = tf.constant(utau_ref, name = 'utau_ref')
        #input_z        = tf.expand_dims(tf.identity(np.absolute(features['zloc_sample'] - 1.0), name = 'input_z'), axis=1) #Height should be the same for entire batch if batch corresponds to one vertical level
        #input_z        = tf.expand_dims(tf.identity(features['zloc_sample'], name = 'input_z'), axis=1) #Height should be the same for entire batch if batch corresponds to one vertical level
        #a8 = tf.print("input_z: ", input_z, output_stream=tf.logging.info, summarize=-1)
        #a9 = tf.print("input_z_shape: ", input_z.shape, output_stream=tf.logging.info, summarize=-1)

        #Vizualize inputs
        tf.summary.histogram('input_u', input_u)
        tf.summary.histogram('input_v', input_v)
        tf.summary.histogram('input_w', input_w)
        #tf.summary.histogram('input_z', input_z)


    else:   
        input_ugradx = tf.identity(features['ugradx_sample'], name = 'input_ugradx')
        input_ugrady = tf.identity(features['ugrady_sample'], name = 'input_ugrady')
        input_ugradz = tf.identity(features['ugradz_sample'], name = 'input_ugradz')
        input_vgradx = tf.identity(features['vgradx_sample'], name = 'input_vgradx')
        input_vgrady = tf.identity(features['vgrady_sample'], name = 'input_vgrady')
        input_vgradz = tf.identity(features['vgradz_sample'], name = 'input_vgradz')
        input_wgradx = tf.identity(features['wgradx_sample'], name = 'input_wgradx')
        input_wgrady = tf.identity(features['wgrady_sample'], name = 'input_wgrady')
        input_wgradz = tf.identity(features['wgradz_sample'], name = 'input_wgradz')
        #input_pgradx = tf.identity(features['pgradx_sample'], name = 'input_pgradx')
        #input_pgrady = tf.identity(features['pgrady_sample'], name = 'input_pgrady')
        #input_pgradz = tf.identity(features['pgradz_sample'], name = 'input_pgradz')
        #input_utau_ref = tf.identity(utau_ref, name = 'input_utau_ref') #Allow to feed utau_ref during inference, which likely helps to achieve Re independent results.
        input_utau_ref = tf.constant(utau_ref, name = 'utau_ref')
        #input_z        = tf.expand_dims(tf.identity(np.absolute(features['zloc_sample'][0] - 1.0), name = 'input_z'), axis=1) #Height should be the same for entire batch if batch corresponds to one vertical level
        #input_z        = tf.expand_dims(tf.identity(features['zloc_sample'], name = 'input_z'), axis=1) #Height should be the same for entire batch if batch corresponds to one vertical level

    #Define function to make input variables non-dimensionless and standardize them
    def _standardization(input_variable, mean_variable, stdev_variable, scaling_factor):
        #a3 = tf.print("input_variable", input_variable[0,:5], output_stream=tf.logging.info, summarize=-1)
        input_variable = tf.math.divide(input_variable, scaling_factor)
        #a4 = tf.print("input_variable", input_variable[0,:5], output_stream=tf.logging.info, summarize=-1)
        input_variable = tf.math.subtract(input_variable, mean_variable)
        #a5 = tf.print("mean_variable",  mean_variable, output_stream=tf.logging.info, summarize=-1)
        #a6 = tf.print("input_variable_mean", input_variable[0,:5], output_stream=tf.logging.info, summarize=-1)
        input_variable = tf.math.divide(input_variable, stdev_variable)
        #a7 = tf.print("stdev_variable", stdev_variable, output_stream=tf.logging.info, summarize=-1)
        #a8 = tf.print("input_variable_final", input_variable[0,:5], output_stream=tf.logging.info, summarize=-1)
        return input_variable#, a3, a4, a5, a6, a7, a8

    #Standardize input variables
    #NOTE: it is on purpose that P is NOT scaled with utau_ref!!!
    if args.gradients is None:
        
        with tf.name_scope("standardization_inputs"): #Group nodes in name scope for easier visualisation in TensorBoard
            input_u_stand  = _standardization(input_u, means_inputs[:,0], stdevs_inputs[:,0], input_utau_ref)
            input_v_stand  = _standardization(input_v, means_inputs[:,1], stdevs_inputs[:,1], input_utau_ref)
            input_w_stand  = _standardization(input_w, means_inputs[:,2], stdevs_inputs[:,2], input_utau_ref)
            #input_p_stand  = _standardization(input_p, means_inputs[:,3], stdevs_inputs[:,3], 1.)
            
            #Visualize non-dimensionless and standardized input values in TensorBoard
            tf.summary.histogram('input_u_stand', input_u_stand)
            tf.summary.histogram('input_v_stand', input_v_stand)
            tf.summary.histogram('input_w_stand', input_w_stand)
            #tf.summary.histogram('input_p_stand', input_p_stand)

    else:

        with tf.name_scope("standardization_inputs"): #Group nodes in name scope for easier visualisation in TensorBoard
            input_ugradx_stand = _standardization(input_ugradx, means_inputs[:,0],  stdevs_inputs[:,0],  input_utau_ref)
            input_ugrady_stand = _standardization(input_ugrady, means_inputs[:,1],  stdevs_inputs[:,1],   input_utau_ref)
            input_ugradz_stand = _standardization(input_ugradz, means_inputs[:,2],  stdevs_inputs[:,2],   input_utau_ref)
            input_vgradx_stand = _standardization(input_vgradx, means_inputs[:,3],  stdevs_inputs[:,3],   input_utau_ref)
            input_vgrady_stand = _standardization(input_vgrady, means_inputs[:,4],  stdevs_inputs[:,4],   input_utau_ref)
            input_vgradz_stand = _standardization(input_vgradz, means_inputs[:,5],  stdevs_inputs[:,5],   input_utau_ref)
            input_wgradx_stand = _standardization(input_wgradx, means_inputs[:,6],  stdevs_inputs[:,6],   input_utau_ref)
            input_wgrady_stand = _standardization(input_wgrady, means_inputs[:,7],  stdevs_inputs[:,7],   input_utau_ref)
            input_wgradz_stand = _standardization(input_wgradz, means_inputs[:,8],  stdevs_inputs[:,8],   input_utau_ref)
            #input_pgradx_stand = _standardization(input_pgradx, means_inputs[:,9],  stdevs_inputs[:,9],   1.)
            #input_pgrady_stand = _standardization(input_pgrady, means_inputs[:,10], stdevs_inputs[:,10],  1.)
            #input_pgradz_stand = _standardization(input_pgradz, means_inputs[:,11], stdevs_inputs[:,11],  1.)
    
    
    #Standardize labels
    #NOTE: the labels are already made dimensionless in the training data procedure, and thus in contrast to the inputs do not have to be multiplied by a scaling factor. 
    with tf.name_scope("standardization_labels"): #Group nodes in name scope for easier visualisation in TensorBoard
        a4 = tf.print("labels: ", labels, output_stream=tf.logging.info, summarize=-1)
        #a8 = tf.print("labels: ", tf.shape(labels), output_stream=tf.logging.info, summarize=-1)
        labels_means = tf.math.subtract(labels, means_labels)
        #a4 = tf.print("labels_means: ", labels_means[0,:], output_stream=tf.logging.info, summarize=-1)
        labels_stand = tf.math.divide(labels_means, stdevs_labels, name = 'labels_stand')
        #a5 = tf.print("labels_stand: ", labels_stand[0,:], output_stream=tf.logging.info, summarize=-1)
    
    #Create mask to disgard boundary conditions during training (currently tested with no-slip BC in vertical direction and periodic BCs in horizontal directions), since these are already known in inference mode. At the top and bottom wall, several components are by definition 0 in turbulent channel flow because of the no-slip BCs. Consequently, the corresponding output values are explicitly set to 0 by masking them such that their influence on the training is diminished.
    #NOTE1:make sure this part of the graph is not executed in inference/predict mode, where the boundary conditions are discarded outside the network anyway as they are already predefined in the model.
    if not mode == tf.estimator.ModeKeys.PREDICT:
        flag_topwall = tf.identity(features['flag_topwall_sample'], name = 'flag_topwall') 
        flag_bottomwall = tf.identity(features['flag_bottomwall_sample'], name = 'flag_bottomwall')
        with tf.name_scope("mask_creation"):
            flag_topwall_bool = tf.expand_dims(tf.math.not_equal(flag_topwall, 1), axis=1) #Select all samples that are not located at the top wall, and extend dim to be compatible with other arrays
            flag_bottomwall_bool = tf.expand_dims(tf.math.not_equal(flag_bottomwall, 1), axis=1) #Select all samples that are not located at the bottom wall, and extend dim to be compatible with other arrays
            #a1 = tf.print("channel_bool: ", channel_bool, output_stream=tf.logging.info, summarize=-1)
        
            #Select all transport components where vertical boundary condition (i.e. no-slip BC) do not apply and that are not discarded in inference mode.
            #NOTE1: zw_upstream and zw_downstream are not selected at the bottom wall, although no explicit no-slip vertical BC is valid there. These components are not used during inference to keep the application of the MLP symmetric, and therefore these components are out of convenience set equal to 0 just as the other components where an explicit vertical no-slip BC is defined. In that way, it does not influence the training and does not introduce asymmetry in the predictions of the MLP.
            #NOTE2: zu/zv_upstream/downstream are not set to 0 at both the bottom and top walls, because we included the unresolved viscous flux, on top of the unresolved turbulent flux, in the labels. This means that these components are not 0 anymore, despite that the no-slip BCs are valid for the unresolved turublent flux contribution. 
            components_topwall_bool = tf.constant(
                    [[True,  True,  #xu_upstream, xu_downstream
                      True,  True,  #yu_upstream, yu_downstream
                      #True,  False, #zu_upstream, zu_downstream
                      True,  True, #zu_upstream, zu_downstream
                      True,  True,  #xv_upstream, xv_downstream
                      True,  True,  #yv_upstream, yv_downstream
                      #True,  False, #zv_upstream, zv_downstream
                      True,  True, #zv_upstream, zv_downstream
                      True,  True,  #xw_upstream, xw_downstream
                      True,  True,  #yw_upstream, yw_downstream
                      True,  True]])#zw_upstream, zw_downstream
            
            components_bottomwall_bool = tf.constant(
                    [[True,  True,  #xu_upstream, xu_downstream
                      True,  True,  #yu_upstream, yu_downstream
                      #False, True,  #zu_upstream, zu_downstream
                      True,  True,  #zu_upstream, zu_downstream
                      True,  True,  #xv_upstream, xv_downstream
                      True,  True,  #yv_upstream, yv_downstream
                      #False, True,  #zv_upstream, zv_downstream
                      True, True,  #zv_upstream, zv_downstream
                      False, False, #xw_upstream, xw_downstream
                      False, False, #yw_upstream, yw_downstream
                      False, False]])#zw_upstream, zw_downstream
            
            #a2 = tf.print("nonstaggered_components_bool: ", nonstaggered_components_bool, output_stream=tf.logging.info, summarize=-1)
            mask_top    = tf.cast(tf.math.logical_or(flag_topwall_bool, components_topwall_bool), tf.float32, name = 'mask_top') #Cast boolean to float for multiplications below
            mask_bottom = tf.cast(tf.math.logical_or(flag_bottomwall_bool, components_bottomwall_bool), tf.float32, name = 'mask_bottom') #Cast boolean to float for multiplications below
            #a3 = tf.print("mask: ", mask, output_stream=tf.logging.info, summarize=-1)
            mask = tf.multiply(mask_top, mask_bottom, name = 'mask_noslipBC')
        #output_layer_mask = tf.math.multiply(output_layer_tot, mask, name = 'output_masked')
        #a3 = tf.print("flag_bottomwall: ", flag_bottomwall, output_stream=tf.logging.info, summarize=-1)
        #a4 = tf.print("flag_topwall: ", flag_topwall, output_stream=tf.logging.info, summarize=-1)
        #a5 = tf.print("mask: ", mask[0,:], output_stream=tf.logging.info, summarize=-1)
        labels_mask = tf.math.multiply(labels_stand, mask, name = 'labels_masked') #NOTE: the concerning labels should be put to 0 because of the applied normalisation.
        #a8 = tf.print("labels_mask: ", labels_mask[0,:], output_stream=tf.logging.info, summarize=-1)
    
    #Call create_MLP three times to construct 3 separate MLPs
    #NOTE1: the sizes of the input are adjusted to train symmetrically. In doing so, it is assumed that the original size of the input was 5*5*5 grid cells!!!
    if args.gradients is None:
        
        def _adjust_sizeinput(input_variable, indices):
            with tf.name_scope('adjust_sizeinput'):
                reshaped_variable = tf.reshape(input_variable,[-1,5,5,5])
                adjusted_size_variable = reshaped_variable[indices]
                zlen = adjusted_size_variable.shape[1]
                ylen = adjusted_size_variable.shape[2]
                xlen = adjusted_size_variable.shape[3]
                final_variable = tf.reshape(adjusted_size_variable,[-1,zlen*ylen*xlen]) #Take into account the adjusted size via zlen, ylen, and xlen.
            return final_variable

        output_layer_u = create_MLP(
           [
               input_u_stand, 
               _adjust_sizeinput(input_v_stand, np.s_[:,:,1:,:-1]),
               _adjust_sizeinput(input_w_stand, np.s_[:,1:,:,:-1])],
               #_adjust_sizeinput(input_p_stand, np.s_[:,:,:,:-1])],
               #input_z],
           'u', params)
        output_layer_v = create_MLP(
           [
               _adjust_sizeinput(input_u_stand, np.s_[:,:,:-1,1:]), 
               input_v_stand, 
               _adjust_sizeinput(input_w_stand, np.s_[:,1:,:-1,:])],
               #_adjust_sizeinput(input_p_stand, np.s_[:,:,:-1,:])],
               #input_z],
           'v', params)
        output_layer_w = create_MLP(
           [
               _adjust_sizeinput(input_u_stand, np.s_[:,:-1,:,1:]), 
               _adjust_sizeinput(input_v_stand, np.s_[:,:-1,1:,:]), 
               input_w_stand],
               #_adjust_sizeinput(input_p_stand, np.s_[:,:-1,:,:])],
               #input_z],
          'w', params)

    else:

        output_layer_u = create_MLP(
           [
               input_ugradx_stand, input_ugrady_stand, input_ugradz_stand,
               input_vgradx_stand, input_vgrady_stand, input_vgradz_stand,
               input_wgradx_stand, input_wgrady_stand, input_wgradz_stand],
               #input_pgradx_stand, input_pgrady_stand, input_pgradz_stand],
               #input_z],
          'u', params)
        output_layer_v = create_MLP(
           [
               input_ugradx_stand, input_ugrady_stand, input_ugradz_stand,     
               input_vgradx_stand, input_vgrady_stand, input_vgradz_stand,
               input_wgradx_stand, input_wgrady_stand, input_wgradz_stand],  
               #input_pgradx_stand, input_pgrady_stand, input_pgradz_stand],    
               #input_z],
          'v', params)
        output_layer_w = create_MLP(
           [
               input_ugradx_stand, input_ugrady_stand, input_ugradz_stand,     
               input_vgradx_stand, input_vgrady_stand, input_vgradz_stand,
               input_wgradx_stand, input_wgrady_stand, input_wgradz_stand],
               #input_pgradx_stand, input_pgrady_stand, input_pgradz_stand],     
               #input_z],
          'w', params)

    #Concatenate output layers
    output_layer_tot = tf.concat([output_layer_u, output_layer_v, output_layer_w], axis=1, name = 'output_layer_tot')

    #Mask output layer, used during training/evaluation but NOT during inference/prediction
    if not mode == tf.estimator.ModeKeys.PREDICT:
        output_layer_mask = tf.multiply(output_layer_tot, mask, name = 'output_layer_masked')
        #Visualize in Tensorboard
        tf.summary.histogram('output_layer_mask', output_layer_mask)
    
    #Visualize outputs in TensorBoard
    tf.summary.histogram('output_layer_tot', output_layer_tot)

    ##Trick to execute tf.print ops defined in this script. For these ops, set output_stream to tf.logging.info and summarize to -1.
    #with tf.control_dependencies([a8,a9]):
    #    output_layer_mask = tf.identity(output_layer_mask)
    #    #output_layer_tot  = tf.identity(output_layer_tot)
    
    #Denormalize the output fluxes for inference/prediction
    #NOTE1: In addition to undoing the standardization, the normalisation includes a multiplication with utau_ref. Earlier in the training data generation procedure, all data was made dimensionless by utau_ref. Therefore, the utau_ref is taken into account in the denormalisation below.
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope("denormalisation_output"): #Group nodes in name scope for easier visualisation in TensorBoard
            output_stdevs      = tf.math.multiply(output_layer_tot, stdevs_labels) #On purpose the output layer without masks applied is selected, see comment before.
            output_means       = tf.math.add(output_stdevs, means_labels)
        output_denorm      = tf.math.multiply(output_means, (utau_ref ** 2), name = 'output_layer_denorm')
        #output_denorm_masked = tf.math.multiply(output_denorm, mask, name = 'output_layer_denorm_masked') NOT needed in inference/predict mode to apply masks, see comment before.
    
    #Denormalize the labels for inference
    #NOTE1: in contrast to the code above, no mask needs to be applied as the concerning labels should already evaluate to 0 after denormalisation.
    #NOTE2: this does not have to be included in the frozen graph, and thus does not have to be included in the main code.
    #NOTE3: similar to the code above, utau_ref is included in the denormalisation.
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope("denormalisation_labels"):
            labels_stdevs = tf.math.multiply(labels_stand, stdevs_labels) #NOTE: on purpose labels_stand instead of labels_mask.
            labels_means  = tf.math.add(labels_stdevs, means_labels)
        labels_denorm = tf.math.multiply(labels_means, (utau_ref ** 2), name = 'labels_denorm')
        
        #Compute predictions
        if args.benchmark is None:
            return tf.estimator.EstimatorSpec(mode, predictions={
                'pred_tau_xu_upstream':  output_denorm[:,0],  'label_tau_xu_upstream':  labels_denorm[:,0],
                'pred_tau_xu_downstream':output_denorm[:,1],  'label_tau_xu_downstream':labels_denorm[:,1],
                'pred_tau_yu_upstream':  output_denorm[:,2],  'label_tau_yu_upstream':  labels_denorm[:,2],
                'pred_tau_yu_downstream':output_denorm[:,3],  'label_tau_yu_downstream':labels_denorm[:,3],
                'pred_tau_zu_upstream':  output_denorm[:,4],  'label_tau_zu_upstream':  labels_denorm[:,4],
                'pred_tau_zu_downstream':output_denorm[:,5],  'label_tau_zu_downstream':labels_denorm[:,5],
                'pred_tau_xv_upstream':  output_denorm[:,6],  'label_tau_xv_upstream':  labels_denorm[:,6],
                'pred_tau_xv_downstream':output_denorm[:,7],  'label_tau_xv_downstream':labels_denorm[:,7],
                'pred_tau_yv_upstream':  output_denorm[:,8],  'label_tau_yv_upstream':  labels_denorm[:,8],
                'pred_tau_yv_downstream':output_denorm[:,9],  'label_tau_yv_downstream':labels_denorm[:,9],
                'pred_tau_zv_upstream':  output_denorm[:,10], 'label_tau_zv_upstream':  labels_denorm[:,10],
                'pred_tau_zv_downstream':output_denorm[:,11], 'label_tau_zv_downstream':labels_denorm[:,11],
                'pred_tau_xw_upstream':  output_denorm[:,12], 'label_tau_xw_upstream':  labels_denorm[:,12],
                'pred_tau_xw_downstream':output_denorm[:,13], 'label_tau_xw_downstream':labels_denorm[:,13],
                'pred_tau_yw_upstream':  output_denorm[:,14], 'label_tau_yw_upstream':  labels_denorm[:,14],
                'pred_tau_yw_downstream':output_denorm[:,15], 'label_tau_yw_downstream':labels_denorm[:,15],
                'pred_tau_zw_upstream':  output_denorm[:,16], 'label_tau_zw_upstream':  labels_denorm[:,16],
                'pred_tau_zw_downstream':output_denorm[:,17], 'label_tau_zw_downstream':labels_denorm[:,17],
                'tstep':features['tstep_sample'], 'zhloc':features['zhloc_sample'],
                'zloc':features['zloc_sample'], 'yhloc':features['yhloc_sample'],
                'yloc':features['yloc_sample'], 'xhloc':features['xhloc_sample'],
                'xloc':features['xloc_sample']})
 
        else:
            return tf.estimator.EstimatorSpec(mode, predictions={
                'pred_tau_xu_upstream':  output_denorm[:,0], 
                'pred_tau_xu_downstream':output_denorm[:,1], 
                'pred_tau_yu_upstream':  output_denorm[:,2], 
                'pred_tau_yu_downstream':output_denorm[:,3], 
                'pred_tau_zu_upstream':  output_denorm[:,4], 
                'pred_tau_zu_downstream':output_denorm[:,5], 
                'pred_tau_xv_upstream':  output_denorm[:,6], 
                'pred_tau_xv_downstream':output_denorm[:,7], 
                'pred_tau_yv_upstream':  output_denorm[:,8], 
                'pred_tau_yv_downstream':output_denorm[:,9], 
                'pred_tau_zv_upstream':  output_denorm[:,10],
                'pred_tau_zv_downstream':output_denorm[:,11],
                'pred_tau_xw_upstream':  output_denorm[:,12],
                'pred_tau_xw_downstream':output_denorm[:,13],
                'pred_tau_yw_upstream':  output_denorm[:,14],
                'pred_tau_yw_downstream':output_denorm[:,15],
                'pred_tau_zw_upstream':  output_denorm[:,16],
                'pred_tau_zw_downstream':output_denorm[:,17]}) 
    
    #Compute loss
    weights = tf.constant([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])
    #weights = tf.constant([[1,1,1,1,10,10,1,1,1,1,1,1,10,10,1,1,1,1]])
    loss = tf.losses.mean_squared_error(labels_mask, output_layer_mask, weights=weights, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) #Mean taken over batch rather than over number of components
    #loss = tf.reduce_mean(mse_tau_total)
        
    #Define function to calculate the logarithm
    def log10(values):
        numerator = tf.log(values)
        denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    #Compute evaluation metrics.
    tf.summary.histogram('labels', labels_mask) #Visualize labels
    if mode == tf.estimator.ModeKeys.EVAL:
        mse_all, update_op = tf.metrics.mean_squared_error(labels_mask, output_layer_mask)
        log_mse_all = log10(mse_all)
        log_mse_all_update_op = log10(update_op)
        rmse_all = tf.math.sqrt(mse_all)
        rmse_all_update_op = tf.math.sqrt(update_op)
        val_metrics = {'mse': (mse_all, update_op), 'rmse':(rmse_all, rmse_all_update_op),'log_loss':(log_mse_all, log_mse_all_update_op)} 
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=val_metrics)

    #Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    #Exponentially decay learning rate
    starter_learning_rate=params['learning_rate']
    decay_rate = params['decay_rate']
    learning_rate = tf.train.exponential_decay(starter_learning_rate, tf.train.get_global_step(), iter_per_epoch, decay_rate, staircase=True, name='exponential_decay_learning_rate') #Decrease learning rate in discrete intervals
    tf.summary.scalar('learning_rate', learning_rate) #Write to TensorBoard

    log_loss_training = log10(loss)
    tf.summary.scalar('log_loss', log_loss_training)

    #optimizer = tf.train.AdamOptimizer(starter_learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    #Write all trainable variables to Tensorboard
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name,var)

    #Return tf.estimator.Estimatorspec for training mode
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

#Define settings
#batch_size = int(args.batch_size) #Uncomment when batch size is specified via input, not when it is equal to size tfrecord
num_steps = args.num_steps #Number of steps, i.e. number of batches times number of epochs
num_labels = 6 #Number of predicted transport components for each sub-MLP
random_seed = 1234
#files_per_snapshot = 53 #Number of tfrecords stored per time snapshot

#Extract friction velocity and heights from training file (where friction velocity is needed for the denormalisation implemented within the MLP)
training_file = nc.Dataset(args.training_filepath, 'r')
utau_ref = np.array(training_file['utau_ref'][:], dtype = 'f4')
heights = np.array(training_file['zc'][:], dtype = 'f4')

#Define filenames of tfrecords for training and validation
#NOTE: each tfrecords contains all the samples from a single 'snapshot' of the flow, and thus corresponds to a single time step.
nt_available = 15 #Amount of time steps that should be used for training/validation, assuming  that the number of the time step in the filenames ranges from 1 to nt_available without gaps. Reduced to prevent memory saturation with preferential sampling
#nt_available = 30 #Amount of time steps that should be used for training/validation, assuming  that the number of the time step in the filenames ranges from 1 to nt_available without gaps.
#nt_total = 100 #Amount of time steps INCLUDING all produced tfrecord files (also the ones not used for training/validation).
nt_total = 30 #Amount of time steps INCLUDING all produced tfrecord files (also the ones not used for training/validation).
#nt_available = 2 #FOR TESTING PURPOSES ONLY!
#nt_total = 3 #FOR TESTING PURPOSES ONLY!
#Preferential sampling: still use same three validation files as before (i.e. time steps 27-29)
#train_stepnumbers = np.array([0,1,2,3,4],dtype=np.int32) #Test performance when only five snapshots are used for training
train_stepnumbers = np.arange(nt_available)
val_stepnumbers = np.array([27,28,29],dtype=np.int32)
#Uncomment two lines below when preferential sampling is not used
#time_numbers = np.arange(nt_available)
#train_stepnumbers, val_stepnumbers = split_train_val(time_numbers, 0.1) #Set aside 10% of files for validation.
train_filenames = np.zeros((len(train_stepnumbers)*(len(heights)+50),), dtype=object)
val_filenames   = np.zeros((len(val_stepnumbers)*(len(heights)+50),), dtype=object)
#train_filenames = np.zeros((len(train_stepnumbers)*files_per_snapshot,), dtype=object) #i.e. 53 files stored per time snapshot, which already contain preferential samples
#val_filenames   = np.zeros((len(val_stepnumbers)*files_per_snapshot,), dtype=object)

#Uncomment lines below when batches should consist of individual vertical levels
#Training files
i=0
for train_stepnumber in train_stepnumbers: #Generate training filenames from selected step numbers and total steps
    k=0
    n=0
    for height in heights: #Loop over all vertical levels where samples are stored

        #Add preferential sampling by selecting the tfrecords close to the bottom/top walls multiple times, while selecting the tfrecords in the middle of the channel only once. This may improve the performance of the MLP close to the walls, where it matters most.

        number_added = max(max(10-2*k,1),10-2*(len(heights)-1-k))

        for c in range(number_added):

            if args.gradients is None:
                train_filenames[i*(len(heights)+50)+n] = args.input_dir + 'training_time_step_{0}_of_{1}_height_{2}.tfrecords'.format(train_stepnumber+1, nt_total, height)
            else:
                train_filenames[i*(len(heights)+50)+n] = args.input_dir + 'training_time_step_{0}_of_{1}_gradients_height_{2}.tfrecords'.format(train_stepnumber+1, nt_total, height)
            n+=1
        k+=1
    i+=1

#Validation files
i=0
for val_stepnumber in val_stepnumbers: #Generate validation filenames from selected step numbers and total steps
    k=0
    n=0
    for height in heights: #Loop over all vertical levels where samples are stored

        #Add preferential sampling by selecting the tfrecords close to the bottom/top walls multiple times, while selecting the tfrecords in the middle of the channel only once. This may improve the performance of the MLP close to the walls, where it matters most.

        number_added = max(max(10-2*k,1),10-2*(len(heights)-1-k))

        for c in range(number_added):

            if args.gradients is None:
                val_filenames[i*(len(heights)+50)+n] = args.input_dir + 'training_time_step_{0}_of_{1}_height_{2}.tfrecords'.format(val_stepnumber+1, nt_total, height)
            else:
                val_filenames[i*(len(heights)+50)+n] = args.input_dir + 'training_time_step_{0}_of_{1}_gradients_height_{2}.tfrecords'.format(val_stepnumber+1, nt_total, height)
            n+=1
        k+=1
    i+=1

##Uncomment lines below when batches should not consist of individual vertical levels!
#k=0
#for train_stepnumber in train_stepnumbers: #Generate training filenames from selected step numbers and total steps
#
#    for i in range(files_per_snapshot): #Loop over all files stored per time snapshot
#
#        #Add preferential sampling by selecting the tfrecords close to the bottom/top walls multiple times, while selecting the tfrecords in the middle of the channel only once. This may improve the performance of the MLP close to the walls, where it matters most.
#        
#        if args.gradients is None:
#            train_filenames[k] = args.input_dir + 'training_time_step_{0}_of_{1}_file_{2}.tfrecords'.format(train_stepnumber+1, nt_total, i+1)
#        else:
#            train_filenames[k] = args.input_dir + 'training_time_step_{0}_of_{1}_gradients_file_{2}.tfrecords'.format(train_stepnumber+1, nt_total, i+1)
#
#        k+=1
#
#k=0
#for val_stepnumber in val_stepnumbers: #Generate training filenames from selected step numbers and total steps
#
#    for i in range(files_per_snapshot): #Loop over all files stored per time snapshot
#
#        #Add preferential sampling by selecting the tfrecords close to the bottom/top walls multiple times, while selecting the tfrecords in the middle of the channel only once. This may improve the performance of the MLP close to the walls, where it matters most.
#        
#        if args.gradients is None:
#            val_filenames[k] = args.input_dir + 'training_time_step_{0}_of_{1}_file_{2}.tfrecords'.format(val_stepnumber+1, nt_total, i+1)
#        else:
#            val_filenames[k] = args.input_dir + 'training_time_step_{0}_of_{1}_gradients_file_{2}.tfrecords'.format(val_stepnumber+1, nt_total, i+1)
#
#        k+=1

#Print filenames to check
np.set_printoptions(threshold=np.inf)
print("Training files:")
print(train_filenames)
print(train_filenames.shape)
print("Validation files:")
print(val_filenames)

#Randomly shuffle filenames
np.random.shuffle(train_filenames)
np.random.shuffle(val_filenames)

#Calculate number of samples per tfrecord, used to ensure that each tfrecord corresponds to one batch. The calculated value is printed to check that the number of stored samples is indeed correct
#NOTE1: this relatively complicated solution is needed because no high-level API exists to determine this
#NOTE2: it is assumed that EACH tfrecord file contains the same number of records!!!
samples_per_tfrecord = 0
for record in tf.python_io.tf_record_iterator(train_filenames[0]):
    samples_per_tfrecord += 1
print("Samples per tfrecord: ", samples_per_tfrecord)
batch_size = samples_per_tfrecord #Set batch size equal to number of samples in tfrecord

#Determine number of iterations per epoch simply by calculating the number of tfrecord files in the training set (ONLY valid when each batch corresponds to one training file)
iter_per_epoch = len(train_filenames)

#Calculate means and stdevs for input variables (which is needed for the normalisation).
#NOTE: in the code below, it is made sure that only the means and stdevs of the time steps used for training are taken into account.
means_stdevs_filepath = args.stored_means_stdevs_filepath
means_stdevs_file     = nc.Dataset(means_stdevs_filepath, 'r')

means_dict_t  = {}
stdevs_dict_t = {}
if args.gradients is None:
    means_dict_t['uc'] = np.array(means_stdevs_file['mean_uc'][:])
    means_dict_t['vc'] = np.array(means_stdevs_file['mean_vc'][:])
    means_dict_t['wc'] = np.array(means_stdevs_file['mean_wc'][:])
    #means_dict_t['pc'] = np.array(means_stdevs_file['mean_pc'][:])
    
    stdevs_dict_t['uc'] = np.array(means_stdevs_file['stdev_uc'][:])
    stdevs_dict_t['vc'] = np.array(means_stdevs_file['stdev_vc'][:])
    stdevs_dict_t['wc'] = np.array(means_stdevs_file['stdev_wc'][:])
    #stdevs_dict_t['pc'] = np.array(means_stdevs_file['stdev_pc'][:])

else:
    means_dict_t['ugradx'] = np.array(means_stdevs_file['mean_ugradx'][:])
    means_dict_t['ugrady'] = np.array(means_stdevs_file['mean_ugrady'][:])
    means_dict_t['ugradz'] = np.array(means_stdevs_file['mean_ugradz'][:])

    means_dict_t['vgradx'] = np.array(means_stdevs_file['mean_vgradx'][:])
    means_dict_t['vgrady'] = np.array(means_stdevs_file['mean_vgrady'][:])
    means_dict_t['vgradz'] = np.array(means_stdevs_file['mean_vgradz'][:])

    means_dict_t['wgradx'] = np.array(means_stdevs_file['mean_wgradx'][:])
    means_dict_t['wgrady'] = np.array(means_stdevs_file['mean_wgrady'][:])
    means_dict_t['wgradz'] = np.array(means_stdevs_file['mean_wgradz'][:])

    #means_dict_t['pgradx'] = np.array(means_stdevs_file['mean_pgradx'][:])
    #means_dict_t['pgrady'] = np.array(means_stdevs_file['mean_pgrady'][:])
    #means_dict_t['pgradz'] = np.array(means_stdevs_file['mean_pgradz'][:])

    stdevs_dict_t['ugradx'] = np.array(means_stdevs_file['stdev_ugradx'][:])
    stdevs_dict_t['ugrady'] = np.array(means_stdevs_file['stdev_ugrady'][:])
    stdevs_dict_t['ugradz'] = np.array(means_stdevs_file['stdev_ugradz'][:])

    stdevs_dict_t['vgradx'] = np.array(means_stdevs_file['stdev_vgradx'][:])
    stdevs_dict_t['vgrady'] = np.array(means_stdevs_file['stdev_vgrady'][:])
    stdevs_dict_t['vgradz'] = np.array(means_stdevs_file['stdev_vgradz'][:])

    stdevs_dict_t['wgradx'] = np.array(means_stdevs_file['stdev_wgradx'][:])
    stdevs_dict_t['wgrady'] = np.array(means_stdevs_file['stdev_wgrady'][:])
    stdevs_dict_t['wgradz'] = np.array(means_stdevs_file['stdev_wgradz'][:])

    #stdevs_dict_t['pgradx'] = np.array(means_stdevs_file['stdev_pgradx'][:])
    #stdevs_dict_t['pgrady'] = np.array(means_stdevs_file['stdev_pgrady'][:])
    #stdevs_dict_t['pgradz'] = np.array(means_stdevs_file['stdev_pgradz'][:])

#Extract mean & standard deviation labels
means_dict_t['unres_tau_xu_sample']    = np.array(means_stdevs_file['mean_unres_tau_xu_sample'][:])
stdevs_dict_t['unres_tau_xu_sample']   = np.array(means_stdevs_file['stdev_unres_tau_xu_sample'][:])
means_dict_t['unres_tau_yu_sample']    = np.array(means_stdevs_file['mean_unres_tau_yu_sample'][:])
stdevs_dict_t['unres_tau_yu_sample']   = np.array(means_stdevs_file['stdev_unres_tau_yu_sample'][:])
means_dict_t['unres_tau_zu_sample']    = np.array(means_stdevs_file['mean_unres_tau_zu_sample'][:])
stdevs_dict_t['unres_tau_zu_sample']   = np.array(means_stdevs_file['stdev_unres_tau_zu_sample'][:])
means_dict_t['unres_tau_xv_sample']    = np.array(means_stdevs_file['mean_unres_tau_xv_sample'][:])
stdevs_dict_t['unres_tau_xv_sample']   = np.array(means_stdevs_file['stdev_unres_tau_xv_sample'][:])
means_dict_t['unres_tau_yv_sample']    = np.array(means_stdevs_file['mean_unres_tau_yv_sample'][:])
stdevs_dict_t['unres_tau_yv_sample']   = np.array(means_stdevs_file['stdev_unres_tau_yv_sample'][:])
means_dict_t['unres_tau_zv_sample']    = np.array(means_stdevs_file['mean_unres_tau_zv_sample'][:])
stdevs_dict_t['unres_tau_zv_sample']   = np.array(means_stdevs_file['stdev_unres_tau_zv_sample'][:])
means_dict_t['unres_tau_xw_sample']    = np.array(means_stdevs_file['mean_unres_tau_xw_sample'][:])
stdevs_dict_t['unres_tau_xw_sample']   = np.array(means_stdevs_file['stdev_unres_tau_xw_sample'][:])
means_dict_t['unres_tau_yw_sample']    = np.array(means_stdevs_file['mean_unres_tau_yw_sample'][:])
stdevs_dict_t['unres_tau_yw_sample']   = np.array(means_stdevs_file['stdev_unres_tau_yw_sample'][:])
means_dict_t['unres_tau_zw_sample']    = np.array(means_stdevs_file['mean_unres_tau_zw_sample'][:])
stdevs_dict_t['unres_tau_zw_sample']   = np.array(means_stdevs_file['stdev_unres_tau_zw_sample'][:])

means_dict_avgt  = {}
stdevs_dict_avgt = {}

if args.gradients is None:
    means_dict_avgt['uc'] = np.mean(means_dict_t['uc'][train_stepnumbers])
    means_dict_avgt['vc'] = np.mean(means_dict_t['vc'][train_stepnumbers])
    means_dict_avgt['wc'] = np.mean(means_dict_t['wc'][train_stepnumbers])
    #means_dict_avgt['pc'] = np.mean(means_dict_t['pc'][train_stepnumbers])
    
    stdevs_dict_avgt['uc'] = np.mean(stdevs_dict_t['uc'][train_stepnumbers])
    stdevs_dict_avgt['vc'] = np.mean(stdevs_dict_t['vc'][train_stepnumbers])
    stdevs_dict_avgt['wc'] = np.mean(stdevs_dict_t['wc'][train_stepnumbers])
    #stdevs_dict_avgt['pc'] = np.mean(stdevs_dict_t['pc'][train_stepnumbers])

else:
    means_dict_avgt['ugradx'] = np.mean(means_dict_t['ugradx'][train_stepnumbers])
    means_dict_avgt['ugrady'] = np.mean(means_dict_t['ugrady'][train_stepnumbers])
    means_dict_avgt['ugradz'] = np.mean(means_dict_t['ugradz'][train_stepnumbers])

    means_dict_avgt['vgradx'] = np.mean(means_dict_t['vgradx'][train_stepnumbers])
    means_dict_avgt['vgrady'] = np.mean(means_dict_t['vgrady'][train_stepnumbers])
    means_dict_avgt['vgradz'] = np.mean(means_dict_t['vgradz'][train_stepnumbers])

    means_dict_avgt['wgradx'] = np.mean(means_dict_t['wgradx'][train_stepnumbers])
    means_dict_avgt['wgrady'] = np.mean(means_dict_t['wgrady'][train_stepnumbers])
    means_dict_avgt['wgradz'] = np.mean(means_dict_t['wgradz'][train_stepnumbers])

    #means_dict_avgt['pgradx'] = np.mean(means_dict_t['pgradx'][train_stepnumbers])
    #means_dict_avgt['pgrady'] = np.mean(means_dict_t['pgrady'][train_stepnumbers])
    #means_dict_avgt['pgradz'] = np.mean(means_dict_t['pgradz'][train_stepnumbers])

    stdevs_dict_avgt['ugradx'] = np.mean(stdevs_dict_t['ugradx'][train_stepnumbers])
    stdevs_dict_avgt['ugrady'] = np.mean(stdevs_dict_t['ugrady'][train_stepnumbers])
    stdevs_dict_avgt['ugradz'] = np.mean(stdevs_dict_t['ugradz'][train_stepnumbers])

    stdevs_dict_avgt['vgradx'] = np.mean(stdevs_dict_t['vgradx'][train_stepnumbers])
    stdevs_dict_avgt['vgrady'] = np.mean(stdevs_dict_t['vgrady'][train_stepnumbers])
    stdevs_dict_avgt['vgradz'] = np.mean(stdevs_dict_t['vgradz'][train_stepnumbers])

    stdevs_dict_avgt['wgradx'] = np.mean(stdevs_dict_t['wgradx'][train_stepnumbers])
    stdevs_dict_avgt['wgrady'] = np.mean(stdevs_dict_t['wgrady'][train_stepnumbers])
    stdevs_dict_avgt['wgradz'] = np.mean(stdevs_dict_t['wgradz'][train_stepnumbers])

    #stdevs_dict_avgt['pgradx'] = np.mean(stdevs_dict_t['pgradx'][train_stepnumbers])
    #stdevs_dict_avgt['pgrady'] = np.mean(stdevs_dict_t['pgrady'][train_stepnumbers])
    #stdevs_dict_avgt['pgradz'] = np.mean(stdevs_dict_t['pgradz'][train_stepnumbers])

#Extract temporally averaged (over the time steps used for training) mean & standard deviation labels
means_dict_avgt['unres_tau_xu_sample']    = np.mean(means_dict_t['unres_tau_xu_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_xu_sample']   = np.mean(stdevs_dict_t['unres_tau_xu_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_yu_sample']    = np.mean(means_dict_t['unres_tau_yu_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_yu_sample']   = np.mean(stdevs_dict_t['unres_tau_yu_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_zu_sample']    = np.mean(means_dict_t['unres_tau_zu_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_zu_sample']   = np.mean(stdevs_dict_t['unres_tau_zu_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_xv_sample']    = np.mean(means_dict_t['unres_tau_xv_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_xv_sample']   = np.mean(stdevs_dict_t['unres_tau_xv_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_yv_sample']    = np.mean(means_dict_t['unres_tau_yv_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_yv_sample']   = np.mean(stdevs_dict_t['unres_tau_yv_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_zv_sample']    = np.mean(means_dict_t['unres_tau_zv_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_zv_sample']   = np.mean(stdevs_dict_t['unres_tau_zv_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_xw_sample']    = np.mean(means_dict_t['unres_tau_xw_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_xw_sample']   = np.mean(stdevs_dict_t['unres_tau_xw_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_yw_sample']    = np.mean(means_dict_t['unres_tau_yw_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_yw_sample']   = np.mean(stdevs_dict_t['unres_tau_yw_sample'][train_stepnumbers])
means_dict_avgt['unres_tau_zw_sample']    = np.mean(means_dict_t['unres_tau_zw_sample'][train_stepnumbers])
stdevs_dict_avgt['unres_tau_zw_sample']   = np.mean(stdevs_dict_t['unres_tau_zw_sample'][train_stepnumbers])

#Set configuration
config = tf.ConfigProto(log_device_placement=False)
# config.gpu_options.allow_growth = True
config.intra_op_parallelism_threads = args.intra_op_parallelism_threads
config.inter_op_parallelism_threads = args.inter_op_parallelism_threads
os.environ['KMP_BLOCKTIME'] = str(0)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = str(args.intra_op_parallelism_threads)

#Set warmstart_dir to None to disable it
warmstart_dir = None

#Create RunConfig object to save check_point in the model_dir according to the specified schedule, and to define the session config
my_checkpointing_config = tf.estimator.RunConfig(model_dir=args.checkpoint_dir, tf_random_seed=random_seed, save_summary_steps=args.summary_steps, save_checkpoints_steps=args.checkpoint_steps, save_checkpoints_secs = None,session_config=config,keep_checkpoint_max=None, keep_checkpoint_every_n_hours=10000, log_step_count_steps=10, train_distribute=None) #Provide tf.contrib.distribute.DistributionStrategy instance to train_distribute parameter for distributed training

#Define hyperparameters
if args.gradients is None:
    kernelsize_conv1 = 5
else:
    kernelsize_conv1 = 3

hyperparams =  {
'n_dense1':args.n_hidden, #Neurons in hidden layer for each control volume
'activation_function':tf.nn.leaky_relu, #NOTE: Define new activation function based on tf.nn.leaky_relu with lambda to adjust the default value for alpha (0.2)
'kernel_initializer':tf.initializers.he_uniform(),
'learning_rate':0.001, #With exponential decay this is only the starting learning rate
'decay_rate':0.993 #Uncomment only with exponential decay learning rate
#'learning_rate':0.0001
#'learning_rate':0.00001
}
print("number of neurons in hidden layer: ", str(args.n_hidden))
print("Checkpoint directory: ", args.checkpoint_dir)

#Instantiate an Estimator with model defined by model_fn
MLP = tf.estimator.Estimator(model_fn = model_fn, config=my_checkpointing_config, params = hyperparams, model_dir=args.checkpoint_dir, warm_start_from = warmstart_dir)

profiler_hook = tf.train.ProfilerHook(save_steps = args.profile_steps, output_dir = args.checkpoint_dir) #Hook designed for storing runtime statistics in Chrome trace JSON-format, which can be used in conjuction with the other summaries stored during training in Tensorboard.

if args.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    hooks = [profiler_hook, debug_hook]
#    hooks = [bcast_hook, debug_hook]
else:
    hooks = [profiler_hook]

#Train and evaluate MLP
train_spec = tf.estimator.TrainSpec(input_fn=lambda:train_input_fn(train_filenames, batch_size), max_steps=num_steps, hooks=hooks)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda:eval_input_fn(val_filenames, batch_size), steps=None, name='MLP1', start_delay_secs=30, throttle_secs=0)#NOTE: throttle_secs=0 implies that for every stored checkpoint the validation error is calculated
tf.estimator.train_and_evaluate(MLP, train_spec, eval_spec)

#NOTE: MLP.predict appeared to be unsuitable to compare the predictions from the MLP to the true labels stored in the TFRecords files: the labels are discarded by the tf.estimator.Estimator in predict mode. The alternative is the 'hacky' solution implemented in the code below.


######
#'Hacky' solution to: 
# 1) Compare the predictions of the MLP to the true labels stored in the TFRecords files (NOT in benchmark mode).
# 2) Set-up and store the inference/prediction graph.
#NOTE1: the input and model function are called manually rather than using the tf.estimator.Estimator syntax.
#NOTE2: the resulting predictions and labels are automatically stored in a netCDF-file called MLP_predictions.nc, which is placed in the specified checkpoint_dir.
#NOTE3: this implementation of the inference is computationally not efficient, but does allow to inspect and visualize the predictions afterwards in detail using the produced netCDF-file and other scripts.
print('Inference mode started.')

create_file = True #Flag to make sure netCDF file is initialized

#Initialize variables for keeping track of iterations
tot_sample_end = 0
tot_sample_begin = tot_sample_end

#Intialize flag to store inference graph only once
store_graph = True

#Loop over val files to prevent memory overflow issues
for val_filename in val_filenames:

    tf.reset_default_graph() #Reset the graph for each tfrecord (i.e. each flow 'snapshot')

    #Generate iterator to extract features and labels from tfrecords
    iterator = eval_input_fn([val_filename], batch_size).make_initializable_iterator() #All samples present in val_filenames are used for validation once.

    #Define operation to extract features and labels from iterator
    fes, lbls = iterator.get_next()

    #Define operation to generate predictions for extracted features and labels
    preds_op = model_fn(fes, lbls, \
                    tf.estimator.ModeKeys.PREDICT, hyperparams).predictions

    #Create saver MLP_model such that it can be restored in the tf.Session() below
    saver = tf.train.Saver()
    
    with tf.Session(config=config) as sess:

        #Restore MLP_model within tf.Session()
        #tf.reset_default_graph() #Make graph empty before restoring
        ckpt  = tf.train.get_checkpoint_state(args.checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)

        #Store inference graph
        tf.io.write_graph(sess.graph, args.checkpoint_dir, 'inference_graph.pbtxt', as_text = True)
        
        #Execute code below NOT in benchmark mode, otherwhise break out of for-loop
        if args.benchmark is not None:
            break

        else:
            #Initialize iterator
            sess.run(iterator.initializer)

            #Create/open netCDF-file to store predictions and labels
            if create_file:
                filepath = args.checkpoint_dir + '/MLP_predictions.nc'
                predictions_file = nc.Dataset(filepath, 'w')
                dim_ns = predictions_file.createDimension("ns",None)

                #Create variables for storage
                var_pred_tau_xu_upstream        = predictions_file.createVariable("preds_values_tau_xu_upstream","f8",("ns",))
                var_pred_random_tau_xu_upstream = predictions_file.createVariable("preds_values_random_tau_xu_upstream","f8",("ns",))
                var_lbl_tau_xu_upstream         = predictions_file.createVariable("lbls_values_tau_xu_upstream","f8",("ns",))
                var_res_tau_xu_upstream         = predictions_file.createVariable("residuals_tau_xu_upstream","f8",("ns",))
                var_res_random_tau_xu_upstream  = predictions_file.createVariable("residuals_random_tau_xu_upstream","f8",("ns",))
                #
                var_pred_tau_xu_downstream        = predictions_file.createVariable("preds_values_tau_xu_downstream","f8",("ns",))
                var_pred_random_tau_xu_downstream = predictions_file.createVariable("preds_values_random_tau_xu_downstream","f8",("ns",))
                var_lbl_tau_xu_downstream         = predictions_file.createVariable("lbls_values_tau_xu_downstream","f8",("ns",))
                var_res_tau_xu_downstream         = predictions_file.createVariable("residuals_tau_xu_downstream","f8",("ns",))
                var_res_random_tau_xu_downstream  = predictions_file.createVariable("residuals_random_tau_xu_downstream","f8",("ns",))
                #
                var_pred_tau_yu_upstream        = predictions_file.createVariable("preds_values_tau_yu_upstream","f8",("ns",))
                var_pred_random_tau_yu_upstream = predictions_file.createVariable("preds_values_random_tau_yu_upstream","f8",("ns",))
                var_lbl_tau_yu_upstream         = predictions_file.createVariable("lbls_values_tau_yu_upstream","f8",("ns",))
                var_res_tau_yu_upstream         = predictions_file.createVariable("residuals_tau_yu_upstream","f8",("ns",))
                var_res_random_tau_yu_upstream  = predictions_file.createVariable("residuals_random_tau_yu_upstream","f8",("ns",))
                #
                var_pred_tau_yu_downstream        = predictions_file.createVariable("preds_values_tau_yu_downstream","f8",("ns",))
                var_pred_random_tau_yu_downstream = predictions_file.createVariable("preds_values_random_tau_yu_downstream","f8",("ns",))
                var_lbl_tau_yu_downstream         = predictions_file.createVariable("lbls_values_tau_yu_downstream","f8",("ns",))
                var_res_tau_yu_downstream         = predictions_file.createVariable("residuals_tau_yu_downstream","f8",("ns",))
                var_res_random_tau_yu_downstream  = predictions_file.createVariable("residuals_random_tau_yu_downstream","f8",("ns",))
                #
                var_pred_tau_zu_upstream        = predictions_file.createVariable("preds_values_tau_zu_upstream","f8",("ns",))
                var_pred_random_tau_zu_upstream = predictions_file.createVariable("preds_values_random_tau_zu_upstream","f8",("ns",))
                var_lbl_tau_zu_upstream         = predictions_file.createVariable("lbls_values_tau_zu_upstream","f8",("ns",))
                var_res_tau_zu_upstream         = predictions_file.createVariable("residuals_tau_zu_upstream","f8",("ns",))
                var_res_random_tau_zu_upstream  = predictions_file.createVariable("residuals_random_tau_zu_upstream","f8",("ns",))
                #
                var_pred_tau_zu_downstream        = predictions_file.createVariable("preds_values_tau_zu_downstream","f8",("ns",))
                var_pred_random_tau_zu_downstream = predictions_file.createVariable("preds_values_random_tau_zu_downstream","f8",("ns",))
                var_lbl_tau_zu_downstream         = predictions_file.createVariable("lbls_values_tau_zu_downstream","f8",("ns",))
                var_res_tau_zu_downstream         = predictions_file.createVariable("residuals_tau_zu_downstream","f8",("ns",))
                var_res_random_tau_zu_downstream  = predictions_file.createVariable("residuals_random_tau_zu_downstream","f8",("ns",))
                #
                var_pred_tau_xv_upstream        = predictions_file.createVariable("preds_values_tau_xv_upstream","f8",("ns",))
                var_pred_random_tau_xv_upstream = predictions_file.createVariable("preds_values_random_tau_xv_upstream","f8",("ns",))
                var_lbl_tau_xv_upstream         = predictions_file.createVariable("lbls_values_tau_xv_upstream","f8",("ns",))
                var_res_tau_xv_upstream         = predictions_file.createVariable("residuals_tau_xv_upstream","f8",("ns",))
                var_res_random_tau_xv_upstream  = predictions_file.createVariable("residuals_random_tau_xv_upstream","f8",("ns",))
                #
                var_pred_tau_xv_downstream        = predictions_file.createVariable("preds_values_tau_xv_downstream","f8",("ns",))
                var_pred_random_tau_xv_downstream = predictions_file.createVariable("preds_values_random_tau_xv_downstream","f8",("ns",))
                var_lbl_tau_xv_downstream         = predictions_file.createVariable("lbls_values_tau_xv_downstream","f8",("ns",))
                var_res_tau_xv_downstream         = predictions_file.createVariable("residuals_tau_xv_downstream","f8",("ns",))
                var_res_random_tau_xv_downstream  = predictions_file.createVariable("residuals_random_tau_xv_downstream","f8",("ns",))
                #
                var_pred_tau_yv_upstream        = predictions_file.createVariable("preds_values_tau_yv_upstream","f8",("ns",))
                var_pred_random_tau_yv_upstream = predictions_file.createVariable("preds_values_random_tau_yv_upstream","f8",("ns",))
                var_lbl_tau_yv_upstream         = predictions_file.createVariable("lbls_values_tau_yv_upstream","f8",("ns",))
                var_res_tau_yv_upstream         = predictions_file.createVariable("residuals_tau_yv_upstream","f8",("ns",))
                var_res_random_tau_yv_upstream  = predictions_file.createVariable("residuals_random_tau_yv_upstream","f8",("ns",))
                #
                var_pred_tau_yv_downstream        = predictions_file.createVariable("preds_values_tau_yv_downstream","f8",("ns",))
                var_pred_random_tau_yv_downstream = predictions_file.createVariable("preds_values_random_tau_yv_downstream","f8",("ns",))
                var_lbl_tau_yv_downstream         = predictions_file.createVariable("lbls_values_tau_yv_downstream","f8",("ns",))
                var_res_tau_yv_downstream         = predictions_file.createVariable("residuals_tau_yv_downstream","f8",("ns",))
                var_res_random_tau_yv_downstream  = predictions_file.createVariable("residuals_random_tau_yv_downstream","f8",("ns",))
                #
                var_pred_tau_zv_upstream        = predictions_file.createVariable("preds_values_tau_zv_upstream","f8",("ns",))
                var_pred_random_tau_zv_upstream = predictions_file.createVariable("preds_values_random_tau_zv_upstream","f8",("ns",))
                var_lbl_tau_zv_upstream         = predictions_file.createVariable("lbls_values_tau_zv_upstream","f8",("ns",))
                var_res_tau_zv_upstream         = predictions_file.createVariable("residuals_tau_zv_upstream","f8",("ns",))
                var_res_random_tau_zv_upstream  = predictions_file.createVariable("residuals_random_tau_zv_upstream","f8",("ns",))
                #
                var_pred_tau_zv_downstream        = predictions_file.createVariable("preds_values_tau_zv_downstream","f8",("ns",))
                var_pred_random_tau_zv_downstream = predictions_file.createVariable("preds_values_random_tau_zv_downstream","f8",("ns",))
                var_lbl_tau_zv_downstream         = predictions_file.createVariable("lbls_values_tau_zv_downstream","f8",("ns",))
                var_res_tau_zv_downstream         = predictions_file.createVariable("residuals_tau_zv_downstream","f8",("ns",))
                var_res_random_tau_zv_downstream  = predictions_file.createVariable("residuals_random_tau_zv_downstream","f8",("ns",))
                #
                var_pred_tau_xw_upstream        = predictions_file.createVariable("preds_values_tau_xw_upstream","f8",("ns",))
                var_pred_random_tau_xw_upstream = predictions_file.createVariable("preds_values_random_tau_xw_upstream","f8",("ns",))
                var_lbl_tau_xw_upstream         = predictions_file.createVariable("lbls_values_tau_xw_upstream","f8",("ns",))
                var_res_tau_xw_upstream         = predictions_file.createVariable("residuals_tau_xw_upstream","f8",("ns",))
                var_res_random_tau_xw_upstream  = predictions_file.createVariable("residuals_random_tau_xw_upstream","f8",("ns",))
                #
                var_pred_tau_xw_downstream        = predictions_file.createVariable("preds_values_tau_xw_downstream","f8",("ns",))
                var_pred_random_tau_xw_downstream = predictions_file.createVariable("preds_values_random_tau_xw_downstream","f8",("ns",))
                var_lbl_tau_xw_downstream         = predictions_file.createVariable("lbls_values_tau_xw_downstream","f8",("ns",))
                var_res_tau_xw_downstream         = predictions_file.createVariable("residuals_tau_xw_downstream","f8",("ns",))
                var_res_random_tau_xw_downstream  = predictions_file.createVariable("residuals_random_tau_xw_downstream","f8",("ns",))
                #
                var_pred_tau_yw_upstream        = predictions_file.createVariable("preds_values_tau_yw_upstream","f8",("ns",))
                var_pred_random_tau_yw_upstream = predictions_file.createVariable("preds_values_random_tau_yw_upstream","f8",("ns",))
                var_lbl_tau_yw_upstream         = predictions_file.createVariable("lbls_values_tau_yw_upstream","f8",("ns",))
                var_res_tau_yw_upstream         = predictions_file.createVariable("residuals_tau_yw_upstream","f8",("ns",))
                var_res_random_tau_yw_upstream  = predictions_file.createVariable("residuals_random_tau_yw_upstream","f8",("ns",))
                #
                var_pred_tau_yw_downstream        = predictions_file.createVariable("preds_values_tau_yw_downstream","f8",("ns",))
                var_pred_random_tau_yw_downstream = predictions_file.createVariable("preds_values_random_tau_yw_downstream","f8",("ns",))
                var_lbl_tau_yw_downstream         = predictions_file.createVariable("lbls_values_tau_yw_downstream","f8",("ns",))
                var_res_tau_yw_downstream         = predictions_file.createVariable("residuals_tau_yw_downstream","f8",("ns",))
                var_res_random_tau_yw_downstream  = predictions_file.createVariable("residuals_random_tau_yw_downstream","f8",("ns",))
                #
                var_pred_tau_zw_upstream        = predictions_file.createVariable("preds_values_tau_zw_upstream","f8",("ns",))
                var_pred_random_tau_zw_upstream = predictions_file.createVariable("preds_values_random_tau_zw_upstream","f8",("ns",))
                var_lbl_tau_zw_upstream         = predictions_file.createVariable("lbls_values_tau_zw_upstream","f8",("ns",))
                var_res_tau_zw_upstream         = predictions_file.createVariable("residuals_tau_zw_upstream","f8",("ns",))
                var_res_random_tau_zw_upstream  = predictions_file.createVariable("residuals_random_tau_zw_upstream","f8",("ns",))
                #
                var_pred_tau_zw_downstream        = predictions_file.createVariable("preds_values_tau_zw_downstream","f8",("ns",))
                var_pred_random_tau_zw_downstream = predictions_file.createVariable("preds_values_random_tau_zw_downstream","f8",("ns",))
                var_lbl_tau_zw_downstream         = predictions_file.createVariable("lbls_values_tau_zw_downstream","f8",("ns",))
                var_res_tau_zw_downstream         = predictions_file.createVariable("residuals_tau_zw_downstream","f8",("ns",))
                var_res_random_tau_zw_downstream  = predictions_file.createVariable("residuals_random_tau_zw_downstream","f8",("ns",))
                #
                vartstep               = predictions_file.createVariable("tstep_samples","f8",("ns",))
                varzhloc               = predictions_file.createVariable("zhloc_samples","f8",("ns",))
                varzloc                = predictions_file.createVariable("zloc_samples","f8",("ns",))
                varyhloc               = predictions_file.createVariable("yhloc_samples","f8",("ns",))
                varyloc                = predictions_file.createVariable("yloc_samples","f8",("ns",))
                varxhloc               = predictions_file.createVariable("xhloc_samples","f8",("ns",))
                varxloc                = predictions_file.createVariable("xloc_samples","f8",("ns",))

                create_file=False #Make sure file is only created once

            else:
                predictions_file = nc.Dataset(filepath, 'r+')

            while True:
                try:
                    #Execute computational graph to generate predictions
                    preds = sess.run(preds_op)

                    #Initialize variables for storage
                    preds_tau_xu_upstream               = []
                    preds_random_tau_xu_upstream        = []
                    lbls_tau_xu_upstream                = []
                    residuals_tau_xu_upstream           = []
                    residuals_random_tau_xu_upstream    = []
                    #
                    preds_tau_xu_downstream               = []
                    preds_random_tau_xu_downstream        = []
                    lbls_tau_xu_downstream                = []
                    residuals_tau_xu_downstream           = []
                    residuals_random_tau_xu_downstream    = []
                    #
                    preds_tau_yu_upstream               = []
                    preds_random_tau_yu_upstream        = []
                    lbls_tau_yu_upstream                = []
                    residuals_tau_yu_upstream           = []
                    residuals_random_tau_yu_upstream    = []
                    #
                    preds_tau_yu_downstream               = []
                    preds_random_tau_yu_downstream        = []
                    lbls_tau_yu_downstream                = []
                    residuals_tau_yu_downstream           = []
                    residuals_random_tau_yu_downstream    = []
                    #
                    preds_tau_zu_upstream               = []
                    preds_random_tau_zu_upstream        = []
                    lbls_tau_zu_upstream                = []
                    residuals_tau_zu_upstream           = []
                    residuals_random_tau_zu_upstream    = []
                    #
                    preds_tau_zu_downstream               = []
                    preds_random_tau_zu_downstream        = []
                    lbls_tau_zu_downstream                = []
                    residuals_tau_zu_downstream           = []
                    residuals_random_tau_zu_downstream    = []
                    #
                    preds_tau_xv_upstream               = []
                    preds_random_tau_xv_upstream        = []
                    lbls_tau_xv_upstream                = []
                    residuals_tau_xv_upstream           = []
                    residuals_random_tau_xv_upstream    = []
                    #
                    preds_tau_xv_downstream               = []
                    preds_random_tau_xv_downstream        = []
                    lbls_tau_xv_downstream                = []
                    residuals_tau_xv_downstream           = []
                    residuals_random_tau_xv_downstream    = []
                    #
                    preds_tau_yv_upstream               = []
                    preds_random_tau_yv_upstream        = []
                    lbls_tau_yv_upstream                = []
                    residuals_tau_yv_upstream           = []
                    residuals_random_tau_yv_upstream    = []
                    #
                    preds_tau_yv_downstream               = []
                    preds_random_tau_yv_downstream        = []
                    lbls_tau_yv_downstream                = []
                    residuals_tau_yv_downstream           = []
                    residuals_random_tau_yv_downstream    = []
                    #
                    preds_tau_zv_upstream               = []
                    preds_random_tau_zv_upstream        = []
                    lbls_tau_zv_upstream                = []
                    residuals_tau_zv_upstream           = []
                    residuals_random_tau_zv_upstream    = []
                    #
                    preds_tau_zv_downstream               = []
                    preds_random_tau_zv_downstream        = []
                    lbls_tau_zv_downstream                = []
                    residuals_tau_zv_downstream           = []
                    residuals_random_tau_zv_downstream    = []
                    #
                    preds_tau_xw_upstream               = []
                    preds_random_tau_xw_upstream        = []
                    lbls_tau_xw_upstream                = []
                    residuals_tau_xw_upstream           = []
                    residuals_random_tau_xw_upstream    = []
                    #
                    preds_tau_xw_downstream               = []
                    preds_random_tau_xw_downstream        = []
                    lbls_tau_xw_downstream                = []
                    residuals_tau_xw_downstream           = []
                    residuals_random_tau_xw_downstream    = []
                    #
                    preds_tau_yw_upstream               = []
                    preds_random_tau_yw_upstream        = []
                    lbls_tau_yw_upstream                = []
                    residuals_tau_yw_upstream           = []
                    residuals_random_tau_yw_upstream    = []
                    #
                    preds_tau_yw_downstream               = []
                    preds_random_tau_yw_downstream        = []
                    lbls_tau_yw_downstream                = []
                    residuals_tau_yw_downstream           = []
                    residuals_random_tau_yw_downstream    = []
                    #
                    preds_tau_zw_upstream               = []
                    preds_random_tau_zw_upstream        = []
                    lbls_tau_zw_upstream                = []
                    residuals_tau_zw_upstream           = []
                    residuals_random_tau_zw_upstream    = []
                    #
                    preds_tau_zw_downstream               = []
                    preds_random_tau_zw_downstream        = []
                    lbls_tau_zw_downstream                = []
                    residuals_tau_zw_downstream           = []
                    residuals_random_tau_zw_downstream    = []
                    #
                    tstep_samples       = []
                    zhloc_samples       = []
                    zloc_samples        = []
                    yhloc_samples       = []
                    yloc_samples        = []
                    xhloc_samples       = []
                    xloc_samples        = []

                    for pred_tau_xu_upstream,   lbl_tau_xu_upstream, \
                        pred_tau_xu_downstream, lbl_tau_xu_downstream, \
                        pred_tau_yu_upstream,   lbl_tau_yu_upstream, \
                        pred_tau_yu_downstream, lbl_tau_yu_downstream, \
                        pred_tau_zu_upstream,   lbl_tau_zu_upstream, \
                        pred_tau_zu_downstream, lbl_tau_zu_downstream, \
                        pred_tau_xv_upstream,   lbl_tau_xv_upstream, \
                        pred_tau_xv_downstream, lbl_tau_xv_downstream, \
                        pred_tau_yv_upstream,   lbl_tau_yv_upstream, \
                        pred_tau_yv_downstream, lbl_tau_yv_downstream, \
                        pred_tau_zv_upstream,   lbl_tau_zv_upstream, \
                        pred_tau_zv_downstream, lbl_tau_zv_downstream, \
                        pred_tau_xw_upstream,   lbl_tau_xw_upstream, \
                        pred_tau_xw_downstream, lbl_tau_xw_downstream, \
                        pred_tau_yw_upstream,   lbl_tau_yw_upstream, \
                        pred_tau_yw_downstream, lbl_tau_yw_downstream, \
                        pred_tau_zw_upstream,   lbl_tau_zw_upstream, \
                        pred_tau_zw_downstream, lbl_tau_zw_downstream, \
                        tstep, zhloc, zloc, yhloc, yloc, xhloc, xloc in zip(
                                preds['pred_tau_xu_upstream'], preds['label_tau_xu_upstream'],
                                preds['pred_tau_xu_downstream'], preds['label_tau_xu_downstream'],
                                preds['pred_tau_yu_upstream'], preds['label_tau_yu_upstream'],
                                preds['pred_tau_yu_downstream'], preds['label_tau_yu_downstream'],
                                preds['pred_tau_zu_upstream'], preds['label_tau_zu_upstream'],
                                preds['pred_tau_zu_downstream'], preds['label_tau_zu_downstream'],
                                preds['pred_tau_xv_upstream'], preds['label_tau_xv_upstream'],
                                preds['pred_tau_xv_downstream'], preds['label_tau_xv_downstream'],
                                preds['pred_tau_yv_upstream'], preds['label_tau_yv_upstream'],
                                preds['pred_tau_yv_downstream'], preds['label_tau_yv_downstream'],
                                preds['pred_tau_zv_upstream'], preds['label_tau_zv_upstream'],
                                preds['pred_tau_zv_downstream'], preds['label_tau_zv_downstream'],
                                preds['pred_tau_xw_upstream'], preds['label_tau_xw_upstream'],
                                preds['pred_tau_xw_downstream'], preds['label_tau_xw_downstream'],
                                preds['pred_tau_yw_upstream'], preds['label_tau_yw_upstream'],
                                preds['pred_tau_yw_downstream'], preds['label_tau_yw_downstream'],
                                preds['pred_tau_zw_upstream'], preds['label_tau_zw_upstream'],
                                preds['pred_tau_zw_downstream'], preds['label_tau_zw_downstream'],
                                preds['tstep'], preds['zhloc'], preds['zloc'], 
                                preds['yhloc'], preds['yloc'], preds['xhloc'], preds['xloc']):
                        # 
                        preds_tau_xu_upstream               += [pred_tau_xu_upstream]
                        lbls_tau_xu_upstream                += [lbl_tau_xu_upstream]
                        residuals_tau_xu_upstream           += [abs(pred_tau_xu_upstream-lbl_tau_xu_upstream)]
                        pred_random_tau_xu_upstream          = random.choice(preds['label_tau_xu_upstream'][:][:]) #Generate random prediction
                        preds_random_tau_xu_upstream        += [pred_random_tau_xu_upstream]
                        residuals_random_tau_xu_upstream    += [abs(pred_random_tau_xu_upstream-lbl_tau_xu_upstream)]
                        #
                        preds_tau_xu_downstream               += [pred_tau_xu_downstream]
                        lbls_tau_xu_downstream                += [lbl_tau_xu_downstream]
                        residuals_tau_xu_downstream           += [abs(pred_tau_xu_downstream-lbl_tau_xu_downstream)]
                        pred_random_tau_xu_downstream          = random.choice(preds['label_tau_xu_downstream'][:][:]) #Generate random prediction
                        preds_random_tau_xu_downstream        += [pred_random_tau_xu_downstream]
                        residuals_random_tau_xu_downstream    += [abs(pred_random_tau_xu_downstream-lbl_tau_xu_downstream)]
                        #
                        preds_tau_yu_upstream               += [pred_tau_yu_upstream]
                        lbls_tau_yu_upstream                += [lbl_tau_yu_upstream]
                        residuals_tau_yu_upstream           += [abs(pred_tau_yu_upstream-lbl_tau_yu_upstream)]
                        pred_random_tau_yu_upstream          = random.choice(preds['label_tau_yu_upstream'][:][:]) #Generate random prediction
                        preds_random_tau_yu_upstream        += [pred_random_tau_yu_upstream]
                        residuals_random_tau_yu_upstream    += [abs(pred_random_tau_yu_upstream-lbl_tau_yu_upstream)]
                        #
                        preds_tau_yu_downstream               += [pred_tau_yu_downstream]
                        lbls_tau_yu_downstream                += [lbl_tau_yu_downstream]
                        residuals_tau_yu_downstream           += [abs(pred_tau_yu_downstream-lbl_tau_yu_downstream)]
                        pred_random_tau_yu_downstream          = random.choice(preds['label_tau_yu_downstream'][:][:]) #Generate random prediction
                        preds_random_tau_yu_downstream        += [pred_random_tau_yu_downstream]
                        residuals_random_tau_yu_downstream    += [abs(pred_random_tau_yu_downstream-lbl_tau_yu_downstream)]
                        #
                        preds_tau_zu_upstream               += [pred_tau_zu_upstream]
                        lbls_tau_zu_upstream                += [lbl_tau_zu_upstream]
                        residuals_tau_zu_upstream           += [abs(pred_tau_zu_upstream-lbl_tau_zu_upstream)]
                        pred_random_tau_zu_upstream          = random.choice(preds['label_tau_zu_upstream'][:][:]) #Generate random prediction
                        preds_random_tau_zu_upstream        += [pred_random_tau_zu_upstream]
                        residuals_random_tau_zu_upstream    += [abs(pred_random_tau_zu_upstream-lbl_tau_zu_upstream)]
                        #
                        preds_tau_zu_downstream               += [pred_tau_zu_downstream]
                        lbls_tau_zu_downstream                += [lbl_tau_zu_downstream]
                        residuals_tau_zu_downstream           += [abs(pred_tau_zu_downstream-lbl_tau_zu_downstream)]
                        pred_random_tau_zu_downstream          = random.choice(preds['label_tau_zu_downstream'][:][:]) #Generate random prediction
                        preds_random_tau_zu_downstream        += [pred_random_tau_zu_downstream]
                        residuals_random_tau_zu_downstream    += [abs(pred_random_tau_zu_downstream-lbl_tau_zu_downstream)]
                        #
                        preds_tau_xv_upstream               += [pred_tau_xv_upstream]
                        lbls_tau_xv_upstream                += [lbl_tau_xv_upstream]
                        residuals_tau_xv_upstream           += [abs(pred_tau_xv_upstream-lbl_tau_xv_upstream)]
                        pred_random_tau_xv_upstream          = random.choice(preds['label_tau_xv_upstream'][:][:]) #Generate random prediction
                        preds_random_tau_xv_upstream        += [pred_random_tau_xv_upstream]
                        residuals_random_tau_xv_upstream    += [abs(pred_random_tau_xv_upstream-lbl_tau_xv_upstream)]
                        #
                        preds_tau_xv_downstream               += [pred_tau_xv_downstream]
                        lbls_tau_xv_downstream                += [lbl_tau_xv_downstream]
                        residuals_tau_xv_downstream           += [abs(pred_tau_xv_downstream-lbl_tau_xv_downstream)]
                        pred_random_tau_xv_downstream          = random.choice(preds['label_tau_xv_downstream'][:][:]) #Generate random prediction
                        preds_random_tau_xv_downstream        += [pred_random_tau_xv_downstream]
                        residuals_random_tau_xv_downstream    += [abs(pred_random_tau_xv_downstream-lbl_tau_xv_downstream)]
                        #
                        preds_tau_yv_upstream               += [pred_tau_yv_upstream]
                        lbls_tau_yv_upstream                += [lbl_tau_yv_upstream]
                        residuals_tau_yv_upstream           += [abs(pred_tau_yv_upstream-lbl_tau_yv_upstream)]
                        pred_random_tau_yv_upstream          = random.choice(preds['label_tau_yv_upstream'][:][:]) #Generate random prediction
                        preds_random_tau_yv_upstream        += [pred_random_tau_yv_upstream]
                        residuals_random_tau_yv_upstream    += [abs(pred_random_tau_yv_upstream-lbl_tau_yv_upstream)]
                        #
                        preds_tau_yv_downstream               += [pred_tau_yv_downstream]
                        lbls_tau_yv_downstream                += [lbl_tau_yv_downstream]
                        residuals_tau_yv_downstream           += [abs(pred_tau_yv_downstream-lbl_tau_yv_downstream)]
                        pred_random_tau_yv_downstream          = random.choice(preds['label_tau_yv_downstream'][:][:]) #Generate random prediction
                        preds_random_tau_yv_downstream        += [pred_random_tau_yv_downstream]
                        residuals_random_tau_yv_downstream    += [abs(pred_random_tau_yv_downstream-lbl_tau_yv_downstream)]
                        #
                        preds_tau_zv_upstream               += [pred_tau_zv_upstream]
                        lbls_tau_zv_upstream                += [lbl_tau_zv_upstream]
                        residuals_tau_zv_upstream           += [abs(pred_tau_zv_upstream-lbl_tau_zv_upstream)]
                        pred_random_tau_zv_upstream          = random.choice(preds['label_tau_zv_upstream'][:][:]) #Generate random prediction
                        preds_random_tau_zv_upstream        += [pred_random_tau_zv_upstream]
                        residuals_random_tau_zv_upstream    += [abs(pred_random_tau_zv_upstream-lbl_tau_zv_upstream)]
                        #
                        preds_tau_zv_downstream               += [pred_tau_zv_downstream]
                        lbls_tau_zv_downstream                += [lbl_tau_zv_downstream]
                        residuals_tau_zv_downstream           += [abs(pred_tau_zv_downstream-lbl_tau_zv_downstream)]
                        pred_random_tau_zv_downstream          = random.choice(preds['label_tau_zv_downstream'][:][:]) #Generate random prediction
                        preds_random_tau_zv_downstream        += [pred_random_tau_zv_downstream]
                        residuals_random_tau_zv_downstream    += [abs(pred_random_tau_zv_downstream-lbl_tau_zv_downstream)]
                        #
                        preds_tau_xw_upstream               += [pred_tau_xw_upstream]
                        lbls_tau_xw_upstream                += [lbl_tau_xw_upstream]
                        residuals_tau_xw_upstream           += [abs(pred_tau_xw_upstream-lbl_tau_xw_upstream)]
                        pred_random_tau_xw_upstream          = random.choice(preds['label_tau_xw_upstream'][:][:]) #Generate random prediction
                        preds_random_tau_xw_upstream        += [pred_random_tau_xw_upstream]
                        residuals_random_tau_xw_upstream    += [abs(pred_random_tau_xw_upstream-lbl_tau_xw_upstream)]
                        #
                        preds_tau_xw_downstream               += [pred_tau_xw_downstream]
                        lbls_tau_xw_downstream                += [lbl_tau_xw_downstream]
                        residuals_tau_xw_downstream           += [abs(pred_tau_xw_downstream-lbl_tau_xw_downstream)]
                        pred_random_tau_xw_downstream          = random.choice(preds['label_tau_xw_downstream'][:][:]) #Generate random prediction
                        preds_random_tau_xw_downstream        += [pred_random_tau_xw_downstream]
                        residuals_random_tau_xw_downstream    += [abs(pred_random_tau_xw_downstream-lbl_tau_xw_downstream)]
                        #
                        preds_tau_yw_upstream               += [pred_tau_yw_upstream]
                        lbls_tau_yw_upstream                += [lbl_tau_yw_upstream]
                        residuals_tau_yw_upstream           += [abs(pred_tau_yw_upstream-lbl_tau_yw_upstream)]
                        pred_random_tau_yw_upstream          = random.choice(preds['label_tau_yw_upstream'][:][:]) #Generate random prediction
                        preds_random_tau_yw_upstream        += [pred_random_tau_yw_upstream]
                        residuals_random_tau_yw_upstream    += [abs(pred_random_tau_yw_upstream-lbl_tau_yw_upstream)]
                        #
                        preds_tau_yw_downstream               += [pred_tau_yw_downstream]
                        lbls_tau_yw_downstream                += [lbl_tau_yw_downstream]
                        residuals_tau_yw_downstream           += [abs(pred_tau_yw_downstream-lbl_tau_yw_downstream)]
                        pred_random_tau_yw_downstream          = random.choice(preds['label_tau_yw_downstream'][:][:]) #Generate random prediction
                        preds_random_tau_yw_downstream        += [pred_random_tau_yw_downstream]
                        residuals_random_tau_yw_downstream    += [abs(pred_random_tau_yw_downstream-lbl_tau_yw_downstream)]
                        #
                        preds_tau_zw_upstream               += [pred_tau_zw_upstream]
                        lbls_tau_zw_upstream                += [lbl_tau_zw_upstream]
                        residuals_tau_zw_upstream           += [abs(pred_tau_zw_upstream-lbl_tau_zw_upstream)]
                        pred_random_tau_zw_upstream          = random.choice(preds['label_tau_zw_upstream'][:][:]) #Generate random prediction
                        preds_random_tau_zw_upstream        += [pred_random_tau_zw_upstream]
                        residuals_random_tau_zw_upstream    += [abs(pred_random_tau_zw_upstream-lbl_tau_zw_upstream)]
                        #
                        preds_tau_zw_downstream               += [pred_tau_zw_downstream]
                        lbls_tau_zw_downstream                += [lbl_tau_zw_downstream]
                        residuals_tau_zw_downstream           += [abs(pred_tau_zw_downstream-lbl_tau_zw_downstream)]
                        pred_random_tau_zw_downstream          = random.choice(preds['label_tau_zw_downstream'][:][:]) #Generate random prediction
                        preds_random_tau_zw_downstream        += [pred_random_tau_zw_downstream]
                        residuals_random_tau_zw_downstream    += [abs(pred_random_tau_zw_downstream-lbl_tau_zw_downstream)]
                        #
                        tstep_samples += [tstep]
                        zhloc_samples += [zhloc]
                        zloc_samples  += [zloc]
                        yhloc_samples += [yhloc]
                        yloc_samples  += [yloc]
                        xhloc_samples += [xhloc]
                        xloc_samples  += [xloc]

                        tot_sample_end +=1
                        #print('next sample')
                    #print('next batch')
                    
                    #Store variables
                    #
                    var_pred_tau_xu_upstream[tot_sample_begin:tot_sample_end]        = preds_tau_xu_upstream[:]
                    var_pred_random_tau_xu_upstream[tot_sample_begin:tot_sample_end] = preds_random_tau_xu_upstream[:]
                    var_lbl_tau_xu_upstream[tot_sample_begin:tot_sample_end]         = lbls_tau_xu_upstream[:]
                    var_res_tau_xu_upstream[tot_sample_begin:tot_sample_end]         = residuals_tau_xu_upstream[:]
                    var_res_random_tau_xu_upstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_xu_upstream[:]
                    #
                    var_pred_tau_xu_downstream[tot_sample_begin:tot_sample_end]        = preds_tau_xu_downstream[:]
                    var_pred_random_tau_xu_downstream[tot_sample_begin:tot_sample_end] = preds_random_tau_xu_downstream[:]
                    var_lbl_tau_xu_downstream[tot_sample_begin:tot_sample_end]         = lbls_tau_xu_downstream[:]
                    var_res_tau_xu_downstream[tot_sample_begin:tot_sample_end]         = residuals_tau_xu_downstream[:]
                    var_res_random_tau_xu_downstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_xu_downstream[:]
                    #
                    var_pred_tau_yu_upstream[tot_sample_begin:tot_sample_end]        = preds_tau_yu_upstream[:]
                    var_pred_random_tau_yu_upstream[tot_sample_begin:tot_sample_end] = preds_random_tau_yu_upstream[:]
                    var_lbl_tau_yu_upstream[tot_sample_begin:tot_sample_end]         = lbls_tau_yu_upstream[:]
                    var_res_tau_yu_upstream[tot_sample_begin:tot_sample_end]         = residuals_tau_yu_upstream[:]
                    var_res_random_tau_yu_upstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_yu_upstream[:]
                    #
                    var_pred_tau_yu_downstream[tot_sample_begin:tot_sample_end]        = preds_tau_yu_downstream[:]
                    var_pred_random_tau_yu_downstream[tot_sample_begin:tot_sample_end] = preds_random_tau_yu_downstream[:]
                    var_lbl_tau_yu_downstream[tot_sample_begin:tot_sample_end]         = lbls_tau_yu_downstream[:]
                    var_res_tau_yu_downstream[tot_sample_begin:tot_sample_end]         = residuals_tau_yu_downstream[:]
                    var_res_random_tau_yu_downstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_yu_downstream[:]
                    #
                    var_pred_tau_zu_upstream[tot_sample_begin:tot_sample_end]        = preds_tau_zu_upstream[:]
                    var_pred_random_tau_zu_upstream[tot_sample_begin:tot_sample_end] = preds_random_tau_zu_upstream[:]
                    var_lbl_tau_zu_upstream[tot_sample_begin:tot_sample_end]         = lbls_tau_zu_upstream[:]
                    var_res_tau_zu_upstream[tot_sample_begin:tot_sample_end]         = residuals_tau_zu_upstream[:]
                    var_res_random_tau_zu_upstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_zu_upstream[:]
                    #
                    var_pred_tau_zu_downstream[tot_sample_begin:tot_sample_end]        = preds_tau_zu_downstream[:]
                    var_pred_random_tau_zu_downstream[tot_sample_begin:tot_sample_end] = preds_random_tau_zu_downstream[:]
                    var_lbl_tau_zu_downstream[tot_sample_begin:tot_sample_end]         = lbls_tau_zu_downstream[:]
                    var_res_tau_zu_downstream[tot_sample_begin:tot_sample_end]         = residuals_tau_zu_downstream[:]
                    var_res_random_tau_zu_downstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_zu_downstream[:]
                    #
                    var_pred_tau_xv_upstream[tot_sample_begin:tot_sample_end]        = preds_tau_xv_upstream[:]
                    var_pred_random_tau_xv_upstream[tot_sample_begin:tot_sample_end] = preds_random_tau_xv_upstream[:]
                    var_lbl_tau_xv_upstream[tot_sample_begin:tot_sample_end]         = lbls_tau_xv_upstream[:]
                    var_res_tau_xv_upstream[tot_sample_begin:tot_sample_end]         = residuals_tau_xv_upstream[:]
                    var_res_random_tau_xv_upstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_xv_upstream[:]
                    #
                    var_pred_tau_xv_downstream[tot_sample_begin:tot_sample_end]        = preds_tau_xv_downstream[:]
                    var_pred_random_tau_xv_downstream[tot_sample_begin:tot_sample_end] = preds_random_tau_xv_downstream[:]
                    var_lbl_tau_xv_downstream[tot_sample_begin:tot_sample_end]         = lbls_tau_xv_downstream[:]
                    var_res_tau_xv_downstream[tot_sample_begin:tot_sample_end]         = residuals_tau_xv_downstream[:]
                    var_res_random_tau_xv_downstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_xv_downstream[:]
                    #
                    var_pred_tau_yv_upstream[tot_sample_begin:tot_sample_end]        = preds_tau_yv_upstream[:]
                    var_pred_random_tau_yv_upstream[tot_sample_begin:tot_sample_end] = preds_random_tau_yv_upstream[:]
                    var_lbl_tau_yv_upstream[tot_sample_begin:tot_sample_end]         = lbls_tau_yv_upstream[:]
                    var_res_tau_yv_upstream[tot_sample_begin:tot_sample_end]         = residuals_tau_yv_upstream[:]
                    var_res_random_tau_yv_upstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_yv_upstream[:]
                    #
                    var_pred_tau_yv_downstream[tot_sample_begin:tot_sample_end]        = preds_tau_yv_downstream[:]
                    var_pred_random_tau_yv_downstream[tot_sample_begin:tot_sample_end] = preds_random_tau_yv_downstream[:]
                    var_lbl_tau_yv_downstream[tot_sample_begin:tot_sample_end]         = lbls_tau_yv_downstream[:]
                    var_res_tau_yv_downstream[tot_sample_begin:tot_sample_end]         = residuals_tau_yv_downstream[:]
                    var_res_random_tau_yv_downstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_yv_downstream[:]
                    #
                    var_pred_tau_zv_upstream[tot_sample_begin:tot_sample_end]        = preds_tau_zv_upstream[:]
                    var_pred_random_tau_zv_upstream[tot_sample_begin:tot_sample_end] = preds_random_tau_zv_upstream[:]
                    var_lbl_tau_zv_upstream[tot_sample_begin:tot_sample_end]         = lbls_tau_zv_upstream[:]
                    var_res_tau_zv_upstream[tot_sample_begin:tot_sample_end]         = residuals_tau_zv_upstream[:]
                    var_res_random_tau_zv_upstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_zv_upstream[:]
                    #
                    var_pred_tau_zv_downstream[tot_sample_begin:tot_sample_end]        = preds_tau_zv_downstream[:]
                    var_pred_random_tau_zv_downstream[tot_sample_begin:tot_sample_end] = preds_random_tau_zv_downstream[:]
                    var_lbl_tau_zv_downstream[tot_sample_begin:tot_sample_end]         = lbls_tau_zv_downstream[:]
                    var_res_tau_zv_downstream[tot_sample_begin:tot_sample_end]         = residuals_tau_zv_downstream[:]
                    var_res_random_tau_zv_downstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_zv_downstream[:]
                    #
                    var_pred_tau_xw_upstream[tot_sample_begin:tot_sample_end]        = preds_tau_xw_upstream[:]
                    var_pred_random_tau_xw_upstream[tot_sample_begin:tot_sample_end] = preds_random_tau_xw_upstream[:]
                    var_lbl_tau_xw_upstream[tot_sample_begin:tot_sample_end]         = lbls_tau_xw_upstream[:]
                    var_res_tau_xw_upstream[tot_sample_begin:tot_sample_end]         = residuals_tau_xw_upstream[:]
                    var_res_random_tau_xw_upstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_xw_upstream[:]
                    #
                    var_pred_tau_xw_downstream[tot_sample_begin:tot_sample_end]        = preds_tau_xw_downstream[:]
                    var_pred_random_tau_xw_downstream[tot_sample_begin:tot_sample_end] = preds_random_tau_xw_downstream[:]
                    var_lbl_tau_xw_downstream[tot_sample_begin:tot_sample_end]         = lbls_tau_xw_downstream[:]
                    var_res_tau_xw_downstream[tot_sample_begin:tot_sample_end]         = residuals_tau_xw_downstream[:]
                    var_res_random_tau_xw_downstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_xw_downstream[:]
                    #
                    var_pred_tau_yw_upstream[tot_sample_begin:tot_sample_end]        = preds_tau_yw_upstream[:]
                    var_pred_random_tau_yw_upstream[tot_sample_begin:tot_sample_end] = preds_random_tau_yw_upstream[:]
                    var_lbl_tau_yw_upstream[tot_sample_begin:tot_sample_end]         = lbls_tau_yw_upstream[:]
                    var_res_tau_yw_upstream[tot_sample_begin:tot_sample_end]         = residuals_tau_yw_upstream[:]
                    var_res_random_tau_yw_upstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_yw_upstream[:]
                    #
                    var_pred_tau_yw_downstream[tot_sample_begin:tot_sample_end]        = preds_tau_yw_downstream[:]
                    var_pred_random_tau_yw_downstream[tot_sample_begin:tot_sample_end] = preds_random_tau_yw_downstream[:]
                    var_lbl_tau_yw_downstream[tot_sample_begin:tot_sample_end]         = lbls_tau_yw_downstream[:]
                    var_res_tau_yw_downstream[tot_sample_begin:tot_sample_end]         = residuals_tau_yw_downstream[:]
                    var_res_random_tau_yw_downstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_yw_downstream[:]
                    #
                    var_pred_tau_zw_upstream[tot_sample_begin:tot_sample_end]        = preds_tau_zw_upstream[:]
                    var_pred_random_tau_zw_upstream[tot_sample_begin:tot_sample_end] = preds_random_tau_zw_upstream[:]
                    var_lbl_tau_zw_upstream[tot_sample_begin:tot_sample_end]         = lbls_tau_zw_upstream[:]
                    var_res_tau_zw_upstream[tot_sample_begin:tot_sample_end]         = residuals_tau_zw_upstream[:]
                    var_res_random_tau_zw_upstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_zw_upstream[:]
                    #
                    var_pred_tau_zw_downstream[tot_sample_begin:tot_sample_end]        = preds_tau_zw_downstream[:]
                    var_pred_random_tau_zw_downstream[tot_sample_begin:tot_sample_end] = preds_random_tau_zw_downstream[:]
                    var_lbl_tau_zw_downstream[tot_sample_begin:tot_sample_end]         = lbls_tau_zw_downstream[:]
                    var_res_tau_zw_downstream[tot_sample_begin:tot_sample_end]         = residuals_tau_zw_downstream[:]
                    var_res_random_tau_zw_downstream[tot_sample_begin:tot_sample_end]  = residuals_random_tau_zw_downstream[:]
                    #
                    vartstep[tot_sample_begin:tot_sample_end]        = tstep_samples[:]
                    varzhloc[tot_sample_begin:tot_sample_end]        = zhloc_samples[:]
                    varzloc[tot_sample_begin:tot_sample_end]         = zloc_samples[:]
                    varyhloc[tot_sample_begin:tot_sample_end]        = yhloc_samples[:]
                    varyloc[tot_sample_begin:tot_sample_end]         = yloc_samples[:]
                    varxhloc[tot_sample_begin:tot_sample_end]        = xhloc_samples[:]
                    varxloc[tot_sample_begin:tot_sample_end]         = xloc_samples[:]

                    tot_sample_begin = tot_sample_end #Make sure stored variables are not overwritten.

                except tf.errors.OutOfRangeError:
                    break #Break out of while-loop after one validation file (i.e. one flow 'snapshot'). NOTE: for this part of the code it is important that the eval_input_fn does not implement the .repeat() method on the created tf.Dataset.

if args.benchmark is None:
    predictions_file.close() #Close netCDF-file after each validation file
    print("Finished making predictions for each validation file.")
###

#Close all nc-files
training_file.close()
means_stdevs_file.close()
