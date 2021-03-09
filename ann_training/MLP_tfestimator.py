#Script that trains the MLP for turbulent channel flow case without distributed learning, making use of the tf.Estimator and tf.Dataset API.
#NOTE: uses TensorFlow v1, script not adapted for v>=2.0
import numpy as np
import netCDF4 as nc
import tensorflow as tf
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

#Number of cores on node
ncores = int(subprocess.check_output(["nproc", "--all"]))

# Instantiate the parser
parser = argparse.ArgumentParser(description='microhh_ML')
parser.add_argument('--checkpoint_dir', type=str, default='.',
                    help='directory where checkpoints are stored')
parser.add_argument('--input_dir', type=str, default='./',
                    help='directory where tfrecord files are located. The tfrecord files assumed to be located in subdirectories of input_dir that correspond to different resolutions.')
parser.add_argument('--stored_means_stdevs_filename', type=str, default='means_stdevs_allfields.nc', \
        help='name of netcdf-file with stored means and standard deviations of input and output variables, which should refer to a nc-file created as part of the training data. It is assumed that the training data associated with multiple resolutions is stored in multiple subdirectories, where each resolution/subdirectory has a separate nc-file with the stored means/stdevs with the provided name.')
parser.add_argument('--benchmark', dest='benchmark', default=None, \
        action='store_true', \
        help='Do a full run when benchmark is false, which includes producing and storing of preditions. Furthermore, in a full run more variables are stored to facilitate reconstruction of the corresponding transport fields. When the benchmark flag is true, the scripts ends immidiately after calculating the validation loss to facilitate benchmark tests.')
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
parser.add_argument('--batch_size', type=int, default=100, \
        help='Number of samples selected in each batch')
parser.add_argument('--profile_steps', type=int, default=10000, \
        help='Every nth step, a profile measurement is performed that is stored in a JSON-file.')
parser.add_argument('--summary_steps', type=int, default=100, \
        help='Every nth step, a summary is written for Tensorboard visualization')
parser.add_argument('--checkpoint_steps', type=int, default=10000, \
        help='Every nth step, a checkpoint of the model is written')
parser.add_argument('--train_dir1', type=str, default=None,
                    help='first directory used for training. If not specified, all directories present in input dir are used for training.')
parser.add_argument('--train_dir2', type=str, default=None,
                    help='second directory used for training. If not specified, all directories present in input dir are used for training, or, alternatively, the directory specified with --train_dir1.')
args = parser.parse_args()

#Define parse function for tfrecord files, which gives for each component in the example_proto 
#the output in format (dict(features),tensor(labels)).
def _parse_function(example_proto):

    if args.benchmark is None:
        keys_to_features = {
            'uc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
            'vc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
            'wc_sample':tf.FixedLenFeature([5*5*5],tf.float32),
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

    return parsed_features,labels


#Define training input function
def train_input_fn(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.shuffle(buffer_size=len(filenames)) #Reshuffle the tfrecord files every epoch, only possible when .cache() is commented out. NOTE:the (prior randomized) content of the files is not reshuffled, so it is not fully random from epoch to epoch.
    dataset = dataset.map(lambda line:_parse_function(line), num_parallel_calls=ncores) #Parallelize map transformation using the total amount of CPU cores available.
    #dataset = dataset.cache() #Put samples from tfrecord files directly in memory
    #dataset = dataset.shuffle(buffer_size=10000) #NOTE: shuffling operation commented out, which causes the samples to be in the same order every epoch. The shuffling would require a large buffer size because of the relatively large batches chosen (1000/3000), which would negatively impact the total involved computational effort (especially since each sample contains quite some data in our case). Instead, we 1) randomly shuffled the samples before storing them in tfrecord files (see sample_training_data_tfrecord.py), and 2) we randomized the order of the tfrecord-files before training (see below). Putting the shuffle in front of cache() results in memory saturation issues.
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.repeat()
    dataset.prefetch(1)

    return dataset

#Define evaluation function
def eval_input_fn(filenames, batch_size):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda line:_parse_function(line), num_parallel_calls=ncores)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset.prefetch(1)

    return dataset    

#Define function for splitting the training, validation, and test set
#NOTE: this split is on purpose not random. For the validation and test set always the flow fields corresponding to the last time steps are selected, such that the flow fields used for validation and testing are as independent as possible from the fields used for training.
def split_train_val_test(time_steps, val_ratio, test_ratio):
    if (val_ratio + test_ratio) >= 1.0:
        raise RuntimeError("The relative contributions of the validation and test set added up cannot exceed 1.")
    flipped_steps = np.flip(time_steps)
    test_set_size =  max(int(len(time_steps) * test_ratio),1) #max(..) makes sure that always at least 1 file is for testing
    val_set_size = max(int(len(time_steps) * val_ratio),1) #max(..) makes sure that always at least 1 file is selected for validation
    if (test_set_size + val_set_size) >= len(time_steps):
        raise RuntimeError("The test and validation set are equal or exceed the total number of time steps available in the training data, leaving no time steps for training. Reduce the relative contribution of the test and validation set appropriately and/or increase the number of time steps available in the training data.")
    test_steps = flipped_steps[:test_set_size]
    val_steps = flipped_steps[test_set_size:test_set_size+val_set_size]
    train_steps = flipped_steps[test_set_size+val_set_size:]
    return train_steps, val_steps, test_steps


#Define function that builds a separate MLP.
def create_MLP(inputs, name_MLP, params):
    '''Function to build a MLP with specified inputs. Inputs should be a list of tf.Tensors containing the individual variables.\\
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
    '''Model function which calls create_MLP multiple times to build MLPs that each predict some of the labels. These separate MLPs are trained together and combined in validation and inference mode. \\
            NOTE: this function accesses the global variables means_dict_avgt, and stdevs_dict_avgt.'''

    #Define tf.constants for storing the means and stdevs of the input variables & labels, which is needed for the normalisation and subsequent denormalisation in this graph
    #NOTE: the means and stdevs for the '_upstream' and '_downstream' labels are the same. This is why each mean and stdev is repeated twice.

    means_inputs = tf.constant([[
        means_dict_avgt['uc'],
        means_dict_avgt['vc'],
        means_dict_avgt['wc']]])
    
    stdevs_inputs = tf.constant([[
        stdevs_dict_avgt['uc'],
        stdevs_dict_avgt['vc'],
        stdevs_dict_avgt['wc']]])
    
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

    #Define identity ops for input variables, which is used to set-up a frozen graph that serves as a basis for inference.
    input_u      = tf.identity(features['uc_sample'], name = 'input_u')
    input_v      = tf.identity(features['vc_sample'], name = 'input_v')
    input_w      = tf.identity(features['wc_sample'], name = 'input_w')
    
    #Create summary to visualize inputs in TensorBoard
    tf.summary.histogram('input_u', input_u)
    tf.summary.histogram('input_v', input_v)
    tf.summary.histogram('input_w', input_w)

    #Define function to standardize input variables
    def _standardization(input_variable, mean_variable, stdev_variable):
        input_variable = tf.math.subtract(input_variable, mean_variable)
        input_variable = tf.math.divide(input_variable, stdev_variable)
        return input_variable

    #Standardize input variables
    with tf.name_scope("standardization_inputs"): #Group nodes in name scope for easier visualisation in TensorBoard
        input_u_stand  = _standardization(input_u, means_inputs[:,0], stdevs_inputs[:,0])
        input_v_stand  = _standardization(input_v, means_inputs[:,1], stdevs_inputs[:,1])
        input_w_stand  = _standardization(input_w, means_inputs[:,2], stdevs_inputs[:,2])
        
        #Create summaries to visualize standardized input values in TensorBoard
        tf.summary.histogram('input_u_stand', input_u_stand)
        tf.summary.histogram('input_v_stand', input_v_stand)
        tf.summary.histogram('input_w_stand', input_w_stand)

    #Standardize labels
    with tf.name_scope("standardization_labels"): #Group nodes in name scope for easier visualisation in TensorBoard
        labels_means = tf.math.subtract(labels, means_labels)
        labels_stand = tf.math.divide(labels_means, stdevs_labels, name = 'labels_stand')
    
    #Create mask to set some transport components to 0 at the bottom wall. For samples centered around the bottom wall, w is 0 at the center grid cell (but u and v are not because of the staggered grid orientation!). Consequently, the corresponding tendency of w is known to be 0 as well, removing the need to determine the individual transport components related to the w control volume (but, again, the others do still need to be determined!). Therefore, the transport components related to the w control volume are explicitly set to 0 during training, such that their influence is diminished. This allows the ANN to still learn the other transport components at the bottom wall that do have to be predicted.
    #NOTE1: because of the staggered grid orientation (where for a certain grid cell w is located at a lower height than u, v), a similar procedure is not needed at the top wall. For the samples closest to the top wall, in the center grid cell w is not directly located at the top wall. To determine all needed transport components, it is at the top wall not necessary to consider samples where in the center grid cell w is directly located at the top wall.
    #NOTE2: zu/zv_upstream/downstream are not set to 0 at both the bottom and top wall desite their location directly at the walls: this is because we included the unresolved viscous flux, on top of the unresolved turbulent flux, in the labels. This means that these components are not 0 anymore, despite that the no-slip BCs are valid for the unresolved turbulent flux contribution.
    #NOTE3: in inference mode, the masking does not have to be applied: within MicroHH, the transport components set to 0 during training are discarded in the tendency calculations. Hence, the masking is removed during inference to reduce the total computational effort.
    #NOTE4: zw_downstream is put to 0 at the bottom wall while it is located within the flow domain above the wall. To ensure symmetry in the predictions of tau_ww between the bottom and top wall, zw_downstream is in MicroHH NOT evaluated at the bottom wall. Instead, the zw_upstream from the second layer above the bottom wall is consistently used.
    if not mode == tf.estimator.ModeKeys.PREDICT:
        flag_bottomwall = tf.identity(features['flag_bottomwall_sample'], name = 'flag_bottomwall')
        with tf.name_scope("mask_creation"):
            flag_bottomwall_bool = tf.expand_dims(tf.math.not_equal(flag_bottomwall, 1), axis=1) #Select all samples that are not located at the bottom wall, and extend dim to be compatible with other arrays
        
            #Select all transport components that should NOT be set to 0.
            components_bottomwall_bool = tf.constant(
                    [[True,  True,  #xu_upstream, xu_downstream
                      True,  True,  #yu_upstream, yu_downstream
                      True,  True,  #zu_upstream, zu_downstream
                      True,  True,  #xv_upstream, xv_downstream
                      True,  True,  #yv_upstream, yv_downstream
                      True,  True,  #zv_upstream, zv_downstream
                      False, False, #xw_upstream, xw_downstream
                      False, False, #yw_upstream, yw_downstream
                      False, False]])#zw_upstream, zw_downstream
            
            mask = tf.cast(tf.math.logical_or(flag_bottomwall_bool, components_bottomwall_bool), tf.float32, name = 'mask_noslipBC') #Cast boolean to float for the multiplication below
      
        #Mask labels that should not be taken into account during training
        labels_mask = tf.math.multiply(labels_stand, mask, name = 'labels_masked')
    
    #Call create_MLP three times to construct 3 separate MLPs
    #NOTE1: the sizes of the input are adjusted to train symmetrically. In doing so, it is assumed that the original size of the input was 5*5*5 grid cells!!!
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
       'u', params)
    output_layer_v = create_MLP(
       [
           _adjust_sizeinput(input_u_stand, np.s_[:,:,:-1,1:]), 
           input_v_stand, 
           _adjust_sizeinput(input_w_stand, np.s_[:,1:,:-1,:])],
       'v', params)
    output_layer_w = create_MLP(
       [
           _adjust_sizeinput(input_u_stand, np.s_[:,:-1,:,1:]), 
           _adjust_sizeinput(input_v_stand, np.s_[:,:-1,1:,:]), 
           input_w_stand],
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

    #Denormalize the output fluxes for inference/prediction
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope("denormalisation_output"): #Group nodes in name scope for easier visualisation in TensorBoard
            output_stdevs      = tf.math.multiply(output_layer_tot, stdevs_labels) #On purpose the output layer without masking applied is selected, see comment before.
            output_denorm      = tf.math.add(output_stdevs, means_labels, name = 'output_layer_denorm')
    
    #Denormalize the labels to have access to them during inference (needed for read_CNNsmagpredictions.py)
    #NOTE1: this does not have to be included when incorporating the ANN in MicroHH
    if mode == tf.estimator.ModeKeys.PREDICT:
        with tf.name_scope("denormalisation_labels"):
            labels_stdevs = tf.math.multiply(labels_stand, stdevs_labels) #NOTE: on purpose labels_stand instead of labels_mask.
            labels_denorm  = tf.math.add(labels_stdevs, means_labels, name = 'labels_denorm')
        
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
    loss = tf.losses.mean_squared_error(labels_mask, output_layer_mask, weights=weights, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE) #Mean taken over batch
        
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

    assert mode == tf.estimator.ModeKeys.TRAIN

    #Create summary of log(loss) for visualization in TensorBoard
    log_loss_training = log10(loss)
    tf.summary.scalar('log_loss', log_loss_training)

    #Create training op.
    optimizer = tf.train.AdamOptimizer(params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    #Write all trainable variables to summaries for visualization in Tensorboard
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name,var)

    #Return tf.estimator.Estimatorspec for training mode
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

#Define settings
batch_size = int(args.batch_size)
num_steps = args.num_steps #Number of iterations/steps, i.e. number of batches times number of epochs
num_labels = 6 #Number of predicted transport components for each sub-MLP
files_per_snapshot = 53 #Number of tfrecord files per time snapshot, with current settings this is equal to 53

#Define filenames of tfrecords for training and validation
nt_total = 31 #Number of time steps that should be used for training/validation/testing, assuming that the number of the time step in the filenames ranges from 1 to nt_total without gaps.
time_numbers = np.arange(nt_total)
train_stepnumbers, val_stepnumbers, test_stepnumbers = split_train_val_test(time_numbers, 0.1, 0.1) #Set aside ~10% of files for validation and ~10% for testing (with current settings, this results in 3 independent time-snapshots each for validation and testing.)
#Extract training filenames for specified training dirs, or all subdirectories if none are specified
if args.train_dir1 is not None:
    train_filenames_dir1 = []
    val_filenames_dir1 = []
    for train_step in train_stepnumbers:
        train_filenames_dir1 += glob.glob(args.input_dir + args.train_dir1 + '/training_time_step_' + str(train_step+1) + '_of_' + str(nt_total) + '*numsamples_1000.tfrecords')
    for val_step in val_stepnumbers:
        val_filenames_dir1 += glob.glob(args.input_dir + args.train_dir1 + '/training_time_step_' + str(val_step+1) + '_of_' + str(nt_total) + '*numsamples_1000.tfrecords')

if args.train_dir2 is not None:
    train_filenames_dir2 = []
    val_filenames_dir2 = []
    for train_step in train_stepnumbers:
        train_filenames_dir2 += glob.glob(args.input_dir + args.train_dir2 + '/training_time_step_' + str(train_step+1) + '_of_' + str(nt_total) + '*numsamples_1000.tfrecords')
    for val_step in val_stepnumbers:
        val_filenames_dir2 += glob.glob(args.input_dir + args.train_dir2 + '/training_time_step_' + str(val_step+1) + '_of_' + str(nt_total) + '*numsamples_1000.tfrecords')

if (args.train_dir1 is None) and (args.train_dir2 is None):
    train_dirs = glob.glob(args.input_dir + '*')
    num_train_dirs = len(train_dirs)
    train_filenames_dirs = [[] for _ in range(num_train_dirs)]
    val_filenames_dirs = [[] for _ in range(num_train_dirs)] #Prevent aliasing by initializing the list of lists like this
    for n in range(num_train_dirs):
        for train_step in train_stepnumbers:
            train_filenames_dirs[n] += glob.glob(str(train_dirs[n]) + '/training_time_step_' + str(train_step+1) + '_of_' + str(nt_total) + '*numsamples_1000.tfrecords')
        for val_step in val_stepnumbers:
            val_filenames_dirs[n] += glob.glob(str(train_dirs[n]) + '/training_time_step_' + str(val_step+1) + '_of_' + str(nt_total) + '*numsamples_1000.tfrecords')
#Make final training file list; ensure different resolutions are equally selected to prevent biased training
train_filenames = []
val_filenames = []
if (args.train_dir1 is None) and (args.train_dir2 is None):
    #Determine maximum length of training/validation dirs
    num_train_dirs = len(train_dirs)
    max_len_train = len(train_filenames_dirs[0])
    max_len_val   = len(val_filenames_dirs[0])
    for n in range(1,num_train_dirs):
        if len(train_filenames_dirs[n]) > max_len_train:
            max_len_train = len(train_filenames_dirs[n])
        if len(val_filenames_dirs[n]) > max_len_val:
            max_len_val   = len(val_filenames_dirs[n])

    #Make list, repeat shorter lists to ensure all resolutions are equally represented in each batch
    for i in range(max_len_train):
        for n in range(num_train_dirs):
            train_filenames += [train_filenames_dirs[n][i % len(train_filenames_dirs[n])]]
    
    for i in range(max_len_val):
        for n in range(num_train_dirs):
            val_filenames += [val_filenames_dirs[n][i % len(val_filenames_dirs[n])]]


elif (args.train_dir1 is not None) and (args.train_dir2 is not None):
    #Determine maximum length of training/validation dirs
    len1_train = len(train_filenames_dir1)
    len2_train = len(train_filenames_dir2)
    max_len_train = max(len1_train,len2_train) 
    len1_val   = len(val_filenames_dir1)
    len2_val   = len(val_filenames_dir2)
    max_len_val = max(len1_val,len2_val) 

    #Make list, repeat shorter lists to ensure all resolutions are equally represented in each batch
    for i in range(max_len_train):
        train_filenames += [train_filenames_dir1[i % len1_train]]
        train_filenames += [train_filenames_dir2[i % len2_train]]
    for i in range(max_len_val):
        val_filenames += [val_filenames_dir1[i % len1_val]]
        val_filenames += [val_filenames_dir2[i % len2_val]]

elif args.train_dir1 is not None:
    train_filenames = train_filenames_dir1
    val_filenames   = val_filenames_dir1

elif args.train_dir2 is not None:
    train_filenames = train_filenames_dir2
    val_filenames   = val_filenames_dir2

#Extract test filenames for all subdirectories, such the ANN is tested for all incorporated resolutions in input_dir.
test_filenames = [] #NOTE: here, no need to ensure that all resolutions (subdirectories) are equally represented 
test_dirs = glob.glob(args.input_dir + '*')
num_test_dirs = len(test_dirs)
test_filenames_dirs = [[]] * num_test_dirs
for n in range(num_test_dirs):
    for test_step in test_stepnumbers:
        test_filenames += glob.glob(str(test_dirs[n]) + '/training_time_step_' + str(test_step+1) + '_of_' + str(nt_total) + '*numsamples_1000.tfrecords')

#Reduce number of validation files if above 2000, to keep computational effort feasible. This is reasonable, as each tfrecord is already randomly shuffled and, if present, the different resolutions are interchanged sequentially in val_filenames.
max_val_files = 2000
if len(val_filenames) > max_val_files:
    val_filenames = val_filenames[:max_val_files]

#Randomly shuffle filenames
#np.random.shuffle(train_filenames) #NOTE: NOT needed, samples were already shuffled before stored in tfrecord files. The current order ensures that each batch (if chosen to be an exact multiple of the number of training subdirectories/resolutions) has an equal share of every resolution used for training.

#Print filenames to check
np.set_printoptions(threshold=np.inf)
print("Training files:")
#print('\n'.join(train_filenames))
print(len(train_filenames))
print("Validation files:")
#print('\n'.join(val_filenames))
print(len(val_filenames))
print("Testing files:")
#print('\n'.join(test_filenames))
print(len(test_filenames))

#Extract means and stdevs for input variables (which is needed for the normalisation).
means_stdevs_filename = args.stored_means_stdevs_filename

if (args.train_dir1 is None) and (args.train_dir2 is None):
    iter_range = range(num_train_dirs)
    train_dirs = glob.glob(args.input_dir + '*') #Repeated for the sake of clarity of the code
elif (args.train_dir1 is not None) and (args.train_dir2 is not None):
    iter_range = range(2)
    train_dirs = [args.input_dir + args.train_dir1, args.input_dir + args.train_dir2]
elif (args.train_dir1 is not None):
    iter_range = range(1)
    train_dirs = [args.input_dir + args.train_dir1]
elif (args.train_dir2 is not None):
    iter_range = range(1)
    train_dirs = [args.input_dir + args.train_dir2]
else:
    raise RuntimeError("This condition should not have been reached. Check code for errors.")


#Initialize dictionary for calculation means+stdevs over resolutions
means_dict_t  = {}
stdevs_dict_t = {}
means_dict_t['uc']  = np.empty((nt_total, len(iter_range)))
means_dict_t['vc']  = np.empty((nt_total, len(iter_range)))
means_dict_t['wc']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['uc']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['vc']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['wc']  = np.empty((nt_total, len(iter_range)))
means_dict_t['unres_tau_xu_sample']  = np.empty((nt_total, len(iter_range))) 
stdevs_dict_t['unres_tau_xu_sample'] = np.empty((nt_total, len(iter_range)))
means_dict_t['unres_tau_yu_sample']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['unres_tau_yu_sample'] = np.empty((nt_total, len(iter_range)))
means_dict_t['unres_tau_zu_sample']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['unres_tau_zu_sample'] = np.empty((nt_total, len(iter_range)))
means_dict_t['unres_tau_xv_sample']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['unres_tau_xv_sample'] = np.empty((nt_total, len(iter_range)))
means_dict_t['unres_tau_yv_sample']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['unres_tau_yv_sample'] = np.empty((nt_total, len(iter_range)))
means_dict_t['unres_tau_zv_sample']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['unres_tau_zv_sample'] = np.empty((nt_total, len(iter_range)))
means_dict_t['unres_tau_xw_sample']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['unres_tau_xw_sample'] = np.empty((nt_total, len(iter_range)))
means_dict_t['unres_tau_yw_sample']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['unres_tau_yw_sample'] = np.empty((nt_total, len(iter_range)))
means_dict_t['unres_tau_zw_sample']  = np.empty((nt_total, len(iter_range)))
stdevs_dict_t['unres_tau_zw_sample'] = np.empty((nt_total, len(iter_range)))

    #
for n in iter_range:
    means_stdevs_file     = nc.Dataset(train_dirs[n] + '/' + means_stdevs_filename, 'r')
     
    means_dict_t['uc'][:,n] = np.array(means_stdevs_file['mean_uc'][:])
    means_dict_t['vc'][:,n] = np.array(means_stdevs_file['mean_vc'][:])
    means_dict_t['wc'][:,n] = np.array(means_stdevs_file['mean_wc'][:])
    
    stdevs_dict_t['uc'][:,n] = np.array(means_stdevs_file['stdev_uc'][:])
    stdevs_dict_t['vc'][:,n] = np.array(means_stdevs_file['stdev_vc'][:])
    stdevs_dict_t['wc'][:,n] = np.array(means_stdevs_file['stdev_wc'][:])
    
    #Extract mean & standard deviation labels
    means_dict_t['unres_tau_xu_sample'][:,n]  = np.array(means_stdevs_file['mean_unres_tau_xu_sample'][:])
    stdevs_dict_t['unres_tau_xu_sample'][:,n] = np.array(means_stdevs_file['stdev_unres_tau_xu_sample'][:])
    means_dict_t['unres_tau_yu_sample'][:,n]  = np.array(means_stdevs_file['mean_unres_tau_yu_sample'][:])
    stdevs_dict_t['unres_tau_yu_sample'][:,n] = np.array(means_stdevs_file['stdev_unres_tau_yu_sample'][:])
    means_dict_t['unres_tau_zu_sample'][:,n]  = np.array(means_stdevs_file['mean_unres_tau_zu_sample'][:])
    stdevs_dict_t['unres_tau_zu_sample'][:,n] = np.array(means_stdevs_file['stdev_unres_tau_zu_sample'][:])
    means_dict_t['unres_tau_xv_sample'][:,n]  = np.array(means_stdevs_file['mean_unres_tau_xv_sample'][:])
    stdevs_dict_t['unres_tau_xv_sample'][:,n] = np.array(means_stdevs_file['stdev_unres_tau_xv_sample'][:])
    means_dict_t['unres_tau_yv_sample'][:,n]  = np.array(means_stdevs_file['mean_unres_tau_yv_sample'][:])
    stdevs_dict_t['unres_tau_yv_sample'][:,n] = np.array(means_stdevs_file['stdev_unres_tau_yv_sample'][:])
    means_dict_t['unres_tau_zv_sample'][:,n]  = np.array(means_stdevs_file['mean_unres_tau_zv_sample'][:])
    stdevs_dict_t['unres_tau_zv_sample'][:,n] = np.array(means_stdevs_file['stdev_unres_tau_zv_sample'][:])
    means_dict_t['unres_tau_xw_sample'][:,n]  = np.array(means_stdevs_file['mean_unres_tau_xw_sample'][:])
    stdevs_dict_t['unres_tau_xw_sample'][:,n] = np.array(means_stdevs_file['stdev_unres_tau_xw_sample'][:])
    means_dict_t['unres_tau_yw_sample'][:,n]  = np.array(means_stdevs_file['mean_unres_tau_yw_sample'][:])
    stdevs_dict_t['unres_tau_yw_sample'][:,n] = np.array(means_stdevs_file['stdev_unres_tau_yw_sample'][:])
    means_dict_t['unres_tau_zw_sample'][:,n]  = np.array(means_stdevs_file['mean_unres_tau_zw_sample'][:])
    stdevs_dict_t['unres_tau_zw_sample'][:,n] = np.array(means_stdevs_file['stdev_unres_tau_zw_sample'][:])

    #Close nc-file
    means_stdevs_file.close()
    
#Take average over training time steps and resolutions
means_dict_avgt  = {}
stdevs_dict_avgt = {}
#
means_dict_avgt['uc'] = np.mean(means_dict_t['uc'][train_stepnumbers,:])
means_dict_avgt['vc'] = np.mean(means_dict_t['vc'][train_stepnumbers,:])
means_dict_avgt['wc'] = np.mean(means_dict_t['wc'][train_stepnumbers,:])
#
stdevs_dict_avgt['uc'] = np.mean(stdevs_dict_t['uc'][train_stepnumbers,:])
stdevs_dict_avgt['vc'] = np.mean(stdevs_dict_t['vc'][train_stepnumbers,:])
stdevs_dict_avgt['wc'] = np.mean(stdevs_dict_t['wc'][train_stepnumbers,:])
#
means_dict_avgt['unres_tau_xu_sample']    = np.mean(means_dict_t['unres_tau_xu_sample'][train_stepnumbers,:])
stdevs_dict_avgt['unres_tau_xu_sample']   = np.mean(stdevs_dict_t['unres_tau_xu_sample'][train_stepnumbers,:])
means_dict_avgt['unres_tau_yu_sample']    = np.mean(means_dict_t['unres_tau_yu_sample'][train_stepnumbers,:])
stdevs_dict_avgt['unres_tau_yu_sample']   = np.mean(stdevs_dict_t['unres_tau_yu_sample'][train_stepnumbers,:])
means_dict_avgt['unres_tau_zu_sample']    = np.mean(means_dict_t['unres_tau_zu_sample'][train_stepnumbers,:])
stdevs_dict_avgt['unres_tau_zu_sample']   = np.mean(stdevs_dict_t['unres_tau_zu_sample'][train_stepnumbers,:])
means_dict_avgt['unres_tau_xv_sample']    = np.mean(means_dict_t['unres_tau_xv_sample'][train_stepnumbers,:])
stdevs_dict_avgt['unres_tau_xv_sample']   = np.mean(stdevs_dict_t['unres_tau_xv_sample'][train_stepnumbers,:])
means_dict_avgt['unres_tau_yv_sample']    = np.mean(means_dict_t['unres_tau_yv_sample'][train_stepnumbers,:])
stdevs_dict_avgt['unres_tau_yv_sample']   = np.mean(stdevs_dict_t['unres_tau_yv_sample'][train_stepnumbers,:])
means_dict_avgt['unres_tau_zv_sample']    = np.mean(means_dict_t['unres_tau_zv_sample'][train_stepnumbers,:])
stdevs_dict_avgt['unres_tau_zv_sample']   = np.mean(stdevs_dict_t['unres_tau_zv_sample'][train_stepnumbers,:])
means_dict_avgt['unres_tau_xw_sample']    = np.mean(means_dict_t['unres_tau_xw_sample'][train_stepnumbers,:])
stdevs_dict_avgt['unres_tau_xw_sample']   = np.mean(stdevs_dict_t['unres_tau_xw_sample'][train_stepnumbers,:])
means_dict_avgt['unres_tau_yw_sample']    = np.mean(means_dict_t['unres_tau_yw_sample'][train_stepnumbers,:])
stdevs_dict_avgt['unres_tau_yw_sample']   = np.mean(stdevs_dict_t['unres_tau_yw_sample'][train_stepnumbers,:])
means_dict_avgt['unres_tau_zw_sample']    = np.mean(means_dict_t['unres_tau_zw_sample'][train_stepnumbers,:])
stdevs_dict_avgt['unres_tau_zw_sample']   = np.mean(stdevs_dict_t['unres_tau_zw_sample'][train_stepnumbers,:])

print('means avg: ', means_dict_avgt)
print('stdevs avg: ', stdevs_dict_avgt)

#Set configuration
config = tf.ConfigProto(log_device_placement=False)
config.intra_op_parallelism_threads = args.intra_op_parallelism_threads
config.inter_op_parallelism_threads = args.inter_op_parallelism_threads
os.environ['KMP_BLOCKTIME'] = str(0)
os.environ['KMP_SETTINGS'] = str(1)
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
os.environ['OMP_NUM_THREADS'] = str(args.intra_op_parallelism_threads)

#Set warmstart_dir to None to disable it
warmstart_dir = None

#Create RunConfig object to save check_point in the model_dir according to the specified schedule, and to define the session config
my_checkpointing_config = tf.estimator.RunConfig(model_dir=args.checkpoint_dir, tf_random_seed=None, save_summary_steps=args.summary_steps, save_checkpoints_steps=args.checkpoint_steps, save_checkpoints_secs = None,session_config=config,keep_checkpoint_max=None, keep_checkpoint_every_n_hours=10000, log_step_count_steps=10, train_distribute=None) #Provide tf.contrib.distribute.DistributionStrategy instance to train_distribute parameter for distributed training

#Define hyperparameters
kernelsize_conv1 = 5

hyperparams =  {
'n_dense1':args.n_hidden, #Neurons in hidden layer for each control volume
'activation_function':tf.nn.leaky_relu, #NOTE: Define new activation function based on tf.nn.leaky_relu with lambda to adjust the default value for alpha (0.2)
'kernel_initializer':tf.initializers.he_uniform(),
'learning_rate':0.0001
}
print("number of neurons in hidden layer: ", str(args.n_hidden))
print("Checkpoint directory: ", args.checkpoint_dir)

#Instantiate an Estimator with model defined by model_fn
MLP = tf.estimator.Estimator(model_fn = model_fn, config=my_checkpointing_config, params = hyperparams, model_dir=args.checkpoint_dir, warm_start_from = warmstart_dir)

profiler_hook = tf.train.ProfilerHook(save_steps = args.profile_steps, output_dir = args.checkpoint_dir) #Hook designed for storing runtime statistics in Chrome trace JSON-format, which can be used in conjuction with the other summaries stored during training in Tensorboard.

if args.debug:
    debug_hook = tf_debug.LocalCLIDebugHook()
    hooks = [profiler_hook, debug_hook]
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
#NOTE4: Regardless how the ANN is trained, it is tested separately for all resolutions/subdirectories. A netCDF-file is created for each resolution.

if args.benchmark is None:
    print('Inference mode started.')

    samples_per_tfrecord = 1000 #Hard-coded value used to ensure correct inference. Change if number of samples per tfrecord is changed.
    
    test_dirs = glob.glob(args.input_dir + '*') #Repeated for the sake of clarity of the code
    num_test_dirs = len(test_dirs)
    test_filenames = np.array(test_filenames) #Convert to numpy array, otherwhise the code below does not work.
    #loop over test dirs, do inference for each of them
    for n in range(num_test_dirs):

        #Extract filenames corresponding to test dir from test filenames
        name_test_dir = test_dirs[n][len(args.input_dir):]
        print('Name current test directory: ', name_test_dir)
        bool_test_dir = np.zeros(len(test_filenames), dtype=bool)
        for i in range(len(test_filenames)):
            if name_test_dir in test_filenames[i]:
                bool_test_dir[i] = True
            else:
                bool_test_dir[i] = False
        test_filenames_dir = test_filenames[bool_test_dir]
        #print('Selected files: ', test_filenames_dir)

        #Create netCDF-file to store predictions and labels
        filepath = args.checkpoint_dir + '/MLP_predictions_' + name_test_dir + '.nc'
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
        
        #Initialize variables for keeping track of iterations
        tot_sample_end = 0
        tot_sample_begin = 0
        
        #Intialize flag to store inference graph only once
        store_graph = True
        
        #Loop over test files to prevent memory overflow issues
        for test_filename in test_filenames_dir:
        
            tf.reset_default_graph() #Reset the graph for each tfrecord
        
            #Generate iterator to extract features and labels from tfrecords
            iterator = eval_input_fn([test_filename], samples_per_tfrecord).make_initializable_iterator() #All samples present in test_filenames are used for validation once.
        
            #Define operation to extract features and labels from iterator
            fes, lbls = iterator.get_next()
        
            #Define operation to generate predictions for extracted features and labels
            preds_op = model_fn(fes, lbls, \
                            tf.estimator.ModeKeys.PREDICT, hyperparams).predictions
        
            #Create saver MLP_model such that it can be restored in the tf.Session() below
            saver = tf.train.Saver()
            
            with tf.Session(config=config) as sess:
        
                #Restore MLP_model within tf.Session()
                ckpt  = tf.train.get_checkpoint_state(args.checkpoint_dir)
                saver.restore(sess, ckpt.model_checkpoint_path)
        
                #Store inference graph
                if store_graph:
                    tf.io.write_graph(sess.graph, args.checkpoint_dir, 'inference_graph.pbtxt', as_text = True)
                    store_graph = False

                #Initialize iterator
                sess.run(iterator.initializer)

                #Generate predictions for every batch in the tfrecord file
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
                        break #Break out of while-loop after one test file. NOTE: for this part of the code it is important that the eval_input_fn does not implement the .repeat() method on the created tf.Dataset.

        predictions_file.close() #Close netCDF-file after all test tfrecord files have been evaluated
        print("Finished making predictions for test dir ", name_test_dir)
###
