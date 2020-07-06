#Script to manually set-up a MLP, and subsequently do inference on flow snapshots stored in training file with possibly permutation errors introduced to assess the importance of individual features.
#Author: Robin Stoffer (robin.stoffer@wur.nl)
import argparse
#import tensorflow as tf
import numpy as np
import netCDF4 as nc
import time
from diff_manual_python import diff_U
from joblib import Parallel, delayed
#from diff_frozen_python import diff_U

class MLP:
    '''Class to manually build MLP and subsequently do inference. NOTE: should be completely equivalent to MLP defined in MLP2_estimator.py!!!'''

    def __init__(self,ndense, variables_filepath): #Specify number of neurons in dense layer when instantiating the MLP
        
        self.ndense             = ndense
        
        #Load all weights and other variables from text files created with extract_variables_graph function located in load_frozen_graph.py
        #NOTE: take transpose of weights to get the shapes required for manual implementation of the MLP
        self.means_inputs       = np.loadtxt(variables_filepath+'means_inputs.txt')
        self.stdevs_inputs      = np.loadtxt(variables_filepath+'stdevs_inputs.txt')
        self.means_labels       = np.loadtxt(variables_filepath+'means_labels.txt')
        self.stdevs_labels      = np.loadtxt(variables_filepath+'stdevs_labels.txt')
        self.MLPu_hidden_kernel = np.loadtxt(variables_filepath+'MLPu_hidden_kernel.txt').transpose()
        self.MLPu_hidden_bias   = np.loadtxt(variables_filepath+'MLPu_hidden_bias.txt')
        self.MLPu_hidden_alpha  = np.loadtxt(variables_filepath+'MLPu_hidden_alpha.txt')
        self.MLPu_output_kernel = np.loadtxt(variables_filepath+'MLPu_output_kernel.txt').transpose()
        self.MLPu_output_bias   = np.loadtxt(variables_filepath+'MLPu_output_bias.txt')
        self.MLPv_hidden_kernel = np.loadtxt(variables_filepath+'MLPv_hidden_kernel.txt').transpose()
        self.MLPv_hidden_bias   = np.loadtxt(variables_filepath+'MLPv_hidden_bias.txt')
        self.MLPv_hidden_alpha  = np.loadtxt(variables_filepath+'MLPv_hidden_alpha.txt')
        self.MLPv_output_kernel = np.loadtxt(variables_filepath+'MLPv_output_kernel.txt').transpose()
        self.MLPv_output_bias   = np.loadtxt(variables_filepath+'MLPv_output_bias.txt')
        self.MLPw_hidden_kernel = np.loadtxt(variables_filepath+'MLPw_hidden_kernel.txt').transpose()
        self.MLPw_hidden_bias   = np.loadtxt(variables_filepath+'MLPw_hidden_bias.txt')
        self.MLPw_hidden_alpha  = np.loadtxt(variables_filepath+'MLPw_hidden_alpha.txt')
        self.MLPw_output_kernel = np.loadtxt(variables_filepath+'MLPw_output_kernel.txt').transpose()
        self.MLPw_output_bias   = np.loadtxt(variables_filepath+'MLPw_output_bias.txt')
        self.utau_ref           = np.loadtxt(variables_filepath+'utau_ref.txt')
        self.output_denorm_utau2= np.loadtxt(variables_filepath+'output_denorm_utau2.txt')

#        self.iteration=0

    #Define private function to make input variables non-dimensionless and standardize them
    def _standardization(self, input_variable, mean_variable, stdev_variable, scaling_factor):
        input_variable = np.divide(input_variable, scaling_factor)
        input_variable = np.subtract(input_variable, mean_variable)
        input_variable = np.divide(input_variable, stdev_variable)
        return input_variable

    
    #Define private function to adjust input size
    def _adjust_sizeinput(self, input_variable, indices):
        reshaped_variable = np.reshape(input_variable, [-1,5,5,5])
        adjusted_size_variable = reshaped_variable[indices]
        zlen = adjusted_size_variable.shape[1]
        ylen = adjusted_size_variable.shape[2]
        xlen = adjusted_size_variable.shape[3]
        final_variable = np.reshape(adjusted_size_variable, [-1,zlen*ylen*xlen])
        return final_variable
    
    #Define private function that executes a separate MLP.
    def _single_MLP(self, inputs, hidden_kernel, hidden_bias, output_kernel, output_bias, alpha):
         '''Private function to execute a MLP with specified input. Inputs should be a list of numpy arrays containing the individual variables.'''
     
         #Make input layer
         input_layer = np.concatenate(inputs, axis=1).flatten()
         #print(input_layer.shape)
     
         #Execute hidden layer with Leaky Relu activation function
         hidden_neurons = np.dot(hidden_kernel, input_layer) + hidden_bias
         #print(hidden_neurons.shape)
         y1 = (hidden_neurons > 0) * hidden_neurons
         y2 = (hidden_neurons <= 0) * hidden_neurons * alpha
         hidden_activations = y1 + y2
         #print(hidden_activations.shape)
         
         #Execute output layer with no activation function
         output_activations = np.expand_dims(np.dot(output_kernel, hidden_activations) + output_bias, axis=0)
         #print(output_activations.shape)
         return output_activations

    def predict(self, input_u, input_v, input_w): #if zw_flag is True, only determine zw-components
        
        #Standardize input variables
        input_u_stand  = self._standardization(input_u, self.means_inputs[0], self.stdevs_inputs[0], self.utau_ref)
        input_v_stand  = self._standardization(input_v, self.means_inputs[1], self.stdevs_inputs[1], self.utau_ref)
        input_w_stand  = self._standardization(input_w, self.means_inputs[2], self.stdevs_inputs[2], self.utau_ref)

        #Execute three single MLPs
        output_layer_u = self._single_MLP([
            input_u_stand, 
            self._adjust_sizeinput(input_v_stand, np.s_[:,:,1:,:-1]),
            self._adjust_sizeinput(input_w_stand, np.s_[:,1:,:,:-1])], 
            self.MLPu_hidden_kernel, self.MLPu_hidden_bias, 
            self.MLPu_output_kernel, self.MLPu_output_bias, 
            self.MLPu_hidden_alpha)
        
        output_layer_v = self._single_MLP([
            self._adjust_sizeinput(input_u_stand, np.s_[:,:,:-1,1:]), 
            input_v_stand,
            self._adjust_sizeinput(input_w_stand, np.s_[:,1:,:-1,:])], 
            self.MLPv_hidden_kernel, self.MLPv_hidden_bias, 
            self.MLPv_output_kernel, self.MLPv_output_bias, 
            self.MLPv_hidden_alpha)
        
        output_layer_w = self._single_MLP([
            self._adjust_sizeinput(input_u_stand, np.s_[:,:-1,:,1:]), 
            self._adjust_sizeinput(input_v_stand, np.s_[:,:-1,1:,:]),
            input_w_stand], 
            self.MLPw_hidden_kernel, self.MLPw_hidden_bias, 
            self.MLPw_output_kernel, self.MLPw_output_bias, 
            self.MLPw_hidden_alpha)
        
        #Concatenate output layers
        output_layer_tot = np.concatenate([output_layer_u, output_layer_v, output_layer_w], axis=1)

        #Denormalize output layer
        output_stdevs = np.multiply(output_layer_tot, self.stdevs_labels)
        output_means  = np.add(output_stdevs, self.means_labels)
        output_denorm = np.multiply(output_means, self.output_denorm_utau2)
        
        return output_denorm

class Grid:
    '''Class to store information about grid.'''

    def __init__(self,coord_center,coord_edge,gc,end_ind_center,end_ind_edge):
        self.zc      = coord_center[0]
        self.yc      = coord_center[1]
        self.xc      = coord_center[2]
        self.ktot    = len(self.zc)
        self.jtot    = len(self.yc)
        self.itot    = len(self.xc)
        self.zhc     = coord_edge[0]
        self.yhc     = coord_edge[1]
        self.xhc     = coord_edge[2]
        self.zsize   = self.zhc[-1]
        self.ysize   = self.yhc[-1]
        self.xsize   = self.xhc[-1]
        self.kstart  = gc[0]
        self.jstart  = gc[1]
        self.istart  = gc[2]
        self.kend    = end_ind_center[0]
        self.jend    = end_ind_center[1]
        self.iend    = end_ind_center[2]
        self.khend   = end_ind_edge[0]
        self.jhend   = end_ind_edge[1]
        self.ihend   = end_ind_edge[2]
        self.kcells  = self.kend   + self.kstart
        self.khcells = self.khend  + self.kstart
        self.jcells  = self.jend   + self.jstart
        self.icells  = self.iend   + self.istart
        self.ijcells = self.icells * self.jcells


def __grid_loop(u, v, w, grid, MLP, b, time_step, permute, loss, flag_only_zu_upstream, flag_only_zu_downstream, k, ksample, jsample, isample):

    if permute:
        #Initialize losses
        loss_permu = 0.0
        loss_permv = 0.0
        loss_permw = 0.0                    

    for j in range(grid.jstart,grid.jend,1):
        for i in range(grid.istart,grid.iend,1):
            
            #Extract grid box flow fields
            input_u_val = np.expand_dims(u[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0) #Flatten and expand dims arrays for MLP
            #print(input_u_val)
            input_v_val = np.expand_dims(v[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0)
            #print(input_v_val)
            input_w_val = np.expand_dims(w[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0)
            #print(input_w_val)
            #raise RuntimeError("Stop run")
    
            if permute:
                
                #u-velocity
    
                #Randomly select other velocity in horizontal
                input_u_val[0,(b*2+1)*(b*2+1)*ksample+(b*2+1)*jsample+isample] = np.random.choice(u[k-b+ksample,:,:].flatten())
                
                #Execute MLP once for selected grid box
                resultu = MLP.predict(input_u_val, input_v_val, input_w_val)
                
                #NOTE1: compensate indices for lack of ghost cells
                #NOTE2: flatten 'result' matrix to have consistent shape for output arrays
                resultu = resultu.flatten()
                i_nogc = i - grid.istart
                j_nogc = j - grid.jstart
                k_nogc = k - grid.kstart
                
                #Calculate mean squared errors, add to loss function
                if flag_only_zu_upstream:
                    loss_permu += (unres_tau_zu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[4]) ** 2.
                elif flag_only_zu_downstream:
                    loss_permu += (unres_tau_zu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[5]) ** 2.
                else:
                    loss_permu += (unres_tau_xu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[0]) ** 2.
                    loss_permu += (unres_tau_xu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[1]) ** 2.
                    loss_permu += (unres_tau_yu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[2]) ** 2.
                    loss_permu += (unres_tau_yu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[3]) ** 2.
                    loss_permu += (unres_tau_zu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[4]) ** 2.
                    loss_permu += (unres_tau_zu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[5]) ** 2.
                    loss_permu += (unres_tau_xv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[6]) ** 2.
                    loss_permu += (unres_tau_xv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[7]) ** 2.
                    loss_permu += (unres_tau_yv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[8]) ** 2.
                    loss_permu += (unres_tau_yv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[9]) ** 2.
                    loss_permu += (unres_tau_zv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[10]) ** 2.
                    loss_permu += (unres_tau_zv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[11]) ** 2.
                    loss_permu += (unres_tau_xw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[12]) ** 2.
                    loss_permu += (unres_tau_xw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[13]) ** 2.
                    loss_permu += (unres_tau_yw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[14]) ** 2.
                    loss_permu += (unres_tau_yw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[15]) ** 2.
                    loss_permu += (unres_tau_zw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultu[16]) ** 2.
                    loss_permu += (unres_tau_zw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultu[17]) ** 2.
            
                #v-velocity
    
                #Recover u-velocity
                input_u_val = np.expand_dims(u[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0) #Flatten and expand dims arrays for MLP
                
                #Randomly select other velocity in horizontal
                input_v_val[0,(b*2+1)*(b*2+1)*ksample+(b*2+1)*jsample+isample] = np.random.choice(v[k-b+ksample,:,:].flatten())
                
                #Execute MLP once for selected grid box
                resultv = MLP.predict(input_u_val, input_v_val, input_w_val)
                
                #NOTE1: compensate indices for lack of ghost cells
                #NOTE2: flatten 'result' matrix to have consistent shape for output arrays
                resultv = resultv.flatten()
                i_nogc = i - grid.istart
                j_nogc = j - grid.jstart
                k_nogc = k - grid.kstart
                
                #Calculate mean squared errors, add to loss function
                if flag_only_zu_upstream:
                    loss_permv += (unres_tau_zu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[4]) ** 2.
                elif flag_only_zu_downstream:
                    loss_permv += (unres_tau_zu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[5]) ** 2.
                else:
                    loss_permv += (unres_tau_xu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[0]) ** 2.
                    loss_permv += (unres_tau_xu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[1]) ** 2.
                    loss_permv += (unres_tau_yu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[2]) ** 2.
                    loss_permv += (unres_tau_yu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[3]) ** 2.
                    loss_permv += (unres_tau_zu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[4]) ** 2.
                    loss_permv += (unres_tau_zu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[5]) ** 2.
                    loss_permv += (unres_tau_xv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[6]) ** 2.
                    loss_permv += (unres_tau_xv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[7]) ** 2.
                    loss_permv += (unres_tau_yv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[8]) ** 2.
                    loss_permv += (unres_tau_yv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[9]) ** 2.
                    loss_permv += (unres_tau_zv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[10]) ** 2.
                    loss_permv += (unres_tau_zv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[11]) ** 2.
                    loss_permv += (unres_tau_xw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[12]) ** 2.
                    loss_permv += (unres_tau_xw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[13]) ** 2.
                    loss_permv += (unres_tau_yw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[14]) ** 2.
                    loss_permv += (unres_tau_yw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[15]) ** 2.
                    loss_permv += (unres_tau_zw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultv[16]) ** 2.
                    loss_permv += (unres_tau_zw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultv[17]) ** 2.
            
                #w-velocity
                #Recover v-velocity
                input_v_val = np.expand_dims(v[k-b:k+b+1,j-b:j+b+1,i-b:i+b+1].flatten(), axis=0) #Flatten and expand dims arrays for MLP
    
                #Randomly select other velocity in horizontal
                input_w_val[0,(b*2+1)*(b*2+1)*ksample+(b*2+1)*jsample+isample] = np.random.choice(w[k-b+ksample,:,:].flatten())
                
                #Execute MLP once for selected grid box
                resultw = MLP.predict(input_u_val, input_v_val, input_w_val)
                
                #NOTE1: compensate indices for lack of ghost cells
                #NOTE2: flatten 'result' matrix to have consistent shape for output arrays
                resultw = resultw.flatten()
                i_nogc = i - grid.istart
                j_nogc = j - grid.jstart
                k_nogc = k - grid.kstart
                
                #Calculate mean squared errors, add to loss function
                if flag_only_zu_upstream:
                    loss_permw += (unres_tau_zu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[4]) ** 2.
                elif flag_only_zu_downstream:
                    loss_permw += (unres_tau_zu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[5]) ** 2.
                else:
                    loss_permw += (unres_tau_xu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[0]) ** 2.
                    loss_permw += (unres_tau_xu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[1]) ** 2.
                    loss_permw += (unres_tau_yu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[2]) ** 2.
                    loss_permw += (unres_tau_yu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[3]) ** 2.
                    loss_permw += (unres_tau_zu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[4]) ** 2.
                    loss_permw += (unres_tau_zu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[5]) ** 2.
                    loss_permw += (unres_tau_xv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[6]) ** 2.
                    loss_permw += (unres_tau_xv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[7]) ** 2.
                    loss_permw += (unres_tau_yv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[8]) ** 2.
                    loss_permw += (unres_tau_yv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[9]) ** 2.
                    loss_permw += (unres_tau_zv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[10]) ** 2.
                    loss_permw += (unres_tau_zv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[11]) ** 2.
                    loss_permw += (unres_tau_xw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[12]) ** 2.
                    loss_permw += (unres_tau_xw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[13]) ** 2.
                    loss_permw += (unres_tau_yw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[14]) ** 2.
                    loss_permw += (unres_tau_yw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[15]) ** 2.
                    loss_permw += (unres_tau_zw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - resultw[16]) ** 2.
                    loss_permw += (unres_tau_zw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - resultw[17]) ** 2.
            
            else:
    
                #Execute MLP once for selected grid box
                result = MLP.predict(input_u_val, input_v_val, input_w_val)
    
                #NOTE1: compensate indices for lack of ghost cells
                #NOTE2: flatten 'result' matrix to have consistent shape for output arrays
                result = result.flatten()
                i_nogc = i - grid.istart
                j_nogc = j - grid.jstart
                k_nogc = k - grid.kstart
                
                #Calculate mean squared errors, add to loss function
                if flag_only_zu_upstream:
                    loss += (unres_tau_zu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[4]) ** 2.
                elif flag_only_zu_downstream:
                    loss += (unres_tau_zu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[5]) ** 2.
                else:
                    loss += (unres_tau_xu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[0]) ** 2.
                    loss += (unres_tau_xu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[1]) ** 2.
                    loss += (unres_tau_yu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[2]) ** 2.
                    loss += (unres_tau_yu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[3]) ** 2.
                    loss += (unres_tau_zu_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[4]) ** 2.
                    loss += (unres_tau_zu_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[5]) ** 2.
                    loss += (unres_tau_xv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[6]) ** 2.
                    loss += (unres_tau_xv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[7]) ** 2.
                    loss += (unres_tau_yv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[8]) ** 2.
                    loss += (unres_tau_yv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[9]) ** 2.
                    loss += (unres_tau_zv_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[10]) ** 2.
                    loss += (unres_tau_zv_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[11]) ** 2.
                    loss += (unres_tau_xw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[12]) ** 2.
                    loss += (unres_tau_xw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[13]) ** 2.
                    loss += (unres_tau_yw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[14]) ** 2.
                    loss += (unres_tau_yw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[15]) ** 2.
                    loss += (unres_tau_zw_lbls_upstream[time_step,k_nogc,j_nogc,i_nogc]   - result[16]) ** 2.
                    loss += (unres_tau_zw_lbls_downstream[time_step,k_nogc,j_nogc,i_nogc] - result[17]) ** 2.
    
                #unres_tau_xu_CNN_upstream[k_nogc,j_nogc,i_nogc]   = result[0]
                #unres_tau_xu_CNN_downstream[k_nogc,j_nogc,i_nogc] = result[1]
                #unres_tau_yu_CNN_upstream[k_nogc,j_nogc,i_nogc]   = result[2]
                #unres_tau_yu_CNN_downstream[k_nogc,j_nogc,i_nogc] = result[3]
                #unres_tau_zu_CNN_upstream[k_nogc,j_nogc,i_nogc]   = result[4]
                #unres_tau_zu_CNN_downstream[k_nogc,j_nogc,i_nogc] = result[5]
                #unres_tau_xv_CNN_upstream[k_nogc,j_nogc,i_nogc]   = result[6]
                #unres_tau_xv_CNN_downstream[k_nogc,j_nogc,i_nogc] = result[7]
                #unres_tau_yv_CNN_upstream[k_nogc,j_nogc,i_nogc]   = result[8]
                #unres_tau_yv_CNN_downstream[k_nogc,j_nogc,i_nogc] = result[9]
                #unres_tau_zv_CNN_upstream[k_nogc,j_nogc,i_nogc]   = result[10]
                #unres_tau_zv_CNN_downstream[k_nogc,j_nogc,i_nogc] = result[11]
                #unres_tau_xw_CNN_upstream[k_nogc,j_nogc,i_nogc]   = result[12]
                #unres_tau_xw_CNN_downstream[k_nogc,j_nogc,i_nogc] = result[13]
                #unres_tau_yw_CNN_upstream[k_nogc,j_nogc,i_nogc]   = result[14]
                #unres_tau_yw_CNN_downstream[k_nogc,j_nogc,i_nogc] = result[15]
                #unres_tau_zw_CNN_upstream[k_nogc,j_nogc,i_nogc]   = result[16]
                #unres_tau_zw_CNN_downstream[k_nogc,j_nogc,i_nogc] = result[17]


    #Return featue importances or loss
    if permute:
        return loss_permu, loss_permv, loss_permw
    else:
        return loss

def inference_MLP(u, v, w, grid, MLP, b, time_step, permute, loss, flag_only_zu_upstream, flag_only_zu_downstream, flag_only_loglayer):
    #Makes currently use of global variables for the labels and storage of loss
    
    ##Initialize fields for storage transport components
    #unres_tau_xu_CNN_upstream = np.zeros((len(zc),len(yc),len(xc)), dtype = np.float64)
    #unres_tau_xv_CNN_upstream = np.zeros((len(zc),len(yhcless),len(xhcless)), dtype = np.float64)
    #unres_tau_xw_CNN_upstream = np.zeros((len(zhcless),len(yc),len(xhcless)), dtype = np.float64)
    #unres_tau_yu_CNN_upstream = np.zeros((len(zc),len(yhcless),len(xhcless)), dtype = np.float64)
    #unres_tau_yv_CNN_upstream = np.zeros((len(zc),len(yc),len(xc)), dtype = np.float64)
    #unres_tau_yw_CNN_upstream = np.zeros((len(zhcless),len(yhcless),len(xc)), dtype = np.float64)
    #unres_tau_zu_CNN_upstream = np.zeros((len(zhcless),len(yc),len(xhcless)), dtype = np.float64)
    #unres_tau_zv_CNN_upstream = np.zeros((len(zhcless),len(yhcless),len(xc)), dtype = np.float64)
    #unres_tau_zw_CNN_upstream = np.zeros((len(zc),len(yc),len(xc)), dtype = np.float64)
    ##
    #unres_tau_xu_CNN_downstream = np.zeros((len(zc),len(yc),len(xc)), dtype = np.float64)
    #unres_tau_xv_CNN_downstream = np.zeros((len(zc),len(yhcless),len(xhcless)), dtype = np.float64)
    #unres_tau_xw_CNN_downstream = np.zeros((len(zhcless),len(yc),len(xhcless)), dtype = np.float64)
    #unres_tau_yu_CNN_downstream = np.zeros((len(zc),len(yhcless),len(xhcless)), dtype = np.float64)
    #unres_tau_yv_CNN_downstream = np.zeros((len(zc),len(yc),len(xc)), dtype = np.float64)
    #unres_tau_yw_CNN_downstream = np.zeros((len(zhcless),len(yhcless),len(xc)), dtype = np.float64)
    #unres_tau_zu_CNN_downstream = np.zeros((len(zhcless),len(yc),len(xhcless)), dtype = np.float64)
    #unres_tau_zv_CNN_downstream = np.zeros((len(zhcless),len(yhcless),len(xc)), dtype = np.float64)
    #unres_tau_zw_CNN_downstream = np.zeros((len(zc),len(yc),len(xc)), dtype = np.float64)
    
    #Initialize empty arrays for temporary combined storage transport components
    #unres_tau_xu_CNN = np.full((len(zc),len(yc),len(xgcextra)), np.nan, dtype=np.float32)
    #unres_tau_xv_CNN = np.full((len(zc),len(yhcless),len(xhc)), np.nan, dtype=np.float32)
    #unres_tau_xw_CNN = np.full((len(zhcless),len(yc),len(xhc)), np.nan, dtype=np.float32)
    #unres_tau_yu_CNN = np.full((len(zc),len(yhc),len(xhcless)), np.nan, dtype=np.float32)
    #unres_tau_yv_CNN = np.full((len(zc),len(ygcextra),len(xc)), np.nan, dtype=np.float32)
    #unres_tau_yw_CNN = np.full((len(zhcless),len(yhc),len(xc)), np.nan, dtype=np.float32)
    #unres_tau_zu_CNN = np.full((len(zhc),len(yc),len(xhcless)), np.nan, dtype=np.float32)
    #unres_tau_zv_CNN = np.full((len(zhc),len(yhcless),len(xc)), np.nan, dtype=np.float32)
    #unres_tau_zw_CNN = np.full((len(zc),len(yc),len(xc)),       np.nan, dtype=np.float32)

    #Reshape 1d arrays to 3d, which is much more convenient for the slicing below.
    u = np.reshape(u, (grid.kcells,  grid.jcells, grid.icells))
    v = np.reshape(v, (grid.kcells,  grid.jcells, grid.icells))
    w = np.reshape(w, (grid.khcells, grid.jcells, grid.icells))

    #Initialize arrays for storage features importances
    if permute:
        ksample = 0
        jsample = 0
        isample = 0
        u_fi = np.zeros((5,5,5), dtype = np.float64)
        v_fi = np.zeros((5,5,5), dtype = np.float64)
        w_fi = np.zeros((5,5,5), dtype = np.float64)
        end_range = b*2+1
        #end_range = 1 #Testing purposes only!
    
    else:
        end_range = 1 #Effectively disables three outer loops

    #Loop over features
    with Parallel(n_jobs=-1, verbose=10) as parallel: #Retain pool of workers, avoid recreation
        for ksample in range(0,end_range):
            for jsample in range(0,end_range):
                for isample in range(0,end_range):

                    if flag_only_loglayer:
                        k = grid.kstart + 3 #Select grid height in log-layer, same as selected in publication
                        result = __grid_loop(u, v, w, grid, MLP, b, time_step, permute, loss, flag_only_zu_upstream, flag_only_zu_downstream, k, ksample, jsample, isample)
                    else:
                        result = parallel(delayed(__grid_loop)(u, v, w, grid, MLP, b, time_step, permute, loss, flag_only_zu_upstream, flag_only_zu_downstream, k, ksample, jsample, isample) for k in range(grid.kstart,grid.kend,1))
                    #for k in range(grid.kstart,grid.kstart+1,1): #FOR TESTING PURPOSES ONLY!!!
                    
                    if permute:
                        if not flag_only_loglayer:
                            loss_permu, loss_permv, loss_permw = zip(*result)
                            loss_permu = np.sum(loss_permu)
                            loss_permv = np.sum(loss_permv)
                            loss_permw = np.sum(loss_permw)
                        else:
                            loss_permu, loss_permv, loss_permw = result
                        #    
                        if (flag_only_zu_upstream or flag_only_zu_downstream) and flag_only_loglayer:
                            loss_permu = (loss_permu / (1. * 1. * grid.jtot * grid.itot)) ** 0.5
                            u_fi[ksample,jsample,isample] = loss_permu / loss[time_step]
                            loss_permv = (loss_permv / (1. * 1. * grid.jtot * grid.itot)) ** 0.5
                            v_fi[ksample,jsample,isample] = loss_permv / loss[time_step]
                            loss_permw = (loss_permw / (1. * 1. * grid.jtot * grid.itot)) ** 0.5
                            w_fi[ksample,jsample,isample] = loss_permw / loss[time_step]
                        elif flag_only_zu_upstream or flag_only_zu_downstream:
                            loss_permu = (loss_permu / (1. * grid.ktot * grid.jtot * grid.itot)) ** 0.5
                            u_fi[ksample,jsample,isample] = loss_permu / loss[time_step]
                            loss_permv = (loss_permv / (1. * grid.ktot * grid.jtot * grid.itot)) ** 0.5
                            v_fi[ksample,jsample,isample] = loss_permv / loss[time_step]
                            loss_permw = (loss_permw / (1. * grid.ktot * grid.jtot * grid.itot)) ** 0.5
                            w_fi[ksample,jsample,isample] = loss_permw / loss[time_step]
                        elif flag_only_loglayer:
                            loss_permu = (loss_permu / (18. * 1. * grid.jtot * grid.itot)) ** 0.5
                            u_fi[ksample,jsample,isample] = loss_permu / loss[time_step]
                            loss_permv = (loss_permv / (18. * 1. * grid.jtot * grid.itot)) ** 0.5
                            v_fi[ksample,jsample,isample] = loss_permv / loss[time_step]
                            loss_permw = (loss_permw / (18. * 1. * grid.jtot * grid.itot)) ** 0.5
                            w_fi[ksample,jsample,isample] = loss_permw / loss[time_step]
                        else:
                            loss_permu = (loss_permu / (18. * grid.ktot * grid.jtot * grid.itot)) ** 0.5
                            u_fi[ksample,jsample,isample] = loss_permu / loss[time_step]
                            loss_permv = (loss_permv / (18. * grid.ktot * grid.jtot * grid.itot)) ** 0.5
                            v_fi[ksample,jsample,isample] = loss_permv / loss[time_step]
                            loss_permw = (loss_permw / (18. * grid.ktot * grid.jtot * grid.itot)) ** 0.5
                            w_fi[ksample,jsample,isample] = loss_permw / loss[time_step]
                    else:
                        loss = np.sum(result)
                        #Take average loss function, take root, and store it in nc-file
                        if (flag_only_zu_upstream or flag_only_zu_downstream) and flag_only_loglayer:
                            loss = (loss / (1. * 1. * grid.jtot * grid.itot)) ** 0.5
                        elif flag_only_zu_upstream or flag_only_zu_downstream:
                            loss = (loss / (1. * grid.ktot * grid.jtot * grid.itot)) ** 0.5
                        elif flag_only_loglayer:
                            loss = (loss / (18. * 1. * grid.jtot * grid.itot)) ** 0.5
                        else:
                            loss = (loss / (18. * grid.ktot * grid.jtot * grid.itot)) ** 0.5

    #Return featue importances or loss
    if permute:
        return u_fi, v_fi, w_fi
    else:
        return loss

    #var_unres_tau_xu_CNN_upstream[time_step,:,:,:] =  unres_tau_xu_CNN_upstream[:,:,:] 
    #var_unres_tau_yu_CNN_upstream[time_step,:,:,:] =  unres_tau_yu_CNN_upstream[:,:,:] 
    #var_unres_tau_zu_CNN_upstream[time_step,:,:,:] =  unres_tau_zu_CNN_upstream[:,:,:] 
    #var_unres_tau_xv_CNN_upstream[time_step,:,:,:] =  unres_tau_xv_CNN_upstream[:,:,:] 
    #var_unres_tau_yv_CNN_upstream[time_step,:,:,:] =  unres_tau_yv_CNN_upstream[:,:,:] 
    #var_unres_tau_zv_CNN_upstream[time_step,:,:,:] =  unres_tau_zv_CNN_upstream[:,:,:] 
    #var_unres_tau_xw_CNN_upstream[time_step,:,:,:] =  unres_tau_xw_CNN_upstream[:,:,:] 
    #var_unres_tau_yw_CNN_upstream[time_step,:,:,:] =  unres_tau_yw_CNN_upstream[:,:,:] 
    #var_unres_tau_zw_CNN_upstream[time_step,:,:,:] =  unres_tau_zw_CNN_upstream[:,:,:] 
    ##
    #var_unres_tau_xu_CNN_downstream[time_step,:,:,:] =  unres_tau_xu_CNN_downstream[:,:,:] 
    #var_unres_tau_yu_CNN_downstream[time_step,:,:,:] =  unres_tau_yu_CNN_downstream[:,:,:] 
    #var_unres_tau_zu_CNN_downstream[time_step,:,:,:] =  unres_tau_zu_CNN_downstream[:,:,:] 
    #var_unres_tau_xv_CNN_downstream[time_step,:,:,:] =  unres_tau_xv_CNN_downstream[:,:,:] 
    #var_unres_tau_yv_CNN_downstream[time_step,:,:,:] =  unres_tau_yv_CNN_downstream[:,:,:] 
    #var_unres_tau_zv_CNN_downstream[time_step,:,:,:] =  unres_tau_zv_CNN_downstream[:,:,:] 
    #var_unres_tau_xw_CNN_downstream[time_step,:,:,:] =  unres_tau_xw_CNN_downstream[:,:,:] 
    #var_unres_tau_yw_CNN_downstream[time_step,:,:,:] =  unres_tau_yw_CNN_downstream[:,:,:] 
    #var_unres_tau_zw_CNN_downstream[time_step,:,:,:] =  unres_tau_zw_CNN_downstream[:,:,:] 

if __name__ == '__main__':
    #Parse input
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_filename", default="training_data.nc")
    parser.add_argument("--loss_filename", default="loss.nc")
    parser.add_argument("--inference_filename", default="inference_reconstructed_fields.nc")
    parser.add_argument("--variables_filepath", default="", help="filepath where extracted variables from the frozen graph are located.")
    #parser.add_argument("--labels_filename", default="reconstructed_fields.nc", help="file where labels are stored")
    parser.add_argument("--only_zu_upstream", dest='only_zu_upstream', action = "store_true", help= "Specify when only zu_upstream component should be considered")
    parser.add_argument("--only_zu_downstream", dest='only_zu_downstream', action = "store_true", help= "Specify when only zu_downstream component should be considered")
    parser.add_argument("--only_loglayer", dest='only_loglayer', action = "store_true", help= "Specify when only log-layer should be considered")
    parser.add_argument("--calc_loss", dest='calc_loss', action = "store_true", help= "Specify when inference file does not yet contain the loss.")
    #parser.add_argument("--batch_size", default=1000)
    args = parser.parse_args()
    #batch_size = int(args.batch_size)

    #Determine flags
    if args.only_zu_upstream and args.only_zu_downstream:
        raise RuntimeError("Not both only_zu_upstream/downstream flags can be specified at the same time.")
    elif args.only_zu_upstream:
        flag_only_zu_upstream = True
        flag_only_zu_downstream = False
    elif args.only_zu_downstream:
        flag_only_zu_upstream = False
        flag_only_zu_downstream = True
    else:
        flag_only_zu_upstream = False
        flag_only_zu_downstream = False

    if args.only_loglayer:
        flag_only_loglayer = True
    else:
        flag_only_loglayer = False

    ###Extract flow fields and from netCDF file###
    #Specify time steps NOTE: SHOULD BE 27 TO 30 to access validation fields (30 to 99 for test fields), CHECK WHETHER THIS IS STILL CONSISTENT!
    #NOTE: flow fields were normalized stored, UNDO first normalisation!
    tstart = 30
    tend   = 99
    tstep_unique = np.arange(tstart, tend)
    nt = tend - tstart
    #
    flowfields = nc.Dataset(args.training_filename)
    #labels     = nc.Dataset(args.labels_filename)
    utau_ref_channel = np.array(flowfields['utau_ref'][:],dtype='f4')
    u = np.array(flowfields['uc'][tstart:tend,:,:,:]) * utau_ref_channel
    v = np.array(flowfields['vc'][tstart:tend,:,:,:]) * utau_ref_channel
    w = np.array(flowfields['wc'][tstart:tend,:,:,:]) * utau_ref_channel
    #
    unres_tau_xu_lbls_upstream = np.array(flowfields['unres_tau_xu_tot'][tstart:tend,:,:,:-1])   * (utau_ref_channel ** 2.) 
    unres_tau_yu_lbls_upstream = np.array(flowfields['unres_tau_yu_tot'])[tstart:tend,:,:-1,:-1] * (utau_ref_channel ** 2.)
    unres_tau_zu_lbls_upstream = np.array(flowfields['unres_tau_zu_tot'])[tstart:tend,:-1,:,:-1] * (utau_ref_channel ** 2.)
    unres_tau_xv_lbls_upstream = np.array(flowfields['unres_tau_xv_tot'])[tstart:tend,:,:-1,:-1] * (utau_ref_channel ** 2.)
    unres_tau_yv_lbls_upstream = np.array(flowfields['unres_tau_yv_tot'])[tstart:tend,:,:-1,:]   * (utau_ref_channel ** 2.)
    unres_tau_zv_lbls_upstream = np.array(flowfields['unres_tau_zv_tot'])[tstart:tend,:-1,:-1,:] * (utau_ref_channel ** 2.)
    unres_tau_xw_lbls_upstream = np.array(flowfields['unres_tau_xw_tot'])[tstart:tend,:-1,:,:-1] * (utau_ref_channel ** 2.)
    unres_tau_yw_lbls_upstream = np.array(flowfields['unres_tau_yw_tot'])[tstart:tend,:-1,:-1,:] * (utau_ref_channel ** 2.)
    unres_tau_zw_lbls_upstream = np.array(flowfields['unres_tau_zw_tot'])[tstart:tend,:-1,:,:]   * (utau_ref_channel ** 2.)
    unres_tau_xu_lbls_downstream = np.array(flowfields['unres_tau_xu_tot'])[tstart:tend,:,:,1:]  * (utau_ref_channel ** 2.)
    unres_tau_yu_lbls_downstream = np.array(flowfields['unres_tau_yu_tot'])[tstart:tend,:,1:,:-1]* (utau_ref_channel ** 2.)
    unres_tau_zu_lbls_downstream = np.array(flowfields['unres_tau_zu_tot'])[tstart:tend,1:,:,:-1]* (utau_ref_channel ** 2.)
    unres_tau_xv_lbls_downstream = np.array(flowfields['unres_tau_xv_tot'])[tstart:tend,:,:-1,1:]* (utau_ref_channel ** 2.)
    unres_tau_yv_lbls_downstream = np.array(flowfields['unres_tau_yv_tot'])[tstart:tend,:,1:,:]  * (utau_ref_channel ** 2.)
    unres_tau_zv_lbls_downstream = np.array(flowfields['unres_tau_zv_tot'])[tstart:tend,1:,:-1,:]* (utau_ref_channel ** 2.)
    unres_tau_xw_lbls_downstream = np.array(flowfields['unres_tau_xw_tot'])[tstart:tend,:-1,:,1:]* (utau_ref_channel ** 2.)
    unres_tau_yw_lbls_downstream = np.array(flowfields['unres_tau_yw_tot'])[tstart:tend,:-1,1:,:]* (utau_ref_channel ** 2.)
    unres_tau_zw_lbls_downstream = np.array(flowfields['unres_tau_zw_tot'])[tstart:tend,1:,:,:]  * (utau_ref_channel ** 2.)

    #Extract coordinates, shape fields, and ghost cells
    zc       = np.array(flowfields['zc'][:])
    zgc      = np.array(flowfields['zgc'][:])
    nz       = len(zc)
    zhc      = np.array(flowfields['zhc'][:])
    zgcextra = np.array(flowfields['zgcextra'][:])
    yc       = np.array(flowfields['yc'][:])
    ny       = len(yc)
    yhc      = np.array(flowfields['yhc'][:])
    ygcextra = np.array(flowfields['ygcextra'][:])
    xc       = np.array(flowfields['xc'][:])
    nx       = len(xc)
    xhc      = np.array(flowfields['xhc'][:])
    xgcextra = np.array(flowfields['xgcextra'][:])
    zhcless  = zhc[:-1]
    yhcless  = yhc[:-1]
    xhcless  = xhc[:-1]
    igc      = int(flowfields['igc'][:])
    jgc      = int(flowfields['jgc'][:])
    kgc      = int(flowfields['kgc_center'][:])
    iend     = int(flowfields['iend'][:])
    ihend    = int(flowfields['ihend'][:])
    jend     = int(flowfields['jend'][:])
    jhend    = int(flowfields['jhend'][:])
    kend     = int(flowfields['kend'][:])
    khend    = int(flowfields['khend'][:])
    #

    #Store grid information in a class object called grid.
    grid = Grid(coord_center = (zc,yc,xc), coord_edge = (zhc,yhc,xhc), gc = (kgc,jgc,igc), end_ind_center = (kend,jend,iend), end_ind_edge = (khend,jhend,ihend))
    
    if not args.calc_loss:

        ###Create file for inference results###
        inference = nc.Dataset(args.inference_filename, 'w')

        #Create dimensions for storage in nc-file
        inference.createDimension("zc", len(zc))
        inference.createDimension("zgcextra", len(zgcextra))
        inference.createDimension("zhc",len(zhc))
        inference.createDimension("zhcless",len(zhcless))
        inference.createDimension("yc", len(yc))
        inference.createDimension("ygcextra", len(ygcextra))
        inference.createDimension("yhc",len(yhc))
        inference.createDimension("yhcless",len(yhcless))
        inference.createDimension("xc", len(xc))
        inference.createDimension("xgcextra", len(xgcextra))
        inference.createDimension("xhc",len(xhc))
        inference.createDimension("xhcless",len(xhcless))
        inference.createDimension("time",None)
        inference.createDimension("k",5)
        inference.createDimension("j",5)
        inference.createDimension("i",5)

        #Create variables for dimensions and store them
        var_zc           = inference.createVariable("zc",           "f8", ("zc",))
        var_zgcextra     = inference.createVariable("zgcextra",     "f8", ("zgcextra",))
        var_zhc          = inference.createVariable("zhc",          "f8", ("zhc",))
        var_zhcless      = inference.createVariable("zhcless",      "f8", ("zhcless",))
        var_yc           = inference.createVariable("yc",           "f8", ("yc",))
        var_ygcextra     = inference.createVariable("ygcextra",     "f8", ("ygcextra",))
        var_yhc          = inference.createVariable("yhc",          "f8", ("yhc",))
        var_yhcless      = inference.createVariable("yhcless",      "f8", ("yhcless",))
        var_xc           = inference.createVariable("xc",           "f8", ("xc",))
        var_xgcextra     = inference.createVariable("xgcextra",     "f8", ("xgcextra",))
        var_xhc          = inference.createVariable("xhc",          "f8", ("xhc",))
        var_xhcless      = inference.createVariable("xhcless",      "f8", ("xhcless",))
        var_k            = inference.createVariable("k",            "f8", ("k",))
        var_j            = inference.createVariable("j",            "f8", ("j",))
        var_i            = inference.createVariable("i",            "f8", ("i",))

        var_zc[:]            = zc
        var_zgcextra[:]      = zgcextra
        var_zhc[:]           = zhc
        var_zhcless[:]       = zhcless
        var_yc[:]            = yc
        var_ygcextra[:]      = ygcextra
        var_yhc[:]           = yhc
        var_yhcless[:]       = yhcless
        var_xc[:]            = xc
        var_xgcextra[:]      = xgcextra
        var_xhc[:]           = xhc
        var_xhcless[:]       = xhcless
        var_k[:]             = np.arange(5)
        var_j[:]             = np.arange(5)
        var_i[:]             = np.arange(5)
        
        #Initialize variables for storage inference results
        var_u_fi                      = inference.createVariable("u_fi","f8",("time","k","j","i"))
        var_v_fi                      = inference.createVariable("v_fi","f8",("time","k","j","i"))
        var_w_fi                      = inference.createVariable("w_fi","f8",("time","k","j","i"))
    ##
    #var_unres_tau_xu_CNN_upstream = inference.createVariable("unres_tau_xu_CNN_upstream","f8",("tstep_unique","zc","yc","xc"))      
    #var_unres_tau_xv_CNN_upstream = inference.createVariable("unres_tau_xv_CNN_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))     
    #var_unres_tau_xw_CNN_upstream = inference.createVariable("unres_tau_xw_CNN_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))     
    #var_unres_tau_yu_CNN_upstream = inference.createVariable("unres_tau_yu_CNN_upstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_unres_tau_yv_CNN_upstream = inference.createVariable("unres_tau_yv_CNN_upstream","f8",("tstep_unique","zc","yc","xc"))
    #var_unres_tau_yw_CNN_upstream = inference.createVariable("unres_tau_yw_CNN_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    #var_unres_tau_zu_CNN_upstream = inference.createVariable("unres_tau_zu_CNN_upstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    #var_unres_tau_zv_CNN_upstream = inference.createVariable("unres_tau_zv_CNN_upstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    #var_unres_tau_zw_CNN_upstream = inference.createVariable("unres_tau_zw_CNN_upstream","f8",("tstep_unique","zc","yc","xc"))
    ##
    #var_unres_tau_xu_CNN_downstream = inference.createVariable("unres_tau_xu_CNN_downstream","f8",("tstep_unique","zc","yc","xc")) 
    #var_unres_tau_xv_CNN_downstream = inference.createVariable("unres_tau_xv_CNN_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))     
    #var_unres_tau_xw_CNN_downstream = inference.createVariable("unres_tau_xw_CNN_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))     
    #var_unres_tau_yu_CNN_downstream = inference.createVariable("unres_tau_yu_CNN_downstream","f8",("tstep_unique","zc","yhcless","xhcless"))
    #var_unres_tau_yv_CNN_downstream = inference.createVariable("unres_tau_yv_CNN_downstream","f8",("tstep_unique","zc","yc","xc"))
    #var_unres_tau_yw_CNN_downstream = inference.createVariable("unres_tau_yw_CNN_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    #var_unres_tau_zu_CNN_downstream = inference.createVariable("unres_tau_zu_CNN_downstream","f8",("tstep_unique","zhcless","yc","xhcless"))
    #var_unres_tau_zv_CNN_downstream = inference.createVariable("unres_tau_zv_CNN_downstream","f8",("tstep_unique","zhcless","yhcless","xc"))
    #var_unres_tau_zw_CNN_downstream = inference.createVariable("unres_tau_zw_CNN_downstream","f8",("tstep_unique","zc","yc","xc"))
    #

    #Store loss in separate file if specified
    if args.calc_loss:
        loss_file = nc.Dataset(args.loss_filename, 'w')
        loss_file.createDimension("tstep_unique",len(tstep_unique))
        var_tstep_unique    = loss_file.createVariable("tstep_unique", "f8", ("tstep_unique",))
        var_tstep_unique[:] = tstep_unique
        var_loss            = loss_file.createVariable("loss","f8",("tstep_unique",))
        loss                = 0.0
    
    else:
        loss_file = nc.Dataset(args.loss_filename, 'r')
        loss = np.array(loss_file['loss'])

    #Instantiate manual MLP class for making predictions
    MLP = MLP(ndense = 512, variables_filepath = args.variables_filepath)
    
    #Loop over flow fields, for each time step in tstep_unique (giving 4 loops in total).
    #For each alternating grid cell, store transport components by calling the 'frozen' MLP within a tf.Session().
    for t in range(nt):
    #for t in range(1):

        #Select flow fields of time step
        u_singletimestep = u[t,:,:,:-1].flatten()#Flatten and remove ghost cells in horizontal staggered dimensions to make shape consistent to arrays in MicroHH
        v_singletimestep = v[t,:,:-1,:].flatten()
        w_singletimestep = w[t,:,:,:].flatten()
       
        #Define loop indices
        #ii = 1
        #jj = icells
        #kk = ijcells
        blocksize = 5 #size of block used as input for MLP
        b = blocksize // 2 

        #Determine normal or permuted loss
        #NOTE: assignment of unresolved transports happens as a side-effect within the function below!
        if args.calc_loss:
            loss = inference_MLP(u = u_singletimestep, v = v_singletimestep, w = w_singletimestep, grid = grid, MLP = MLP, b = b, time_step = t, permute = False, loss = loss, flag_only_zu_upstream = flag_only_zu_upstream, flag_only_zu_downstream = flag_only_zu_downstream, flag_only_loglayer = flag_only_loglayer) #Call for scripts based on manual implementation MLP
            var_loss[t] = loss
            loss = 0.0 #Re-initialize for next time step
        else:
            u_fi, v_fi, w_fi = inference_MLP(u = u_singletimestep, v = v_singletimestep, w = w_singletimestep, grid = grid, MLP = MLP, b = b, time_step = t, permute = True, loss = loss, flag_only_zu_upstream = flag_only_zu_upstream, flag_only_zu_downstream = flag_only_zu_downstream, flag_only_loglayer = flag_only_loglayer) #Call for scripts based on manual implementation MLP
            var_u_fi[t,:,:,:] = u_fi
            var_v_fi[t,:,:,:] = v_fi
            var_w_fi[t,:,:,:] = w_fi

        t1_end = time.perf_counter()

        #Determine importance feature by permuting inputs 

    #Close loss file
    loss_file.close()

    if not args.calc_loss:
        #Close inference file
        inference.close()

    #Close flow fields file
    flowfields.close()
