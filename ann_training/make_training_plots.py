#Read Tensorflow log file and make training plots
#Taken from: https://gist.github.com/tomrunia/1e1d383fb21841e8f144
#Adapted by: Robin Stoffer (robin.stoffer@wur.nl)
#Date of creation: 11-6-2019
import warnings
import glob
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
mpl.use('Agg') #Use 'Agg' backend, Tk backend not supported on cartesius
mpl.rcParams.update({'figure.autolayout':True})
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

def plot_tensorflow_log(log_files_training, log_files_validation, n_hidden_list, save_path):
    ''' Function to load Tensorflow Event files and make plots to visualize the training.
    log_files_training:   list of strings specifying location of Tensorflow Event file for training.
    log_files_validation: list of strings specifying location of Tensorflow Event file for validation.
    n_hidden_list: list of integers specifying for each MLP stored in log_files_training/validation how many neurons are present in the hidden layer.
    save_path: string specifying path to store generated plots.'''

    # Loading too much data is slow...
    tf_size_guidance = {
            'COMPRESSED_HISTOGRAMS':1,
            'IMAGES':1,
            'AUDIO':1,
            'SCALARS':0, #0 means all values are loaded
            'HISTOGRAMS':1,
            'TENSORS':1,
            }
    
    #Set boolean variables to ensure plots are made correctly
    first_iteration = True
    last_iteration = False

    #List with plot colors, should be longer, or have the same length, as the list with Event files
    colorspec = pl.cm.jet(np.linspace(0,1,10));

    #Start loop over numbered MLPs
    for j in range(len(log_files_training)):
    
        #Set boolean variable to True when last iteration starts
        if j == (len(log_files_training)-1):
            last_iteration = True

        event_acc_training   = EventAccumulator(log_files_training[j], tf_size_guidance)
        event_acc_training.Reload()
        event_acc_validation = EventAccumulator(log_files_validation[j], tf_size_guidance)
        event_acc_validation.Reload()

        # Show all tags in the log file
        print("Tags present in training file: ")
        print(event_acc_training.Tags())
        print("Tags present in validation file: ")
        print(event_acc_validation.Tags())

        training_loss     = event_acc_training.Scalars('log_loss')
        validation_loss   = event_acc_validation.Scalars('log_loss')
        validation_rmse   = event_acc_validation.Scalars('rmse')
        train_steps       = len(training_loss)
        validation_steps  = len(validation_loss)
        print("Training steps: ", train_steps)
        print("Validation steps: ",validation_steps)
        #NOTE: step size of training loss will be changed to ensure only training steps stored at same iterations as validation errors, making the resulting plot less crowded.
        x_train           = np.arange(0,train_steps, int(train_steps / validation_steps), dtype='f') #NOTE: specified as floats to prevent rounding off issues
        #x_train           = x_train / int(train_steps / validation_steps)
        x_validation      = np.arange(validation_steps, dtype='f') 
        y_train           = np.arange(0,train_steps, int(train_steps / validation_steps), dtype='f')
        #y_train           = y_train / int(train_steps / validation_steps)
        y_validation_loss = np.arange(validation_steps, dtype='f')
        y_validation_rmse = np.arange(validation_steps, dtype='f')
        #
        try:
            counter = 0
            for i in range(0,train_steps, int(train_steps / validation_steps)): #NOTE: step size changed to ensure only training steps stored at same iterations as validation errors are used.
                x_train[counter]    = training_loss[i][1]   # get training iteration
                y_train[counter]    = training_loss[i][2]   # get value
                counter += 1
        except IndexError as error:
           warnings.warn("The amount of training steps specified is larger than what is present in the log file. By default, the script will continue to execute while selecting only the steps that are available. Check whether this is intended behaviour.", RuntimeWarning)

        try:
            for i in range(validation_steps):
                x_validation[i]      = validation_loss[i][1] # get training iteration
                y_validation_loss[i] = validation_loss[i][2] # get value
                y_validation_rmse[i] = validation_rmse[i][2] # get value
        except IndexError as error:
           warnings.warn("The amount of validation steps specified is larger than what is present in the log file. By default, the script will continue to execute while selecting only the steps that are available. Check whether this is intended behaviour.", RuntimeWarning) 

        #Make plot for training/validation loss
        plt.figure(1)
        plt.plot(x_train*(10**(-5)), y_train, color=colorspec[j], linestyle=':',  linewidth=1.0, marker='o')
        plt.plot(x_validation*(10**(-5)), y_validation_loss, color=colorspec[j], label=str(n_hidden_list[j]), linewidth=2., marker='o')
        
        if first_iteration:
            plt.xlabel(r'$\rm Training\ iterations\ *\ 10^5\ [-]$', fontsize=20)
            plt.ylabel(r'$\rm log(loss)\ [-]$', fontsize=20)
            plt.xticks(fontsize = 16, rotation = 0)
            plt.yticks(fontsize = 16, rotation = 0)
        
        elif last_iteration:
            leg = plt.legend(bbox_to_anchor=(1.04,1.0), loc='upper left', title=r'$\mathrm{n_{hidden}}$', fontsize=16, title_fontsize=20, frameon=True)
            #Set linewidth in legend (different than in plot)
            for line in leg.get_lines():
                line.set_linewidth(2.)
            leg.get_frame().set_linewidth(1.0)
            leg.get_frame().set_edgecolor('k')
            #plt.tight_layout(rect=[0,0,1.4,1.0])
            ax = plt.gca()
            plt.savefig(save_path + "/loss.png", bbox_extra_artists=(leg,ax), bbox_inches='tight')

        #Make plot for RMSE
        plt.figure(2)

        plt.plot(x_validation*(10**(-5)), y_validation_rmse, color=colorspec[j], label=str(n_hidden_list[j]), linewidth=2., marker='o')
        
        if first_iteration:
            plt.xlabel(r'$\rm Training\ iterations\ *\ 10^5\ [-]$', fontsize=20)
            plt.ylabel(r'$\rm RMSE\ [-]$', fontsize=20)
            plt.xticks(fontsize = 16, rotation = 0)
            plt.yticks(fontsize = 16, rotation = 0)
            first_iteration = False
        
        elif last_iteration:
            leg = plt.legend(bbox_to_anchor=(1.04,1.0), loc='upper left', title=r'$\mathrm{n_{hidden}}$', fontsize=16, title_fontsize=20, frameon=True)
            #Set linewidth in legend (different than in plot)
            for line in leg.get_lines():
                line.set_linewidth(2.)
            leg.get_frame().set_linewidth(1.0)
            leg.get_frame().set_edgecolor('k')
            #plt.tight_layout(rect=[0,0,1.4,1.0])
            ax = plt.gca()
            plt.savefig(save_path + "/rmse.png", bbox_extra_artists=(leg,ax), bbox_inches='tight')

        ##Make plot for RMSE
        #plt.figure()
        #plt.plot(x_validation*(10**(-5)), y_validation_rmse, 'b-', label='Validation', linewidth=2.)
        #plt.xlabel(r'$\rm Training\ iterations\ *\ 10^5\ [-]$', fontsize=20)
        #plt.ylabel(r'$RMSE [-]$', fontsize=20)
        #plt.xticks(fontsize = 16, rotation = 0)
        #plt.yticks(fontsize = 16, rotation = 0)
        #leg = plt.legend(loc='upper right', fontsize=20, frameon=True)
        #leg.get_frame().set_linewidth(1.0)
        #leg.get_frame().set_edgecolor('k')
        #plt.savefig(save_path + "/rmse.png")

    #Close figures
    plt.close()

if __name__ == '__main__':
    log_files_training = [];
    log_files_validation = [];
    n_hidden_list = [1,2,4,8,16,32,64,128,256,512]; #Number of hidden neurons in each trained MLP, ordered according to the numbering of the MLPs

    for i in range(0,10): #Ensure ordering of trained MLPs is according to their numbering, which should be consistent with n_hidden_list
        log_files_training += glob.glob("../CNN_checkpoints/real_data_MLP3" + str(i) + "/*tfevents*")
        log_files_validation += glob.glob("../CNN_checkpoints/real_data_MLP3" + str(i) + "/eval_MLP1/*tfevents*")
    save_path = "/home/robinst/microhh/cases/moser600/git_repository/CNN_checkpoints/real_data_MLP30"
    plot_tensorflow_log(log_files_training, log_files_validation, n_hidden_list, save_path)
