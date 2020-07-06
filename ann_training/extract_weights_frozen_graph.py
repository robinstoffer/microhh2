#Script to extract weights from a frozen tensorflow graph.
import argparse
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util
import numpy as np

def extract_weights_graph(frozen_graph_filename, variables_filepath):
    #Load protopub file from disk (i.e. the frozen graph) and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #Import graph_def into a new Graph and return it
    with tf.Graph().as_default() as graph: #Make sure to define a new graph
        # The name var will prefix every op/nodes in the graph.
        # Since everything is loaded in a new graph, this is not needed.
        tf.import_graph_def(graph_def, name='')
        const_nodes = [n for n in graph_def.node if n.op == 'Const']

    for n in range(len(const_nodes)):
        print(n, const_nodes[n].name)

    #List of all nodes/layers in const_nodes that need to be saved in text files for manual implementation of the MLP.
    nodes_idx_name = {
            'means_inputs'       : 0,
            'stdevs_inputs'      : 1,
            'means_labels'       : 2,
            'stdevs_labels'      : 3,
            'utau_ref'           : 4,
            'MLPu_hidden_kernel' : 34,
            'MLPu_hidden_bias'   : 35,
            'MLPu_hidden_alpha'  : 36,
            'MLPu_output_kernel' : 37,
            'MLPu_output_bias'   : 38,
            'MLPv_hidden_kernel' : 50,
            'MLPv_hidden_bias'   : 51,
            'MLPv_hidden_alpha'  : 52,
            'MLPv_output_kernel' : 53,
            'MLPv_output_bias'   : 54,
            'MLPw_hidden_kernel' : 66,
            'MLPw_hidden_bias'   : 67,
            'MLPw_hidden_alpha'  : 68,
            'MLPw_output_kernel' : 69,
            'MLPw_output_bias'   : 70,
            'output_denorm_utau2': 72,
            }

    #Loop over all nodes/layers specified above to change them into numpy arrays and subsequently store them in text files.
    for i in nodes_idx_name.keys():
        values = tensor_util.MakeNdarray(const_nodes[nodes_idx_name[i]].attr['value'].tensor)
        #if len(values.shape) == 2:
        #    print(values.shape)
        #    #Flatten weight arrays to ease the manual implementation of the MLP
        #    values = values.flatten()
        #Make sure all arrays are at least 1D (needed for np.savetxt below)
        values = np.atleast_1d(values)
        np.savetxt(variables_filepath+i+'.txt', values)

if __name__ == '__main__':
    #If script executed, extract weights from specified frozen graph
    
    #Parse input
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_graph_filename", default="frozen_inference_graph.pb")
    parser.add_argument("--variables_filepath", default="", help="directory where variables extracted from the frozen graph are stored.")
    args = parser.parse_args()

    #Call function that extracts and stores weights
    extract_weights_graph(args.frozen_graph_filename, args.variables_filepath)
