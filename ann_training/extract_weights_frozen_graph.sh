#!/bin/bash

#Select MLP with N_hidden=64 (and trained based only on 8dx4dz; the same resolution as used a posteriori) to incorporate in MicroHH. Do necessary post-processing.
export MLPNUM=7

#Two commands to create optimized frozen graph:
python3 optimize_for_inference.py --input /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP${MLPNUM}/inference_graph.pbtxt --output /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP${MLPNUM}/optimized_inference_graph.pb --input_names input_u,input_v,input_w --placeholder_type_enum 1,1,1 --output_names denormalisation_output/output_layer_denorm,save/restore_all --frozen_graph=False #integers refer to data types in DataType enum; 1 is tf.float32, 9 is tf.int64
python3 freeze_graph.py --input_graph /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP${MLPNUM}/optimized_inference_graph.pb --input_checkpoint /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP${MLPNUM}/model.ckpt-500000 --output_graph /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP${MLPNUM}/frozen_inference_graph.pb --output_node_names denormalisation_output/output_layer_denorm

#Extract weights from optimzed frozen graph for manual inference in MicroHH
python3 extract_weights_frozen_graph.py --frozen_graph_filename /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP${MLPNUM}/frozen_inference_graph.pb --variables_filepath /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP${MLPNUM}/

##Move selected MLP to separate directory
##Create directory for MLP to be selected, remove it first if it already exists
#if [ -d "/projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP_selected" ]; then
#    rm -r /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP_selected
#fi
#
#mkdir /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP_selected
#
##Copy the txt-files, containing the paramters of the MLP selected for incorporation in MicroHH, and the predictions nc-file to another directory. This is needed because the path containing the ANN parameters is hard-coded in MicroHH and in the job script job_readCNNsmagpredictions.
#cp /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP${MLPNUM}/*.txt /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP_selected
#cp /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP${MLPNUM}/*.nc  /projects/1/flowsim/gmd_results_training/train_8dx4dz_test_all/MLP_selected
##

#Select MLP with N_hidden=64 (and trained based only on 4dx4dz+12dx4dz) to incorporate in MicroHH. Do necessary post-processing.
export MLPNUM=7

#Two commands to create optimized frozen graph:
python3 optimize_for_inference.py --input /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP${MLPNUM}/inference_graph.pbtxt --output /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP${MLPNUM}/optimized_inference_graph.pb --input_names input_u,input_v,input_w --placeholder_type_enum 1,1,1 --output_names denormalisation_output/output_layer_denorm,save/restore_all --frozen_graph=False #integers refer to data types in DataType enum; 1 is tf.float32, 9 is tf.int64
python3 freeze_graph.py --input_graph /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP${MLPNUM}/optimized_inference_graph.pb --input_checkpoint /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP${MLPNUM}/model.ckpt-500000 --output_graph /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP${MLPNUM}/frozen_inference_graph.pb --output_node_names denormalisation_output/output_layer_denorm

#Extract weights from optimzed frozen graph for manual inference in MicroHH
python3 extract_weights_frozen_graph.py --frozen_graph_filename /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP${MLPNUM}/frozen_inference_graph.pb --variables_filepath /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP${MLPNUM}/

##Move selected MLP to separate directory
##Create directory for MLP to be selected, remove it first if it already exists
#if [ -d "/projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP_selected" ]; then
#    rm -r /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP_selected
#fi
#
#mkdir /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP_selected
#
##Copy the txt-files, containing the paramters of the MLP selected for incorporation in MicroHH, and the predictions nc-file to another directory. This is needed because the path containing the ANN parameters is hard-coded in MicroHH and in the job script job_readCNNsmagpredictions.
#cp /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP${MLPNUM}/*.txt /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP_selected
#cp /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP${MLPNUM}/*.nc  /projects/1/flowsim/gmd_results_training/train_4dx4dz_12dx4dz_test_all/MLP_selected
#
#Select MLP with N_hidden=64 (and trained based on all horizontal coarse-graining factors) to incorporate in MicroHH. Do necessary post-processing.
export MLPNUM=7

#Two commands to create optimized frozen graph:
python3 optimize_for_inference.py --input /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP${MLPNUM}/inference_graph.pbtxt --output /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP${MLPNUM}/optimized_inference_graph.pb --input_names input_u,input_v,input_w --placeholder_type_enum 1,1,1 --output_names denormalisation_output/output_layer_denorm,save/restore_all --frozen_graph=False #integers refer to data types in DataType enum; 1 is tf.float32, 9 is tf.int64
python3 freeze_graph.py --input_graph /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP${MLPNUM}/optimized_inference_graph.pb --input_checkpoint /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP${MLPNUM}/model.ckpt-500000 --output_graph /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP${MLPNUM}/frozen_inference_graph.pb --output_node_names denormalisation_output/output_layer_denorm

#Extract weights from optimzed frozen graph for manual inference in MicroHH
python3 extract_weights_frozen_graph.py --frozen_graph_filename /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP${MLPNUM}/frozen_inference_graph.pb --variables_filepath /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP${MLPNUM}/

##Move selected MLP to separate directory
##Create directory for MLP to be selected, remove it first if it already exists
#if [ -d "/projects/1/flowsim/gmd_results_training/train_all_test_all/MLP_selected" ]; then
#    rm -r /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP_selected
#fi
#
#mkdir /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP_selected
#
##Copy the txt-files, containing the paramters of the MLP selected for incorporation in MicroHH, and the predictions nc-file to another directory. This is needed because the path containing the ANN parameters is hard-coded in MicroHH and in the job script job_readCNNsmagpredictions.
#cp /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP${MLPNUM}/*.txt /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP_selected
#cp /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP${MLPNUM}/*.nc  /projects/1/flowsim/gmd_results_training/train_all_test_all/MLP_selected
#
