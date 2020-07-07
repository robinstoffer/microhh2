#!/bin/bash

#Select MLP with N_hidden=512 to incorporate in MicroHH (see `job_MLP_training`)
export MLPNUM=10

#Two commands to create optimized frozen graph:
python3 optimize_for_inference.py --input ./MLP${MLPNUM}/inference_graph.pbtxt --output ./MLP${MLPNUM}/optimized_inference_graph.pb --input_names input_u,input_v,input_w --placeholder_type_enum 1,1,1 --output_names denormalisation_output/output_layer_denorm,save/restore_all --frozen_graph=False #integers refer to data types in DataType enum; 1 is tf.float32, 9 is tf.int64
python3 freeze_graph.py --input_graph ./MLP${MLPNUM}/optimized_inference_graph.pb --input_checkpoint ./MLP${MLPNUM}/model.ckpt-500000 --output_graph ./MLP${MLPNUM}/frozen_inference_graph.pb --output_node_names denormalisation_output/output_layer_denorm

#Extract weights from optimzed frozen graph for manual inference in MicroHH
python3 extract_weights_frozen_graph.py --frozen_graph_filename ./MLP${MLPNUM}/frozen_inference_graph.pb --variables_filepath ./MLP${MLPNUM}/
