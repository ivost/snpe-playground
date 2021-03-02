#!/bin/bash

#INPUT_TYPE=image_tensor
#PIPELINE_CONFIG_PATH=../assets/pipeline.config
#TRAINED_CKPT_PREFIX=../assets/model.ckpt
#EXPORT_DIR=../assets/exported
#
#pushd $HOME/github/tfmodels/models/research
#python object_detection/export_inference_graph.py \
#--input_type=${INPUT_TYPE} \
#--pipeline_config_path=${PIPELINE_CONFIG_PATH} \
#--trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
#--output_directory=${EXPORT_DIR}
#popd

# snpe-tensorflow-to-dlc --input_network <path_to>/exported/frozen_inference_graph.pb --input_dim Preprocessor/sub 1,300,300,3 --out_node detection_classes --out_node detection_boxes --out_node detection_scores ---output_path mobilenet_ssd.dlc --allow_unconsumed_nodes

snpe-tensorflow-to-dlc \
  --input_network ../assets/frozen_inference_graph.pb \
  --input_dim Preprocessor/sub 1,300,300,3 \
  --out_node detection_classes \
  --out_node detection_boxes \
  --out_node detection_scores \
  --output_path ../dlc/mobilenet_ssd.dlc 

# --show_unconsumed_nodes
#  --allow_unconsumed_nodes
