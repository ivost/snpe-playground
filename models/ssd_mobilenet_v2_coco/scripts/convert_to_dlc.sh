#!/bin/bash

snpe-tensorflow-to-dlc \
  --input_network ../assets/frozen_inference_graph.pb \
  --input_dim Preprocessor/sub 1,300,300,3 \
  --out_node detection_classes \
  --out_node detection_boxes \
  --out_node detection_scores \
  --output_path ../dlc/mobilenet_ssd.dlc 
