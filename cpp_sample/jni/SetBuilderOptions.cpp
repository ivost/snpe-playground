//==============================================================================
//
//  Copyright (c) 2017-2019 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "SetBuilderOptions.hpp"

#include "SNPE/SNPE.hpp"
#include "DlContainer/IDlContainer.hpp"
#include "SNPE/SNPEBuilder.hpp"

std::unique_ptr<zdl::SNPE::SNPE> setBuilderOptions(std::unique_ptr<zdl::DlContainer::IDlContainer> & container,
                                                   zdl::DlSystem::Runtime_t runtime,
                                                   zdl::DlSystem::RuntimeList runtimeList,
                                                   zdl::DlSystem::UDLBundle udlBundle,
                                                   bool useUserSuppliedBuffers,
                                                   zdl::DlSystem::PlatformConfig platformConfig,
                                                   bool useCaching,
                                                   bool isSSD)
{
    std::unique_ptr<zdl::SNPE::SNPE> snpe;
    zdl::SNPE::SNPEBuilder snpeBuilder(container.get());
    // snpeBuilder.setDebugMode(true);

    if(runtimeList.empty())
    {
        runtimeList.add(runtime);
    }

    zdl::DlSystem::StringList layers;
    if (isSSD) {
        layers.append("add");
        layers.append("Postprocessor/BatchMultiClassNonMaxSuppression");
        // detection_boxes,detection_scores,detection_classes
//        layers.append("detection_classes");
//        layers.append("detection_boxes");
//        layers.append("detection_scores");
    }
    // detection_boxes,detection_scores,detection_classes,num_detections
    snpe = snpeBuilder
            .setOutputLayers(layers)
            .setRuntimeProcessor(runtime)
       //.setRuntimeProcessorOrder(runtimeList)
       //.setUdlBundle(udlBundle)
       //.setUseUserSuppliedBuffers(useUserSuppliedBuffers)
       //.setPlatformConfig(platformConfig)
       //.setInitCacheMode(useCaching)
       .build();

    return snpe;
}

/*
 * Convert the frozen graph using the snpe-tensorflow-to-dlc converter.

snpe-tensorflow-to-dlc --input_network <path_to>/exported/frozen_inference_graph.pb --input_dim Preprocessor/sub 1,300,300,3 --out_node detection_classes --out_node detection_boxes --out_node detection_scores ---output_path mobilenet_ssd.dlc --allow_unconsumed_nodes
After SNPE conversion you should have a mobilenet_ssd.dlc that can be loaded and run in the SNPE runtimes.

The output layers for the model are:

Postprocessor/BatchMultiClassNonMaxSuppression
add
The output buffer names are:

(classes) detection_classes:0 (+1 index offset)
(classes) Postprocessor/BatchMultiClassNonMaxSuppression_classes (0 index offset)
(boxes) Postprocessor/BatchMultiClassNonMaxSuppression_boxes
(scores) Postprocessor/BatchMultiClassNonMaxSuppression_scores
 */