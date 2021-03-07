#
# Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import argparse
import heapq
from pathlib import Path

import numpy as np
import os

'''

'''
ROOT = "/home/ivo/github/snpe-playground/models/ssd_mobilenet_v2_coco"
LABELS = "data/labels.txt"
FILE_LIST = "processed/file_list.txt"
RESULTS = "/tmp/output"


def main():
    parser = argparse.ArgumentParser(description='Display inference results.')
    parser.add_argument('-i', '--input_list', default=os.path.join(ROOT, FILE_LIST),
                        help='File containing input list used to generate output_dir.')
    parser.add_argument('-o', '--output_dir',
                        help='Output directory containing result files matching input_list.',
                        default=RESULTS)
    parser.add_argument('-l', '--labels_file',
                        help='Path to labels', default=os.path.join(ROOT, LABELS))
    parser.add_argument('-v', '--verbose_results',
                        help='Verbose', action='store_true')
    args = parser.parse_args()

    input_list = os.path.abspath(args.input_list)
    output_dir = os.path.abspath(args.output_dir)
    labels_file = os.path.abspath(args.labels_file)
    verbose = args.verbose_results

    if not os.path.isfile(input_list):
        raise RuntimeError('input_list %s does not exist' % input_list)
    if not os.path.isdir(output_dir):
        raise RuntimeError('output_dir %s does not exist' % output_dir)
    if not os.path.isfile(labels_file):
        raise RuntimeError('labels_file %s does not exist' % labels_file)
    with open(labels_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    # if len(labels) != 1001:
    #     raise RuntimeError('Invalid labels_file: need 1000 categories')
    with open(input_list, 'r') as f:
        input_files = [line.strip() for line in f.readlines() if not line.startswith("#")]
        input_files = [Path(f).name for f in input_files if Path(f).exists()]

    if len(input_files) <= 0:
        print('No files listed in input_files')
        return

    print('Inference results')
    max_filename_len = max([len(file) for file in input_files])

    for idx, file in enumerate(input_files):
        print("Image", file)
        output = os.path.join(output_dir, 'Result_' + str(idx))
        num_detections = read_results(output, 'Postprocessor/BatchMultiClassNonMaxSuppression_num_detections.raw')
        # print("num_detections", num_detections)
        n = int(num_detections[0])
        if verbose:
            print("detected", n, "objects")
        if n < 1:
            continue
        scores = read_results(output, 'Postprocessor/BatchMultiClassNonMaxSuppression_scores.raw')
        scores = np.resize(scores, n)
        print("scores", scores)
        classes = read_results(output, 'detection_classes:0.raw')
        classes = np.resize(classes, n)
        print("classes", classes)
        # same
        # classes2 = read_results(output, 'Postprocessor/BatchMultiClassNonMaxSuppression_classes.raw')
        # classes2 = np.resize(classes2, n)
        # print("classes2", classes)

        boxes = read_results(output, 'Postprocessor/BatchMultiClassNonMaxSuppression_boxes.raw')
        boxes = np.resize(boxes, n*4)
        print("boxes", boxes)


def read_results(dir, name):
    results_file = os.path.join(dir, name)
    arr = np.fromfile(results_file, dtype=np.float32)
    # print(name, arr)
    return arr


if __name__ == '__main__':
    main()

'''
'''
