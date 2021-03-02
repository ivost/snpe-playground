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
FILE_LIST = "data/cropped/file_list.txt"
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
        input_files = [line.strip() for line in f.readlines()]
        input_files = [Path(f).name for f in input_files if Path(f).exists()]

    if len(input_files) <= 0:
        print('No files listed in input_files')
    else:
        print('Inference results')
        max_filename_len = max([len(file) for file in input_files])

        for idx, val in enumerate(input_files):
            cur_results_dir = 'Result_' + str(idx)
            cur_results_file = os.path.join(output_dir, cur_results_dir, 'detection_classes:0.raw')
            if not os.path.isfile(cur_results_file):
                raise RuntimeError('missing results file: ' + cur_results_file)
            # create numpy array from results tensor file
            float_array = np.fromfile(cur_results_file, dtype=np.float32)
            print(float_array)
            # if len(float_array) != 1001:
            #     raise RuntimeError(str(len(float_array)) + ' outputs in ' + cur_results_file)

            # if not verbose:
            #     max_prob = max(float_array)
            #     max_prob_index = np.where(float_array == max_prob)[0][0]
            #     max_prob_category = labels[max_prob_index]
            #
            #     display_text = '%s %.2f %s %s' % (
            #     val.ljust(max_filename_len), max_prob, str(max_prob_index).rjust(3), max_prob_category)
            #     print(display_text)
            # else:
            #     top5_prob = heapq.nlargest(5, range(len(float_array)), float_array.take)
            #     for i, idx in enumerate(top5_prob):
            #         prob = float_array[idx]
            #         prob_category = labels[idx]
            #         display_text = '%s %f %s %s' % (
            #             val.ljust(max_filename_len), prob, str(idx).rjust(3), prob_category)
            #         print(display_text)


if __name__ == '__main__':
    main()
