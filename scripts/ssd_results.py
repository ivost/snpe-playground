#
# Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#

import argparse
import os
# import heapq
from pathlib import Path

import PIL.Image as Image
# import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np

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
    labels = read_labels()
    verbose = args.verbose_results

    if not os.path.isfile(input_list):
        raise RuntimeError('input_list %s does not exist' % input_list)
    if not os.path.isdir(output_dir):
        raise RuntimeError('output_dir %s does not exist' % output_dir)
    with open(input_list, 'r') as f:
        input_files = [line.strip() for line in f.readlines() if not line.startswith("#")]
        input_files = [Path(f).name for f in input_files if Path(f).exists()]

    if len(input_files) <= 0:
        print('No files listed in input_files')
        return

    print('Inference results')
    max_filename_len = max([len(file) for file in input_files])
    results_dir = Path(ROOT, "results")
    if not results_dir.exists():
        results_dir.mkdir()

    for idx, name in enumerate(input_files):
        print("Image", name)
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

        boxes = read_results(output, 'Postprocessor/BatchMultiClassNonMaxSuppression_boxes.raw')
        boxes = np.resize(boxes, n * 4)
        print("boxes", boxes)

        name = name.replace("raw", "jpg")
        img_file = Path(ROOT, "processed", name)
        img_file_out = Path(ROOT, "results", name)
        img = Image.open(img_file)

        for i in range(n):
            j = i*4
            cat = int(classes[i])
            label = labels[cat-1]
            confidence = int(100*scores[i])
            msg = "{}({}%)".format(label, confidence)
            # msg = "%d - %d %" % (cat, conf)
            draw_bounding_box_on_image(img,
                                       boxes[j],
                                       boxes[j+1],
                                       boxes[j+2],
                                       boxes[j+3],
                                       color='yellow',
                                       display_str_list=[msg])

        img.save(img_file_out, "JPEG")
        img.show()


def read_results(dir, name):
    results_file = os.path.join(dir, name)
    arr = np.fromfile(results_file, dtype=np.float32)
    # print(name, arr)
    return arr

def read_labels():
    labels_file = os.path.join(ROOT, LABELS)
    with open(labels_file, 'r') as f:
        labels = [line.strip().split(" ")[1] for line in f.readlines()]
    return labels

def draw_bounding_box_on_image(image,
                               ymin,
                               xmin,
                               ymax,
                               xmax,
                               color='red',
                               thickness=3,
                               display_str_list=(),
                               use_normalized_coordinates=True):
    """Adds a bounding box to an image.

  Bounding box coordinates can be specified in either absolute (pixel) or
  normalized coordinates by setting the use_normalized_coordinates argument.

  Each string in display_str_list is displayed on a separate line above the
  bounding box in black text on a rectangle filled with the input 'color'.
  If the top of the bounding box extends to the edge of the image, the strings
  are displayed below the bounding box.

  Args:
    image: a PIL.Image object.
    ymin: ymin of bounding box.
    xmin: xmin of bounding box.
    ymax: ymax of bounding box.
    xmax: xmax of bounding box.
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list: list of strings to display in box
                      (each to be shown on its own line).
    use_normalized_coordinates: If True (default), treat coordinates
      ymin, xmin, ymax, xmax as relative to the image.  Otherwise treat
      coordinates as absolute.
  """
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    if use_normalized_coordinates:
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                      ymin * im_height, ymax * im_height)
    else:
        (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    draw.line([(left, top), (left, bottom), (right, bottom),
               (right, top), (left, top)], width=thickness, fill=color)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                              text_bottom)],
            fill=color)
        draw.text(
            (left + margin, text_bottom - text_height - margin),
            display_str,
            fill='black',
            font=font)
        text_bottom -= text_height - 2 * margin


def draw_bounding_boxes_on_image_array(image,
                                       boxes,
                                       color='red',
                                       thickness=4,
                                       display_str_list_list=()):
    """Draws bounding boxes on image (numpy array).

  Args:
    image: a numpy array object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
    image_pil = Image.fromarray(image)
    draw_bounding_boxes_on_image(image_pil, boxes, color, thickness,
                                 display_str_list_list)
    np.copyto(image, np.array(image_pil))


def draw_bounding_boxes_on_image(image,
                                 boxes,
                                 color='yellow',
                                 thickness=2,
                                 display_str_list_list=()):
    """Draws bounding boxes on image.

  Args:
    image: a PIL.Image object.
    boxes: a 2 dimensional numpy array of [N, 4]: (ymin, xmin, ymax, xmax).
           The coordinates are in normalized format between [0, 1].
    color: color to draw bounding box. Default is red.
    thickness: line thickness. Default value is 4.
    display_str_list_list: list of list of strings.
                           a list of strings for each bounding box.
                           The reason to pass a list of strings for a
                           bounding box is that it might contain
                           multiple labels.

  Raises:
    ValueError: if boxes is not a [N, 4] array
  """
    boxes_shape = boxes.shape
    if not boxes_shape:
        return
    if len(boxes_shape) != 2 or boxes_shape[1] != 4:
        raise ValueError('Input must be of size [N, 4]')
    for i in range(boxes_shape[0]):
        display_str_list = ()
        if display_str_list_list:
            display_str_list = display_str_list_list[i]
        draw_bounding_box_on_image(image, boxes[i, 0], boxes[i, 1], boxes[i, 2],
                                   boxes[i, 3], color, thickness, display_str_list)


if __name__ == '__main__':
    main()

'''
'''
