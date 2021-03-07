#
# Copyright (c) 2016,2018-2019 Qualcomm Technologies, Inc.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
import argparse
import subprocess
from pathlib import Path

import numpy as np
import os

from PIL import Image

ROOT = "/home/ivo/github/snpe-playground/models/ssd_mobilenet_v2_coco"

RESIZE_METHOD_ANTIALIAS = "antialias"
RESIZE_METHOD_BILINEAR = "bilinear"


def __get_img_raw(img_filepath):
    img_filepath = os.path.abspath(img_filepath)
    img = Image.open(img_filepath)
    arr = np.array(img)  # read it
    if len(arr.shape) != 3:
        raise RuntimeError('Image shape' + str(arr.shape))
    if arr.shape[2] != 3:
        raise RuntimeError('Require image with rgb but channel is %d' % arr.shape[2])
    # reverse last dimension: rgb -> bgr
    return arr

def __create_mean_raw(img_raw, mean_rgb):
    if img_raw.shape[2] != 3:
        raise RuntimeError('Require image with rgb but channel is %d' % img_raw.shape[2])
    img_dim = (img_raw.shape[0], img_raw.shape[1])
    mean_raw_r = np.empty(img_dim)
    mean_raw_r.fill(mean_rgb[0])
    mean_raw_g = np.empty(img_dim)
    mean_raw_g.fill(mean_rgb[1])
    mean_raw_b = np.empty(img_dim)
    mean_raw_b.fill(mean_rgb[2])
    # create with c, h, w shape first
    tmp_transpose_dim = (img_raw.shape[2], img_raw.shape[0], img_raw.shape[1])
    mean_raw = np.empty(tmp_transpose_dim)
    mean_raw[0] = mean_raw_r
    mean_raw[1] = mean_raw_g
    mean_raw[2] = mean_raw_b
    # back to h, w, c
    mean_raw = np.transpose(mean_raw, (1, 2, 0))
    return mean_raw.astype(np.float32)


def __create_raw_incv3(img_filepath, mean_rgb, div, req_bgr_raw, save_uint8):
    img_raw = __get_img_raw(img_filepath)
    mean_raw = __create_mean_raw(img_raw, mean_rgb)

    snpe_raw = img_raw - mean_raw
    snpe_raw = snpe_raw.astype(np.float32)
    # scalar data divide
    snpe_raw /= div

    if req_bgr_raw:
        snpe_raw = snpe_raw[..., ::-1]

    if save_uint8:
        snpe_raw = snpe_raw.astype(np.uint8)
    else:
        snpe_raw = snpe_raw.astype(np.float32)

    img_filepath = os.path.abspath(img_filepath)
    filename, ext = os.path.splitext(img_filepath)
    snpe_raw_filename = filename
    snpe_raw_filename += '.raw'
    snpe_raw.tofile(snpe_raw_filename)

    return 0


def __resize_square_to_jpg(src, dst, size, resize_type):
    src_img = Image.open(src)
    # If black and white image, convert to rgb (all 3 channels the same)
    if len(np.shape(src_img)) == 2:
        src_img = src_img.convert(mode='RGB')
    # center crop to square
    width, height = src_img.size
    short_dim = min(height, width)
    crop_coord = (
        (width - short_dim) / 2,
        (height - short_dim) / 2,
        (width + short_dim) / 2,
        (height + short_dim) / 2
    )
    img = src_img.crop(crop_coord)
    if resize_type == RESIZE_METHOD_BILINEAR:
        dst_img = img.resize((size, size), Image.BILINEAR)
    else:
        dst_img = img.resize((size, size), Image.ANTIALIAS)
    # save output - save determined from file extension
    dst_img.save(dst)
    return 0


def convert_img(src, dest, size, resize_type):
    print("Converting images for ssd mobilenet.")

    print("Scaling to square: " + src)
    for root, dirs, files in os.walk(src):
        for jpgs in files:
            src_image = os.path.join(root, jpgs)
            if '.jpg' in src_image:
                print(src_image)
                dest_image = os.path.join(dest, jpgs)
                __resize_square_to_jpg(src_image, dest_image, size, resize_type)

    print("Normalizing images in: " + dest)
    for root, dirs, files in os.walk(dest):
        for jpgs in files:
            src_image = os.path.join(root, jpgs)
            if '.jpg' in src_image:
                print(src_image)
                mean_rgb = (128, 128, 128)
                __create_raw_incv3(src_image, mean_rgb, 128, False, False)


def main():
    parser = argparse.ArgumentParser(description="Batch convert jpgs",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--img_folder', type=str, default=os.path.join(ROOT, 'data'))
    parser.add_argument('-d', '--dest', type=str, default=os.path.join(ROOT, 'processed'))
    parser.add_argument('-s', '--size', type=int, default=300)
    parser.add_argument('-r', '--resize_type', type=str, default=RESIZE_METHOD_BILINEAR,
                        help='Select image resize type antialias or bilinear. Image resize type should match '
                             'resize type used on images with which model was trained, otherwise there may be impact '
                             'on model accuracy measurement.')

    args = parser.parse_args()
    #scripts_dir = os.path.join('..', '..', '..', 'scripts')
    scripts_dir = '.'
    create_file_list_script = os.path.join(scripts_dir, 'create_file_list.py')

    size = args.size
    src = os.path.abspath(args.img_folder)
    if not Path(src).exists():
        os.mkdir(args.dest)
    dest = os.path.abspath(args.dest)
    if not Path(dest).exists():
        os.mkdir(dest)
    resize_type = args.resize_type
    assert resize_type == RESIZE_METHOD_BILINEAR or resize_type == RESIZE_METHOD_ANTIALIAS, \
        "Image resize method should be antialias or bilinear"

    cmd = ['python', create_file_list_script,
           '-i', src,
           '-o', os.path.join(src, "file_list.txt"),
           '-e', '*.jpg',
           ]
    subprocess.call(cmd)

    convert_img(src, dest, size, resize_type)

    cmd = ['python', create_file_list_script,
           '-i', dest,
           '-o', os.path.join(dest, "file_list.txt"),
           '-e', '*.raw',
           ]
    subprocess.call(cmd)


if __name__ == '__main__':
    main()
