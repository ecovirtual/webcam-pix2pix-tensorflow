#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: memo

loads bunch of images from a folder (and recursively from subfolders)
preprocesses (resize or crop, canny edge detection) and saves into a new folder
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import os
import cv2
import PIL.Image

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", required=True, help="path to folder containing images")
parser.add_argument("--output_dir", required=True, help="output path")
# parser.add_argument("--operation", required=True, choices=["grayscale", "resize", "blank", "combine", "edges"])
# parser.add_argument("--workers", type=int, default=1, help="number of workers")
parser.add_argument("--crop", action="store_true", help="resizes shortest edge to target dimensions and crops other edge. If false, does non-uniform resize")
# resize
parser.add_argument("--pad", action="store_true", help="pad instead of crop for resize operation")
parser.add_argument("--size", type=int, default=256, help="size to use for resize operation")
# threshold 
parser.add_argument("--canny_threshold_1", type=int, default=100, help="canny 1st threshold")
parser.add_argument("--canny_threshold_2", type=int, default=200, help="canny 2nd threshold")

a = parser.parse_args()

# dim = 256  # target dimensions, 
# do_crop = False # if true, resizes shortest edge to target dimeensions and crops other edge. If false, does non-uniform resize
do_crop = a.crop
canny_thresh1 = a.canny_threshold_1
canny_thresh2 = a.canny_threshold_2

# root_path = '../../../data'
in_path = os.path.dirname(a.input_dir)
out_path = os.path.dirname(a.output_dir)


#########################################
out_path += '_' + str(a.size) + '_p2p_canny'
if do_crop:
    out_path += '_crop'

out_shape = (a.size, a.size)

if os.path.exists(out_path) == False:
    os.makedirs(out_path)

# eCryptfs file system has filename length limit of around 143 chars! 
# https://unix.stackexchange.com/questions/32795/what-is-the-maximum-allowed-filename-and-folder-size-with-ecryptfs
max_fname_len = 140 # leave room for extension


def get_file_list(path, extensions=['jpg', 'jpeg', 'png']):
    '''returns a (flat) list of paths of all files of (certain types) recursively under a path'''
    paths = [os.path.join(root, name)
             for root, dirs, files in os.walk(path)
             for name in files
             if name.lower().endswith(tuple(extensions))]
    return paths


paths = get_file_list(in_path)
print('{} files found'.format(len(paths)))


for i,path in enumerate(paths):
    path_d, path_f = os.path.split(path)
    
    # combine path and filename to create unique new filename
    out_fname = path_d.split('/')[-1] + '_' + path_f
                            
    # take last n characters so doesn't go over filename length limit
    out_fname = os.path.splitext(out_fname)[0][-max_fname_len+4:] + '.jpg'
    
    print('File {} of {}, {}'.format(i, len(paths), out_fname))
    im = PIL.Image.open(path)
    im = im.convert('RGB')
    if do_crop:
        resize_shape = list(out_shape)
        if im.width < im.height:
            resize_shape[1] = int(round(float(im.height) / im.width * a.size))
        else:
            resize_shape[0] = int(round(float(im.width) / im.height * a.size))
        im = im.resize(resize_shape, PIL.Image.BICUBIC)
        hw = int(im.width / 2)
        hh = int(im.height / 2)
        hd = int(a.size/2)
        area = (hw-hd, hh-hd, hw+hd, hh+hd)
        im = im.crop(area)            
            
    else:
        im = im.resize(out_shape, PIL.Image.BICUBIC)
        
    a1 = np.array(im) 
    a2 = cv2.Canny(a1, canny_thresh1, canny_thresh2)
    a2 = cv2.cvtColor(a2, cv2.COLOR_GRAY2RGB)                 
    a3 = np.concatenate((a1,a2), axis=1)
    im = PIL.Image.fromarray(a3)                     
                       
    im.save(os.path.join(out_path, out_fname))

