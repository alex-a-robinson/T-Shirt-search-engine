#!/usr/bin/env python

from rgbhistogram import RGBHistogram
import argparse
import cPickle
import glob
import cv2

# Constuct the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
    help = 'Path to the directory which contains the images')
ap.add_argument('-i', '--index', required=True,
    help = 'Path to where the computed index will be stored')
args = vars(ap.parse_args())

# index dictionary key - image filename, value - computed features
index = {}

# initialize 3D histogram image descriptor
# 8 bits per channel
desc = RGBHistogram([8, 8, 8])

# loop over all pngs in dataset directory
for path in glob.glob(args['dataset'] + '/*.jpg'):
    
    # extract filename
    k = path[path.rfind('/') + 1:]
    
    # load image, describe it with RGB histogram descriptor
    # and update index
    image = cv2.imread(path)
    features = desc.describe(image)
    index[k] = features
    
# index image
f = open(args['index'], 'w')
f.write(cPickle.dumps(index))
f.close()
