#!/bin/usr/env pyton
from searcher import Searcher
from rgbhistogram import RGBHistogram
import numpy as np
import argparse
import cPickle
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True,
    help = 'Path to the directory that contains the images we just indexed')
ap.add_argument('-i', '--index', required = True,
    help = 'Path to where we stored our index')
ap.add_argument('-q', '--query', required = True,
    help = 'Path to query image')
args = vars(ap.parse_args())

# load the query image and show it
queryImage = cv2.imread(args['query'])
cv2.imshow('Query', queryImage)
print('query: %s' % (args['query']))

# describe the query
desc = RGBHistogram([8, 8, 8])
queryFeatures = desc.describe(queryImage)

# load the index and initialise our searcher
index = cPickle.loads(open(args['index']).read())
searcher = Searcher(index)
results = searcher.search(queryFeatures)

# initialise the a montage to display the results
# displaying the top 5 results (images are 460x636 pixels)
montage = np.zeros((460 * 5, 636, 3), dtype = 'uint8')

for j in xrange(0, 5):
    # grab results and load the result image
    (score, imageName) = results[j]
    path = args['dataset'] + '/%s' % (imageName)
    result = cv2.imread(path)
    print('\t%d. %s : %.3f' % (j + 1, imageName, score))

    montage[j * 460:(j + 1) * 460, :] = result

cv2.imshow('Result 1-5', montage)
cv2.waitKey(0)
