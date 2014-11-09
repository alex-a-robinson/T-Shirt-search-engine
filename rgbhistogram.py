#!/usr/bin/env python

import numpy as np
import cv2

class RGBHistogram:
    def __init__(self, bins):
        ''' Store the number of bins ths histogram will use'''
        self.bins = bins
    def describe(self, image):
        ''' Computes a 3D histogram in RGB colourspace then
        Normalize the histogram so that images iwth the same
        content, but either scaled larger or smaller will 
        have (roughly) the same histgram'''
        hist = cv2.calcHist([image], [0, 1, 2], None, 
            self.bins, [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist)
        
        # return 3D histogram as a flattened array
        return hist.flatten()
