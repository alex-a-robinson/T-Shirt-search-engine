#!/bin/usr/env python
import numpy as np

class Searcher:
    def __init__(self, index):
        '''Store index of images'''
        self.index = index
        
    def search(self, queryFeatures):
        '''Searches for the most relevent images
        from the data set'''
        results = {}
        
        for (k, features) in self.index.items():
            # compute the chi-squared distance between the
            # features
            d = self.chi2_distance(features, queryFeatures)
            
            results[k] = d
        
        # sort the resutls, so smaller distances i.e. more 
        # relevant images are at the front of the list
        results = sorted([(v, k) for (k, v) in results.items()])
        
        return results
        
    def chi2_distance(self, histA, histB, eps = 1e-10):
        '''Calculate the chi-squared distance (often used
        in computer vision field to compare histograms)'''
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
            
        return d
