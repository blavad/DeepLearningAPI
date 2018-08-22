import sys
sys.path.append("/home/david/Bureau/Documents/Travail/GAN_Dundee/GAN/project/lib/")

import os
from utils import *
import random

class Dataset(object):
    
    def __init__(self, data_train, data_test):
        self.data = data_train
        self.test = data_test
        
    def getNextBatch(self, batch_size):
        """
	Return next batch in the dataset
	"""
        raise NotImplementedError("(!) You must implement %s method" % type(self).__name__)
    
    @property
    def num_examples(self):
        return len(self.data)
    
    def getSize(self):
        return self.num_examples
    
    def shuffle(self, r=None):
        if r is None :
            random.shuffle(self.data)
        else :
	    random.shuffle(self.data,lambda : r)
            
    def getData(self):
	return self.data
    
