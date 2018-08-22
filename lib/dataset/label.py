import os
from utils import *
import random

from dataset import Dataset

class Label(Dataset):
    
    def __init__(self, dataset):
        self.data = dataset
        
    def getNextBatch(self, batch_size,index):
        """
	Return next batch in the dataset
	"""
	return self.data[index*batch_size:(index+1)*batch_size]
