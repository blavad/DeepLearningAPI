import sys
sys.path.append("/home/david/Bureau/Documents/Travail/GAN_Dundee/GAN/project/lib/")

import os
from utils import *
import random
from dataset import Dataset

class MnistDataset(Dataset):
    
    def __init__(self, dataset):
        self.data = dataset
        
    def getNextBatch(self, batch_size, step):
        """
	Return next batch in the dataset
	"""
        return self.data.next_batch(batch_size)
        
    @property
    def num_examples(self):
        return self.data.num_examples
    
