import os
from utils import *
import random

from dataset import Dataset

class LabeledDataset(Dataset):
    
    def __init__(self, dataset, labels):
        self.data = dataset
        self.labels = labels

    @property
    def num_examples(self):
        return self.data.num_examples
    
    def shuffle(self):
        r = random.random()
        self.data.shuffle(r)
        self.labels.shuffle(r)
        return
    
    def getNextBatch(self, batch_size, index):
        """
	Return next batch in the dataset
	"""
        if ((index+1)*batch_size > self.num_examples):
            print("(!) Essaye de lire infos hors data")
        else :
            return self.data.getNextBatch(batch_size,index) , self.labels.getNextBatch(batch_size,index)
    
    def getData(self):
        print(self.data.getData())
        print(self.labels.getData())
	return zip(self.data.getData(), self.labels.getData())
