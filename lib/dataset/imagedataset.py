from dataset import Dataset

import os
from glob import glob
import numpy as np
from utils import *

import matplotlib.pyplot as plt


class ImageDataset(Dataset):

    def __init__(self, regex_file, input_width = 28, input_height = 28, as_line = True):
        self.data = glob(regex_file)
        if self.num_examples == 0 :
            raise Exception("(!) No data found")
        #print(self.data)
        self.grayscale = self._grayscale()
        print("Grayscale = "+str(self.grayscale))
        self.input_height = input_height
        self.input_width = input_width
        self.as_line = as_line

    def _grayscale(self):
        imgTest = imread(self.data[0])
        if len(imgTest.shape) >= 3: 
            return 0
        else:
            return 1

    def getNextBatch(self, batch_size, index):
        """
	Return next batch in the dataset
	"""
        batch_files = self.data[index*batch_size:(index+1)*batch_size]
        batch = [ get_image(batch_file, self.input_height, self.input_width,
                            resize_height=self.input_height, resize_width=self.input_width,
                            grayscale=self.grayscale)
                  for batch_file in batch_files]
        #plt.imshow(batch[0])
        #plt.show()
        if self.as_line :
            if self.grayscale :
                return np.reshape(batch,(batch_size,self.input_height*self.input_width))
            # return np.array(batch).astype(np.float32)[:, :, None]
            else:
                return np.reshape(batch,(batch_size,self.input_height*self.input_width*3))
            #return np.array(batch).astype(np.float32)
        else :
            return batch

