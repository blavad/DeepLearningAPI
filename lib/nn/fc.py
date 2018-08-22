from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from neuralnetwork import NeuralNetwork

from ops import fully_connected

class FullyConnected(NeuralNetwork):
        
        def __init__(self, input_length, output_length, hidden_layers = [30,10,30], learning_rate = 0.01,name = 'fcn', build = False): 

                NeuralNetwork.__init__(self,input_length,output_length,
			               learning_rate = learning_rate,
                                       name = name,
                                       build = build)
                self.keep_proba = None
                if build:
                        self.build()
