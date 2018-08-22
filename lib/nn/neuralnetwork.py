from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from network import Network
from ops import *

class NeuralNetwork(Network):
        
        def __init__(self, input_shape,
                     output_shape,
                     hidden_layers = [30,10,30],
                     optimizer_function = tf.train.GradientDescentOptimizer,
		     learning_rate = 0.001,
                     name = 'nn',
                     build = False):
                
                
                Network.__init__(self,[input_shape],output_shape,
                                 optimizer_function = optimizer_function,
				 learning_rate = learning_rate,
                                 name = name)
                
                self.hidden_layers = hidden_layers

                # Additional placeholders for Supervised neural network 
                self.output_placeholder = tf.placeholder(tf.float32,
                                                         shape = [None] + [output_shape],
                                                         name = 'Output_Placeholder')                      
                self.keep_proba = tf.placeholder(tf.float32,name='dropout')
                
                if build :
                        self.build()        
                 
	def eval(self):
                fc_input = self.input_placeholder
                print(fc_input)
       	        for i, dim_layer in enumerate(self.hidden_layers) :
                        fc_input = fully_connected(fc_input, dim_layer, scope_name = 'fc_'+str(i))
                        fc_input = tf.nn.dropout(fc_input, self.keep_proba)
                fc_out = fully_connected(fc_input, self.output_shape, scope_name = 'fc_out')
                return fc_out
                        
        def loss(self):
                with tf.name_scope('loss'):
                        #loss = tf.reduce_sum(self.output_placeholder - self.outputs)
                        entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self.output_placeholder,
                                                                          logits = self.outputs,
                                                                          name='entropy')
                        #entropy =  tf.reduce_sum(self.output_placeholder * tf.log(self.outputs))
                        loss = tf.reduce_mean(entropy, name = 'loss')
                        return loss
