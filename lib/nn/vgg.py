from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from network import Network
from ops import *

"""
File for Network class     
"""

class VGG(Network):
	"""
	Abstract class which allows to create easily new Neural Networks    
	"""
        def __init__(self, input_shape, output_shape,
                     optimizer_function = tf.train.GradientDescentOptimizer,
                     learning_rate = 1e-4, name = 'VGG', build = False):
		"""
		args :
		input_shape : shape of input data
		output_shape : shape of network output
		loss_funtion : loss function to use
                optimizer_function : optimizer function to use
		learning_rate : learning rate for optimizer
		"""
                Network.__init__(self,input_shape, output_shape,
                                 optimizer_function = optimizer_function,
				 learning_rate = learning_rate,
                                 name = name)

                self.output_placeholder = tf.placeholder(tf.float32, shape = [None]+[output_shape], name = 'Output_Placeholder')
                self.keep_proba = tf.placeholder(tf.float32,name='dropout')
                
                if build :
                        self.build()        
               
                
	def eval(self):
		"""
		Given an input, evaluate the network output for it
		"""
                c = conv2d(self.input_placeholder, [3, 3, 3, 10], scope_name='conv_1')
                print(c)
		bn = tf.layers.batch_normalization(c, training=True)
                c = conv2d(bn, [3, 3, 10, 20], scope_name='conv_2')
                print(c)
                max_p_1 = tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                print(max_p_1)
                
                c = conv2d(max_p_1, [3, 3, 20, 40], scope_name='conv_3')
                print(c)
		bn = tf.layers.batch_normalization(c, training=True)                
		c = conv2d(bn, [3, 3, 40, 80], scope_name='conv_4')
                print(c)
                max_p_2 = tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                print(max_p_2)
                #
                c = conv2d(max_p_2, [3, 3, 80, 80], scope_name='conv_5')
                print(c)
		bn = tf.layers.batch_normalization(c, training=True)
                c = conv2d(bn, [3, 3, 80, 160], scope_name='conv_6')
                print(c)
                max_p_3 = tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                print(max_p_3)
                #
                c = conv2d(max_p_3, [3, 3, 160, 160], scope_name='conv_7')
                print(c)
                bn = tf.layers.batch_normalization(c, training=True)
                c = conv2d(bn, [3, 3, 160, 160], scope_name='conv_8')
                print(c)
                max_p_4 = tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                print(max_p_4)
                #
                c = conv2d(max_p_4, [3, 3, 160, 320], scope_name='conv_9')
                print(c)
                bn = tf.layers.batch_normalization(c, training=True)
                c = conv2d(bn, [3, 3, 320, 320], scope_name='conv_10')
                print(c)
                c = conv2d(c, [3, 3, 320, 320], scope_name='conv_11')
                bn = tf.layers.batch_normalization(c, training=True)
                print(c)
                c = conv2d(bn, [3, 3, 320, 320], scope_name='conv_12')
                print(c)
                max_p_5 = tf.nn.max_pool(c, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                
                print(max_p_5)
                
                to_fu_c = max_p_5
                
                pooled_shape = to_fu_c.get_shape()
                sum_shape = pooled_shape[1].value * pooled_shape[2].value * pooled_shape[3].value 
                fc_input = tf.reshape(to_fu_c, [-1, sum_shape])
                
                fc_1 = fully_connected(fc_input, 200, scope_name="fc_1")
                
                fc_2 = fully_connected(fc_1, 100, scope_name="fc_2")
                
                #fc_1_drop = tf.nn.dropout(fc_2, self.keep_proba)
                
                outputs = fully_connected(fc_2, self.output_shape ,activation=tf.nn.sigmoid, scope_name="softmax")
                
                return outputs
                

        def loss(self):
                """
		Return the loss fonction
		"""
                with tf.name_scope('loss'):
                    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.output_placeholder,
                                                                         logits = self.outputs,
                                                                         name='entropy')
                    
                    loss = tf.reduce_mean(entropy, name = 'loss')
                    return loss

       
