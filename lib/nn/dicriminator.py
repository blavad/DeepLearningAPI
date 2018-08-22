from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from network import Network
from ops import *

class Dicriminator(Network):
        
        def __init__(self, input_shape,
                     output_shape,
                     optimizer_function = tf.train.GradientDescentOptimizer,
		     learning_rate = 0.001,
                     name = 'dicrim',
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
                if build :
                        self.build()        

        def eval(self, image, y=None, reuse=False):
            
            with tf.variable_scope("discriminator") as scope:
                
                # image is 256 x 256 x (input_c_dim + output_c_dim)
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                else:
                    assert tf.get_variable_scope().reuse == False

                    h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                    # h0 is (128 x 128 x self.df_dim)
                    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                    # h1 is (64 x 64 x self.df_dim*2)
                    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                    # h2 is (32x 32 x self.df_dim*4)
                    h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
                    # h3 is (16 x 16 x self.df_dim*8)
                    h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                    return tf.nn.sigmoid(h4), h4

	def eval(self):
            with tf.variable_scope("discriminator") as scope:
                
                # image is 256 x 256 x (input_c_dim + output_c_dim)
                if reuse:
                    tf.get_variable_scope().reuse_variables()
                else:
                    assert tf.get_variable_scope().reuse == False

                    h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                    # h0 is (128 x 128 x self.df_dim)
                    h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
                    # h1 is (64 x 64 x self.df_dim*2)
                    h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
                    # h2 is (32x 32 x self.df_dim*4)
                    h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, d_h=1, d_w=1, name='d_h3_conv')))
                    # h3 is (16 x 16 x self.df_dim*8)
                    h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                    return tf.nn.sigmoid(h4), h4            
        def loss(self):
                with tf.name_scope('loss'):
                        #loss = tf.reduce_sum(self.output_placeholder - self.outputs)
                        entropy = tf.nn.softmax_cross_entropy_with_logits(labels = self.output_placeholder,
                                                                          logits = self.outputs,
                                                                          name='entropy')
                        #entropy =  tf.reduce_sum(self.output_placeholder * tf.log(self.outputs))
                        loss = tf.reduce_mean(entropy, name = 'loss')
                        return loss
