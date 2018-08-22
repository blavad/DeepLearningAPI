from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


##############
# Batch Norm #
##############

class batch_norm(object):
            # h1 = lrelu(tf.contrib.layers.batch_norm(conv2d(h0, self.df_dim*2, name='d_h1_conv'),decay=0.9,updates_collections=None,epsilon=0.00001,scale=True,scope="d_h1_conv"))
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, x, train=True):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon, scale=True, scope=self.name)


##############
# PARAMETERS #
##############
def get_filter(shape, scope_name=''):
	return tf.get_variable(scope_name+"_filter", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def get_weights(shape, scope_name=''):
        return tf.get_variable(scope_name+"_weights", shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def get_biases(shape, scope_name=''):
        return tf.get_variable(scope_name+"_bias", shape=shape, initializer=tf.constant_initializer(0.0))

##############
# ACTIVATION #
##############

def sigmoid(inputs):
        return tf.nn.sigmoid(inputs)

def tanh(inputs):
        return tf.nn.tanh(inputs)

def relu(inputs):
	return tf.nn.relu(inputs)	

def leaky_relu(inputs,leak=0.2, name = 'leaky_relu'):
        return tf.maximum(x,leak*inputs, name = name)

##########
# LAYERS #
##########
def max_pool(self, input):
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv2d(input, filter_shape, activation=tf.nn.relu, scope_name = 'conv2d'):
        with tf.variable_scope(scope_name) as scope:
                filter = get_filter(filter_shape, scope_name= scope_name)
                bias = get_biases([ filter_shape[-1] ], scope_name=scope_name)
                out = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding='SAME', name = "scope_name") + bias
                if activation :
                        out = activation(out)
        return out

def deconv2d(input_, output_shape,
             k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
             name="deconv2d", with_w=False):
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

        if with_w:
            return deconv, w, biases
        else:
            return deconv


def fully_connected(inputs, out_dim, activation=tf.nn.relu, scope_name='fc'):
        '''
        A fully connected linear layer on inputs
        '''
        with tf.variable_scope(scope_name) as scope:
                in_dim = inputs.shape[-1]
		w = get_weights([in_dim, out_dim], scope_name = scope_name)
                bias = get_biases([out_dim],scope_name=scope_name)
                out = tf.matmul(inputs, w) + bias
                if activation :
                        out = activation(out)
        return out
