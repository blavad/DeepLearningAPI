from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os

"""
File for Model class
"""

class Model(object):
	"""
	Abstract class which allows to create and train easily new Neural Networks Models     
	"""
        def __init__(self, sess, networks, dataset ,checkpoint_dir='./checkpoint', graph_dir = './graphs/', model_name = 'network_model',build = False):
		"""
		args :
		input_shape : shape of input data
                output_shape : shape of network output
	        loss_funtion : loss function to use
		optimizer : optimizer to use
		learning_rate : learning rate for optimizer
		"""
                self.sess = sess
		self.name = model_name

                self.net = networks
                        
                self.dataset = dataset
                
                self.sess.run(tf.global_variables_initializer())
        
                self.saver = tf.train.Saver()
                
                self.checkpoint_dir = checkpoint_dir
                self.is_built = False
                self.writer = tf.summary.FileWriter(graph_dir+self.name, self.sess.graph)
                
        """
        def build(self):
        """
        #Built graph if it is not ever done
        """
                print("(*) Building {}".format(self))
                if self.is_built :
                        print("(!): "+self.name + " is already built")
                else :
                        for id, n in enumerate(self.net) :
                                n.build()
                        self.is_built = True
        """
                
        def getNbBatches(self, batch_size):
                """
                Return number of batches to train over all the dataset
                """
                return int(self.dataset.num_examples/batch_size)
                                
        def train(self, config):
	        """
		Train the model
		"""
                raise NotImplementedError("(!) You must implement %s method" % type(self).__name__)
        
        def test(self):
                """
		Test the model
		"""
                raise NotImplementedError("(!) You must implement %s method" % type(self).__name__)
         

        def runOLS(self):
                """
		Return optimizer,
		"""
                raise NotImplementedError("(!) You must implement %s method" % type(self).__name__)
                
        
        def add_to_summary(self, summary_op, global_step):
                self.writer.add_summary(summary_op, global_step = global_step)

                
        def save(self, checkpoint_dir, step):
                print("(*) Saving checkpoints...")
                checkpoint_dir = os.path.join(checkpoint_dir, self.name)
                
                if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                self.saver.save(self.sess,
                                os.path.join(checkpoint_dir, self.name+'.ckpt'),
                                global_step=step)
        
        def load(self, checkpoint_dir):
                import re
                print("(*) Reading checkpoints...")
                checkpoint_dir = os.path.join(checkpoint_dir, self.name)
                if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
                        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
                        counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
                        print("(*) Success to read {}".format(ckpt_name))
                        return True, counter
                else:
                        print("(!) Failed to find a checkpoint")
                        return False, 0


        def __str__(self):
                return "Abstract Model of Neural Network poblem <{}>".format(self.name)
