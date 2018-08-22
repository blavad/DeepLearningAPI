import tensorflow as tf
import numpy as np

"""
File for Network class     
"""

class Network(object):
	"""
	Abstract class which allows to create easily new Neural Networks    
	"""
        def __init__(self, input_shape, output_shape,
                     optimizer_function = tf.train.AdamOptimizer,
                     learning_rate = 1e-4, name = 'net'):
		"""
		args :
		input_shape : shape of input data
		output_shape : shape of network output
		loss_funtion : loss function to use
                optimizer_function : optimizer function to use
		learning_rate : learning rate for optimizer
		"""

                self.name = name

                self.input_placeholder = tf.placeholder(tf.float32, shape = [None]+input_shape, name = 'Input_Placeholder')
		self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='Global_Step')

                self.input_shape = input_shape
                self.output_shape = output_shape
                
                # HyperParatmers
                self.optimizer_function = optimizer_function
                self.learning_rate = learning_rate
                
                # This class is abstract and so can't be built
                self.is_built = False

                
	def eval(self):
		"""
		Given an input, evaluate the network output for it
		"""
                raise NotImplementedError("(!) You must implement %s method" % type(self).__name__)

        def loss(self):
                """
		Return the loss fonction
		"""
                raise NotImplementedError("(!) You must implement %s method" % type(self).__name__)

        def optimizer(self):
                """
		Return the optimizer
		"""
                return self.optimizer_function(self.learning_rate).minimize(self.loss, global_step=self.global_step)               

	def summary(self):
                """
                Return summaries to write on TensorBoard
                """
                with tf.name_scope('summary'):
                        tf.summary.scalar('loss', self.loss)
                        #tf.summary.scalar('accuracy', self.accuracy)
                        tf.summary.histogram('histogram loss', self.loss)
          	        summary_op = tf.summary.merge_all()
		return summary_op

        def build(self):
                """
                Built graph if it is not ever done
                """
                print("(*) Building {0}".format(self)+" ...")
                if self.is_built :
                        print("(!) " + self.name + " is already built")
                else :
                        
                        self.outputs = self.eval()
                        self.loss = self.loss()
                        self.optimizer = self.optimizer()
                        self.summary_op = self.summary()
                        self.is_built = True
        
        def getOptimizer(self):
		"""
                Return optimizer
                """
                if not self.is_built :
                        print("(!) {0} {1} is not built".format(self, self.name))
	        else:
                        return self.optimizer

        def getLoss(self):
		"""
                Return loss
                """
                if not self.is_built :
                        print("(!) {0} {1} is not built".format(self, self.name))
	        else:
                        return self.loss
        
        def getSummary(self):
		"""
                Return summary
                """
                if not self.is_built :
                        print("(!) {0} {1} is not built".format(self, self.name))
	        else:
                        return self.summary_op

        def __str__(self):
                return "Abstract Neural Network <{}>".format(self.name)
