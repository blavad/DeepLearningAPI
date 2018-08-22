from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
from model import Model


"""
File for 1DModel class
"""

class Classifier1D(Model):
	"""
	Abstract class which allows to create and train easily new 1 Dimension Neural Networks Models
	"""
        def __init__(self, sess, network, dataset, checkpoint_dir='./checkpoint', graph_dir = './graphs/', model_name = 'Classifier_1D', build = False):
	        """
	        args :
	        sess : 
                network :
                chekpoint_dir :
                graph_dir :
                model_name :
                """
                Model.__init__(self, sess, network, dataset,
                               checkpoint_dir=checkpoint_dir,
                               graph_dir = graph_dir,
                               model_name = model_name, build = build)
                self.network = self.net[0]
                print(tf.trainable_variables())
        
        def train(self, config):
	        """
	        Train the model
	        """
                self.sess.run(tf.global_variables_initializer())

                counter = 1
                could_load, checkpoint_counter = self.load(config.checkpoint_dir)
                if could_load:
                        counter = checkpoint_counter

                print(counter)
                
                n_batches = self.getNbBatches(config.batch_size)

                start_time = time.time()
                print("Algo time start : "+time.asctime(time.localtime(start_time)))

                total_loss = 0.0
                for epoch in range(config.epoch):
                        self.dataset.shuffle()
                        for step in range(n_batches):
                                X_batch, Y_batch = self.dataset.getNextBatch(config.batch_size, step)
                                _, loss_batch, outputs, summary = self.sess.run([self.network.optimizer,
                                                                                 self.network.loss,
                                                                                 self.network.outputs,
                                                                                 self.network.summary_op], 
                                                                feed_dict={
								        self.network.input_placeholder : X_batch,
                                                                        self.network.output_placeholder : Y_batch,
                                                                        self.network.keep_proba: config.dropout
                                                                })
                                
                                self.add_to_summary(summary, global_step=counter)
                                total_loss += loss_batch
                                if counter % config.skip_step == 0:
                                        
                                        print(Y_batch)
                                        print("--------------")
                                        print(outputs)
                                        print('(*) Epoch: [{}/{}] [{}/{}], loss {:5.3f}'
                                              .format(epoch,
                                                      config.epoch,
                                                      step,
                                                      n_batches,
                                                      total_loss / config.skip_step))
                                        print("\t--> Total time: {0} seconds".format(time.time() - start_time))
                                        total_loss = 0.0
                                counter += 1

                        self.save(self.checkpoint_dir, counter)
                                        
                                        
                print("Optimization Finished!") # should be around 0.35 after 25 epochs
                print("Total time: {0} seconds".format(time.time() - start_time))

        def test(self, config):
                """
	        Test the model
	        """
                n_batches = self.getNbBatches(config.batch_size)
                
                total_correct = 0
                for i in range(n_batches):
                        X_batch, Y_batch = self.dataset.getNextBatch(config.batch_size, i)
                        _, loss_batch, outputs = self.sess.run([self.network.optimizer,
                                                                         self.network.loss,
                                                                         self.network.outputs],
                                                                        feed_dict={
								        self.network.input_placeholder : X_batch,
                                                                        self.network.output_placeholder : Y_batch,
                                                                        self.network.keep_proba: 1.0
                                                                        })
                        preds = tf.nn.softmax(outputs)
                        correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(Y_batch,1))
                        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
                        total_correct += self.sess.run(accuracy)
			print("p ({}) / l ({})  -- ( {}/{} )".format(tf.argmax(preds,1).eval(),tf.argmax(Y_batch,1).eval(),total_correct,i+1))
                print("Accuracy {0}".format(total_correct/(n_batches*config.batch_size)))
                                                                
        

            
        def __str__(self):
                return "Abstract Model of 1 Neural Network problem <{}>".format(self.name)
