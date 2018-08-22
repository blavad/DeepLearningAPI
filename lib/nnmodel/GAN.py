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

class pix2pix(Model):
	"""
	Abstract class which allows to create and train easily new Generative Adversarial Network Models
	"""
        def __init__(self, sess, networks, dataset, checkpoint_dir='./checkpoint', graph_dir = './graphs/', model_name = 'Pix2pix', build = False):
	        """
	        args :
	        sess : 
                network :
                chekpoint_dir :
                graph_dir :
                model_name :
                """
                Model.__init__(self, sess, networks, dataset,
                               checkpoint_dir=checkpoint_dir,
                               graph_dir = graph_dir,
                               model_name = model_name, build = build)
                self.generator = self.net[0]
                self.discriminator = self.net[1]
                
                
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
                                X_batch = self.dataset.getNextBatch(config.batch_size, step)
                                X_batch_d =
                                X_batch_
                                _,  summary = self.sess.run([self.discriminator.optimizer,
                                                             self.discriminator.summary_op], 
                                                            feed_dict={
								    self.discriminator.input_placeholder : X_batch,
                                                                    self.dicriminator.keep_proba: config.dropout
                                                                })
                                self.add_to_summary(summary, global_step=counter)
                                
                                _,  summary = self.sess.run([self.generator.optimizer,
                                                             self.generator.summary_op], 
                                                            feed_dict={
								    self.generator.input_placeholder : X_batch,
                                                                    self.generator.keep_proba: config.dropout
                                                                })
                                self.add_to_summary(summary, global_step=counter)
                                _,  summary = self.sess.run([self.generator.optimizer,
                                                             self.generator.summary_op], 
                                                            feed_dict={
								    self.generator.input_placeholder : X_batch,
                                                                    self.generator.keep_proba: config.dropout
                                                                })
                                self.add_to_summary(summary, global_step=counter)
                                
                                counter += 1
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
                        _, su = self.sess.run([self.network.optimizer,
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
                print("Accuracy {0}".format(total_correct/(n_batches*config.batch_size)))
                                                                
        

            
        def __str__(self):
                return "Abstract Model of 1 Neural Network problem <{}>".format(self.name)
