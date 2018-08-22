import sys

sys.path.append("/home/david/Bureau/Documents/Travail/GAN_Dundee/GAN/project/lib/") 
sys.path.append("/home/david/Bureau/Documents/Travail/GAN_Dundee/GAN/project/lib/dataset/") 
sys.path.append("/home/david/Bureau/Documents/Travail/GAN_Dundee/GAN/project/lib/nn/")
sys.path.append("/home/david/Bureau/Documents/Travail/GAN_Dundee/GAN/project/lib/nnmodel/") 

import os
import re
import pprint as pp
import numpy as np
import tensorflow as tf

from glob import glob
from imagedataset import ImageDataset
from label import Label
from labeleddataset import LabeledDataset

from neuralnetwork import NeuralNetwork as NN
from classifier1d import Classifier1D 

flags = tf.app.flags
flags.DEFINE_integer("epoch", 10, "Epoch to train ")
flags.DEFINE_float("learning_rate", 0.002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("dropout", 1.0, "Dropout [0.5]")
#flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
#flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 12, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 80, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", 60, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_string("dataset", "yugioh", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("model_name", "YuGiOh_Classifier", "The name of model (and checkpoint dir)")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoints", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "./data/", "Root directory of dataset [data]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", True, "True for visualizing, False for nothing [False]")
flags.DEFINE_integer("skip_step", 3, "Number of step without display training parameters & save chackpoint . [108]")
FLAGS = flags.FLAGS

    
def main(_):
    pp.pprint(flags.FLAGS.__flags)

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    #run_config.gpu_options.allow_growth=True
    
    
    # Data initialisation
    data_path = os.path.join(FLAGS.data_dir, FLAGS.dataset, FLAGS.input_fname_pattern)
    data = ImageDataset(regex_file = data_path, input_height = FLAGS.input_height, input_width = FLAGS.input_width)
       
    with tf.Session() as sess :
        
        net = NN(FLAGS.input_height*FLAGS.input_width*3, 2, hidden_layers =  [60,10, 30],
                 learning_rate = FLAGS.learning_rate, build = True, name = 'FCNeuralNet')
        model = Classifier1D(sess, [net] ,dataset,
                             checkpoint_dir= FLAGS.checkpoint_dir,
                             model_name = FLAGS.model_name)
        
        old, step = model.load(FLAGS.checkpoint_dir)
        print("(*) Already train : {} {}".format(old, ("--> step "+str(step) if old else "")))
        
        if FLAGS.train:
            model.train(FLAGS)
        else:
            if not model.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("(!) Train a model first, then run test mode")
        
        
        
if __name__ == '__main__':
    tf.app.run()
