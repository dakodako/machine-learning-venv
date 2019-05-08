#%%
from keras.layers import Layer, Input, Conv2D, Activation, add, BatchNormalization, UpSampling2D, ZeroPadding2D, Conv2DTranspose, Flatten, MaxPooling2D, AveragePooling2D
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.backend import mean
from keras.models import Model, model_from_json
from keras.utils import plot_model
from keras.engine.topology import Network
#from keras.engine.topology import Container
#%%
from collections import OrderedDict
from skimage.io import imsave

import numpy as np 
import random
import datetime
import time
import json
import math
import csv
import sys
import os

import tensorflow as tf 
import keras.backend as K 

np.random.seed(seed = 12345)

class CycleGAN():
    def __init__(self, lr_D = 2e-4, lr_G = 2e-4, image_shape = (256,256), image_folder = 'p2m',date_time_string_addition=''):
        self.img_shape = image_shape
        self.channels = 1
        self.normalization = InstanceNormalization
        #hyper params
        self.lambda_1 = 10.0 # cyclic loss weight A2B
        self.lambda_2 = 10.0 # cyclic loss weight B2A
        self.lambda_D = 1.0 # weight for loss from discriminator guess on synthetic images
        self.learning_rate_D = lr_D
        self.learning_rate_G = lr_G
        self.generator_iterations = 1 # number of generator training iterations in each training loop
        self.discriminator_iterations = 1 # 
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.batch_size = 1
        self.epochs = 200
        self.save_interval = 1
        self.synthetic_pool_size = 50

        self.use_linear_decay = True
        self.decay_epoch = 101
        
        self.use_identity_learning = False
        self.identity_mapping_modulus = 10

        self.use_patchgan = True
        
        self.use_multislice_discriminator = False

        self.use_supervised_learning = False
        self.supervised_weight = 10.0

        self.use_data_generator = False

        self.REAL_LABEL = 1.0 # Use e.g. 0.9 to avoid training the discriminators to zero loss

        # Used as storage folder name
        self.date_time = time.strftime('%Y%m%d-%H%M%S', time.localtime()) + date_time_string_addition

        # optimizer
        self.opt_D = Adam(self.learning_rate_D, self.beta_1, self.beta_2)
        self.opt_G = Adam(self.learning_rate_G, self.beta_1, self.beta_2)

