import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

# U-Net structure replicating what is exactly described in the u-net paper

inputs = Input((IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))
s = Lambda(lambda x: x/255)(inputs)
c1 = Conv2D(64,(3,3),activation = 'relu')(s)
# c1 = Dropout(0.1)(c1) # ????
c1 = Conv2D(64,(3,3), activation = 'relu')(c1)
p1 = MaxPooling2D((2,2), strides = (2,2))(c1)
c2 = Conv2D(128, (3,3), activation = 'relu')(p1)
# c2 = Dropout(0.1)(c2) # ????
c2 = Conv2D(128, (3,3))(c2)
p2 = MaxPooling2D((2,2), strides = (2,2))(c2)

c3 = Conv2D(256,(3,3), activation = 'relu')(p2)
c3 = Dropout(0.1)(c3) # ????
c3 = Conv2D(256,(3,3), activation = 'relu')(c3)
p3 = MaxPooling2D((2,2), strides = (2,2))(c3)

c4 = Conv2D(512, (3,3), activation = 'relu')(p3)
c4 = Dropout(0.1)(c4) # ????
c4 = Conv2D(512, (3,3), activation = 'relu')(c4)
p4 = MaxPooling2D((2,2), strides = (2,2))(c4)

c5 = Conv2D(1024, (3,3),activation = 'relu')(p4)
c5 = Dropout(0.1)(c5) # ????
c5 = Conv2D(1024, (3,3),activation = 'relu')(c5)

u6 = Conv2DTranspose(512,(2,2))(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(512, (3,3), activation = 'relu')(u6)
c6 = Dropout(0.1)(c6)
c6 = Conv2D(512,(3,3), activation = 'relu')(c6)

u7 = Conv2DTranspose(256, (2,2))(c6)
u7 = concatenate([u7,p3])
c7 = Conv2D(256, (3,3), activation = 'relu')(u7)
c7 = Dropout(0.1)(c7)
c7 = Conv2D(256, (3,3), activation = 'relu')(c7)

u8 = Conv2DTranspose(128, (2,2))(c7)
u8 = concatenate([u8, p2])
c8 = Conv2D(128, (3,3), activation = 'relu')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(128, (3,3), activation = 'relu')(c8)

u9 = Conv2DTranspose(64, (2,2))(c8)
u9 = concatenate([u9, p1])
c9 = Conv2D(64,(3,3), activation = 'relu')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(64,(3,3), activation = 'relu')(c9)
output = Conv2D(2,(1,1), activation = 'relu')(c9) 