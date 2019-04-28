#%%
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras.utils import plot_model
from time import time
from keras.callbacks import TensorBoard
#from tensorflow.python.keras.callbacks import TensorBoard
#%%
import os
import numpy as np
import scipy.misc
import numpy.random as rng 
from PIL import Image, ImageDraw, ImageFont
from sklearn.utils import shuffle
import nibabel as nib # python library for reading MR images
from sklearn.model_selection import train_test_split

import math
import glob
from matplotlib import pyplot as plt
import sys
def pad_zero_margins(input_img):
    output_img = np.zeros((256,256))
    output_img[:,31:224] = input_img
    return output_img

def open_images(filepath):
    images = []
    for f in sorted(glob.glob(filepath)):
        b = nib.load(f)
        b = b.get_data()
        mid = int(b.shape[1]/2)
        b = b[:,:,mid-25:mid + 25]
        for i in range(b.shape[2]):
            temp = b[:,:,i]
            temp = np.reshape(temp,[b.shape[0],b.shape[1]])
            temp = pad_zero_margins(temp)
            images.append(temp)
    images = np.asarray(images)
    images = images.reshape(-1,images.shape[1],images.shape[2],1)
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi)/(m - mi)
    return images


filepath_X = sys.argv[1]
filepath_ground = sys.argv[2]
images_X = open_images(filepath_X)