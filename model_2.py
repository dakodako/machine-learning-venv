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
from random import sample
#from preprocessing import open_images, open_images_add_corruption

def open_images(filepath, padding = True, pad_size = 3):
    images = []
    #ff = glob.glob(filepath)
    #print(ff)
    for f in sorted(glob.glob(filepath)):
        # print(f)
        a = nib.load(f)
        a = a.get_data()
        # extracting the central 50 slices
        mid = int(a.shape[1]/2)
        a = a[:,mid-25:mid + 25,:]
        for i in range(a.shape[1]):
            images.append((a[:,i,:]))
    images = np.asarray(images)
    images = images.reshape(-1,images.shape[1],images.shape[2],1)
    if padding == True:
        temp = np.zeros([images.shape[0],images.shape[1] + pad_size,images.shape[2] + pad_size,1])
        temp[:,3:,3:,:] = images
        images = temp
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi)/(m - mi)
    return images
def open_images_add_corruption(filepath, padding = True, pad_size = 3):
    #print(filepath)
    images = []
    #ff = glob.glob(filepath)
    #print(ff)
    for f in sorted(glob.glob(filepath)):
        # print(f)
        a = nib.load(f)
        a = a.get_data()
        # extracting the central 50 slices
        mid = int(a.shape[1]/2)
        a = a[:,mid-25:mid + 25,:]
        for i in range(a.shape[1]):
            s = a[:,i,:]
            s_flat = np.reshape(s, (np.product(s.shape),))
            idx = sample(range(np.product(s.shape)), int(0.2*np.product(s.shape)))
            #print(idx[10:50])
            s_flat[idx] = 0
            s_downsampled = np.reshape(s_flat, s.shape)
            images.append(s_downsampled)
    images = np.asarray(images)
    images = images.reshape(-1,images.shape[1],images.shape[2],1)
    if padding == True:
        temp = np.zeros([images.shape[0],images.shape[1] + pad_size,images.shape[2] + pad_size,1])
        temp[:,3:,3:,:] = images
        images = temp
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi)/(m - mi)
    return images



#%%
filepath_X = '../Documents/MRI_data/dataset/X/*'
filepath_ground = '../Documents/MRI_data/dataset/ground/*'
filepath_test_X = '../Documents/MRI_data/dataset2/X/*'
filepath_test_ground = '../Documents/MRI_data/dataset2/ground/*'
images_X = open_images_add_corruption(filepath_X)
images_ground = open_images(filepath_ground)
images_test_X = open_images_add_corruption(filepath_test_X)
images_test_ground = open_images(filepath_test_ground)
#%%
train_X,valid_X,train_ground,valid_ground = train_test_split(images_X,images_ground,test_size=0.2,random_state=13)

print("Dataset (images) shape: {shape}".format(shape=images_X.shape))
print("Dataset (images) shape: {shape}".format(shape=images_ground.shape))
plt.figure(figsize=[5,5])
#%%
# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(train_X[0], (116,116))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(valid_X[0], (116,116))
plt.imshow(curr_img, cmap='gray')

#%%
plt.figure(figsize = [5,5])

plt.subplot(121)
curr_img = np.reshape(train_ground[0], (116,116))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(valid_ground[0], (116,116))
plt.imshow(curr_img, cmap='gray')

