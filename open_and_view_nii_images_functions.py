#%%
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
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

filepath = 'dataset/MRI_data/dataset/X/*'
filename = '/Users/chid/machine-learning-venv/dataset/T1/943862_T1w_restore.1.60.nii.gz'
#%%
def open_image(filename, padding = True, pad_size = 3):
    image = []
    a = nib.load(filename)
    a = a.get_data()
    mid = int(a.shape[1]/2)
    a = a[:,mid-25:mid + 25,:]
    for i in range(a.shape[1]):
            image.append((a[:,i,:]))
    image = np.asarray(image)
    image = image.reshape(-1,image.shape[1],image.shape[2],1)
    if padding == True:
        temp = np.zeros([image.shape[0],image.shape[1] + pad_size,image.shape[2] +pad_size,1])
        temp[:,pad_size:,pad_size:,:] = image
        image = temp
    m = np.max(image)
    mi = np.min(image)
    image = (image - mi)/(m - mi)
    return image
def extract_a_slice(index, volume):
    s = volume[index,:,:,:]
    return s
def open_images(filepath, padding = True, pad_size = 3):
    images = []
    ff = glob.glob(filepath)
    print(ff)
    for f in range(len(ff)):
        a = nib.load(ff[f])
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

#%%
images = open_images(filepath)
images.shape
#images = open_image(filename)
#s = extract_a_slice(1, images)
#print(images.shape)
#print(s.shape)
#plt.figure(figsize = [5,5])
#curr_img = np.reshape(s, (113,113))
#plt.imshow(curr_img, cmap='gray')
#plt.show()

'''
images = open_images(filepath)
temp = np.zeros([images.shape[0],images.shape[1] + 3,images.shape[2] + 3,1])
temp[:,3:,3:,:] = images

images = temp
train_X,valid_X,train_ground,valid_ground = train_test_split(images,
                                                             images,
                                                             test_size=0.2,
                                                             random_state=13)


print("Dataset (images) shape: {shape}".format(shape=images.shape))
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(train_X[0], (116,116))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(valid_X[0], (116,116))
plt.imshow(curr_img, cmap='gray')

plt.show()
'''

