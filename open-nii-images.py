#%%
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
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
%matplotlib inline

#%%

ff = glob.glob('dataset/T1/*')

print(ff)

print(len(ff))


#Now you are all set to load the 3D volumes using nibabel. 

#Note that when you load a Nifti format volume, Nibabel does not load the image array. 
#It waits until you ask for the array data. The normal way to ask for the array data is to call the get_data() method.

#Since you want the 2D slices instead of 3D, you will initialise a list in which; 
#every time you read a volume, you will iterate over all the complete 207 slices of the 3D volume and append each slice one by one in to a list.

#%%
images = []
for f in range(len(ff)):
    a = nib.load(ff[f])
    a = a.get_data()
    mid = a.shape[1]/2
    print(int(mid))
    a = a[:,43:94,:]
    for i in range(a.shape[1]):
        images.append((a[:,i,:]))
print (a.shape)

print(a.shape[1])

#extrace one slice out

a[:,0,:].shape
#%%
images = np.asarray(images)
print(images.shape[1])
images = images.reshape(-1,113,113,1)
images.shape
#%%
m = np.max(images)
mi = np.min(images)
images = (images - mi)/(m - mi)
#%%
temp = np.zeros([459,116,116,1])
temp[:,3:,3:,:] = images

images = temp
train_X,valid_X,train_ground,valid_ground = train_test_split(images,
                                                             images,
                                                             test_size=0.2,
                                                             random_state=13)

#%%
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

#%%
