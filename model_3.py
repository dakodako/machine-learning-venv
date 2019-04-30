#%%
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.layers.core import Dropout, Lambda
from keras.layers.merge import concatenate
from keras.utils import plot_model
from time import time
from keras.callbacks import TensorBoard
from skimage.transform import resize, rotate
from keras.preprocessing.image import ImageDataGenerator
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
#%%
def unet3(input_img):
	#s = Lambda(lambda x: x/255)(input_img)
	c1 = Conv2D(16,(3,3),activation = 'relu', padding = 'same')(input_img)
	c1 = Dropout(0.1)(c1) # ????
	c1 = Conv2D(16,(3,3), activation = 'relu', padding = 'same')(c1)
	p1 = MaxPooling2D((2,2), strides = (2,2))(c1)
	c2 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(p1)
	c2 = Dropout(0.1)(c2) # ????
	c2 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(c2)
	p2 = MaxPooling2D((2,2), strides = (2,2))(c2)

	c3 = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(p2)
	c3 = Dropout(0.1)(c3) # ????
	c3 = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(c3)
	p3 = MaxPooling2D((2,2), strides = (2,2))(c3)

	c4 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(p3)
	c4 = Dropout(0.1)(c4) # ????
	c4 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(c4)
	p4 = MaxPooling2D((2,2), strides = (2,2))(c4)

	c5 = Conv2D(256, (3,3),activation = 'relu', padding = 'same')(p4)
	c5 = Dropout(0.1)(c5) # ????
	c5 = Conv2D(256, (3,3),activation = 'relu', padding = 'same')(c5)

	u6 = Conv2DTranspose(256,(2,2), strides = (2,2), padding = 'same')(c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(u6)
	c6 = Dropout(0.1)(c6)
	c6 = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(c6)
	u7 = Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'valid')(c6)
	u7 = concatenate([u7,c3])
	c7 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(u7)
	c7 = Dropout(0.1)(c7)
	c7 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(c7)

	u8 = Conv2DTranspose(32, (2,2), strides = (2,2), padding = 'same')(c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(u8)
	c8 = Dropout(0.1)(c8)
	c8 = Conv2D(32, (3,3), activation = 'relu', padding = 'same')(c8)

	u9 = Conv2DTranspose(16, (2,2), strides = (2,2), padding = 'same')(c8)
	u9 = concatenate([u9, c1])
	c9 = Conv2D(16,(3,3), activation = 'relu', padding = 'same')(u9)
	c9 = Dropout(0.1)(c9)
	c9 = Conv2D(16,(3,3), activation = 'relu', padding = 'same')(c9)
	output = Conv2D(1,(1,1), activation = 'relu', padding = 'same')(c9) 
	return output
#%%
def unet2(input_img):
	#s = Lambda(lambda x: x/255)(input_img)
	c1 = Conv2D(32,(3,3),activation = 'relu', padding = 'same')(input_img)
	c1 = Dropout(0.1)(c1) # ????
	c1 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(c1)
	p1 = MaxPooling2D((2,2), strides = (2,2))(c1)
	c2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(p1)
	c2 = Dropout(0.1)(c2) # ????
	c2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(c2)
	p2 = MaxPooling2D((2,2), strides = (2,2))(c2)

	c3 = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(p2)
	c3 = Dropout(0.1)(c3) # ????
	c3 = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(c3)
	p3 = MaxPooling2D((2,2), strides = (2,2))(c3)

	c4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(p3)
	c4 = Dropout(0.1)(c4) # ????
	c4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(c4)
	p4 = MaxPooling2D((2,2), strides = (2,2))(c4)

	c5 = Conv2D(512, (3,3),activation = 'relu', padding = 'same')(p4)
	c5 = Dropout(0.1)(c5) # ????
	c5 = Conv2D(512, (3,3),activation = 'relu', padding = 'same')(c5)

	u6 = Conv2DTranspose(256,(2,2), strides = (2,2), padding = 'same')(c5)
	u6 = concatenate([u6, c4])
	c6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(u6)
	c6 = Dropout(0.1)(c6)
	c6 = Conv2D(256,(3,3), activation = 'relu', padding = 'same')(c6)
	u7 = Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'valid')(c6)
	u7 = concatenate([u7,c3])
	c7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(u7)
	c7 = Dropout(0.1)(c7)
	c7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(c7)

	u8 = Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same')(c7)
	u8 = concatenate([u8, c2])
	c8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(u8)
	c8 = Dropout(0.1)(c8)
	c8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(c8)

	u9 = Conv2DTranspose(32, (2,2), strides = (2,2), padding = 'same')(c8)
	u9 = concatenate([u9, c1])
	c9 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(u9)
	c9 = Dropout(0.1)(c9)
	c9 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(c9)
	output = Conv2D(1,(1,1), activation = 'relu', padding = 'same')(c9) 
	return output
#%%
def autoencoder2(input_img):
	#s = Lambda(lambda x: x/255)(input_img)
	c1 = Conv2D(32,(3,3),activation = 'relu', padding = 'same')(input_img)
	c1 = Dropout(0.1)(c1) # ????
	c1 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(c1)
	p1 = MaxPooling2D((2,2), strides = (2,2))(c1)
	c2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(p1)
	c2 = Dropout(0.1)(c2) # ????
	c2 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(c2)
	p2 = MaxPooling2D((2,2), strides = (2,2))(c2)

	c3 = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(p2)
	c3 = Dropout(0.1)(c3) # ????
	c3 = Conv2D(128,(3,3), activation = 'relu', padding = 'same')(c3)
	p3 = MaxPooling2D((2,2), strides = (2,2))(c3)

	c4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(p3)
	c4 = Dropout(0.1)(c4) # ????
	c4 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(c4)
	p4 = MaxPooling2D((2,2), strides = (2,2))(c4)

	c5 = Conv2D(512, (3,3),activation = 'relu', padding = 'same')(p4)
	c5 = Dropout(0.1)(c5) # ????
	c5 = Conv2D(512, (3,3),activation = 'relu', padding = 'same')(c5)

	u6 = Conv2DTranspose(512,(2,2), strides = (2,2), padding = 'same')(c5)
	#u6 = concatenate([u6, c4])
	c6 = Conv2D(256, (3,3), activation = 'relu', padding = 'same')(u6)
	c6 = Dropout(0.1)(c6)
	c6 = Conv2D(256,(3,3), activation = 'relu', padding = 'same')(c6)
	u7 = Conv2DTranspose(256, (2,2), strides = (2,2), padding = 'valid')(c6)
	#u7 = concatenate([u7,c3])
	c7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(u7)
	c7 = Dropout(0.1)(c7)
	c7 = Conv2D(128, (3,3), activation = 'relu', padding = 'same')(c7)

	u8 = Conv2DTranspose(128, (2,2), strides = (2,2), padding = 'same')(c7)
	#u8 = concatenate([u8, c2])
	c8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(u8)
	c8 = Dropout(0.1)(c8)
	c8 = Conv2D(64, (3,3), activation = 'relu', padding = 'same')(c8)

	u9 = Conv2DTranspose(64, (2,2), strides = (2,2), padding = 'same')(c8)
	#u9 = concatenate([u9, c1])
	c9 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(u9)
	c9 = Dropout(0.1)(c9)
	c9 = Conv2D(32,(3,3), activation = 'relu', padding = 'same')(c9)
	output = Conv2D(1,(1,1), activation = 'relu', padding = 'same')(c9) 
	return output
#%%
def pad_zero_margins(input_img):
    output_img = np.zeros((256,256))
    output_img[:,31:224] = input_img
    return output_img
def pad_zero_margins2(input_img, size):
    width = input_img.shape[1]
    start = int(np.floor((size - width)/2))
    output_img = np.zeros((size,size))
    output_img[:,start:(start + width)] = input_img
    return output_img
#%%
def open_images2(filepath):
    images = []
    for f in sorted(glob.glob(filepath)):
        a = nib.load(f)
        a = a.get_data()
        mid = int(a.shape[2]/2)
        a = a[:,:,mid-25:mid+25]
        for i in range(a.shape[2]):
            temp = rotate(a[:,:,i],90, resize = True)
            temp = pad_zero_margins2(temp,136)
            temp = resize(temp, (256,256))
            images.append(temp)
    images = np.asarray(images)
    images = images.reshape(-1,images.shape[1],images.shape[2],1)
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi)/(m - mi)
    return images
def open_images(filepath):
    images = []
    for f in sorted(glob.glob(filepath)):
        b = nib.load(f)
        b = b.get_data()
        mid = int(b.shape[2]/2)
        b = b[:,:,mid-25:mid + 25]
        for i in range(b.shape[2]):
            temp = b[:,:,i]
            temp = np.reshape(temp,[b.shape[0],b.shape[1]])
            temp = pad_zero_margins2(temp, 256)
            images.append(temp)
    images = np.asarray(images)
    images = images.reshape(-1,images.shape[1],images.shape[2],1)
    m = np.max(images)
    mi = np.min(images)
    images = (images - mi)/(m - mi)
    return images



#%%

filepath_X = sys.argv[1]
filepath_ground = sys.argv[2]
#filepath_X = 'PETRA2/*'
#filepath_ground = 'MP2RAGE2/*'
images_X = open_images(filepath_X)
images_ground = open_images(filepath_ground)
print(images_X.shape)

#%%
train_X,valid_X,train_ground,valid_ground = train_test_split(images_X,images_ground,test_size=0.2,random_state=13)

print("Dataset (images) shape: {shape}".format(shape=images_X.shape))
print("Dataset (images) shape: {shape}".format(shape=images_ground.shape))
plt.figure(figsize=[5,5])

#%%
datagen = ImageDataGenerator(
	rotation_range = 90,
	width_shift_range = 0.2,
	height_shift_range = 0.2,
	shear_range = 0.2,
	zoom_range = 0.1,
	horizontal_flip = True,
	fill_mode = 'nearest'
)
#%%
# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(train_X[0], (256,256))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(valid_X[0], (256,256))
plt.imshow(curr_img, cmap='gray')

#%%
plt.figure(figsize = [5,5])

plt.subplot(121)
curr_img = np.reshape(train_ground[0], (256,256))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(valid_ground[0], (256,256))
plt.imshow(curr_img, cmap='gray')

#%%

batch_size = 2
epochs = 10
inChannel = 1
x, y = 256, 256
input_img = Input(shape = (x,y,inChannel))
autoencoder = Model(input_img, unet2(input_img))
rmsprop = optimizers.RMSprop(lr = 0.005)
autoencoder.compile(loss='mean_squared_error', optimizer = rmsprop)
autoencoder.summary()
#%%
tensorboard = TensorBoard(log_dir="autoencoder2_data_aug_logs/{}".format(time()))
autoencoder.fit_generator(datagen.flow(train_X, train_ground, batch_size = batch_size),steps_per_epoch =300, epochs = epochs,validation_data = datagen.flow(valid_X, valid_ground, batch_size = 1),validation_steps = 170, callbacks=[tensorboard])
#autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground), callbacks=[tensorboard])
#loss = autoencoder_train.history['loss']
#val_loss = autoencoder_train.history['val_loss']
autoencoder.save('autoencoder2_petra_data_aug.h5')
'''
#%%
filepath_test_X = sys.argv[3]#'../Documents/MRI_data/dataset/X/*'#sys.argv[3]
filepath_test_ground = sys.argv[4]
test_X = open_images(filepath_test_X)
test_ground = open_images(filepath_test_ground)

pred = autoencoder.predict(test_X)
pda lt.figure(figsize=(20, 4))
print("Test Images inputs")
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_X[i, ..., 0], cmap = 'gray')
plt.show()

plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(test_ground[i, ..., 0], cmap='gray')
plt.show()    

plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')  
plt.show()

'''

