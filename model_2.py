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
	u7 = Conv2DTranspose(128, (3,3), strides = (2,2), padding = 'valid')(c6)
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
	u7 = Conv2DTranspose(256, (3,3), strides = (2,2), padding = 'valid')(c6)
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
            idx = sample(range(np.product(s.shape)), int(0.5*np.product(s.shape)))
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
filepath_X = '../MRI_data/MRI_data/denoise/dataset/X/*' # for ubuntu
filepath_ground = '../MRI_data/MRI_data/denoise/dataset/ground/*' # for ubuntu
filepath_test_X = '../MRI_data/MRI_data/denoise/dataset2/X/*' # for ubuntu
filepath_test_ground = '../MRI_data/MRI_data/denoise/dataset2/ground/*' # for ubuntu
#filepath_X = '../Documents/MRI_data/dataset/X/*' # for macos
#filepath_ground = '../Documents/MRI_data/dataset/ground/*' # for macos
#filepath_test_X = '../Documents/MRI_data/dataset2/X/*' # for macos
#filepath_test_ground = '../Documents/MRI_data/dataset2/ground/*' # for macos
#images_X = open_images_add_corruption(filepath_X)
images_X = open_images_add_corruption(filepath_X)
images_ground = open_images(filepath_ground)
#images_test_X = open_images_add_corruption(filepath_test_X)
#images_test_ground = open_images(filepath_test_ground)
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

#plt.show()
#%%
plt.figure(figsize = [5,5])

plt.subplot(121)
curr_img = np.reshape(train_ground[0], (116,116))
plt.imshow(curr_img, cmap='gray')

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(valid_ground[0], (116,116))
plt.imshow(curr_img, cmap='gray')

#plt.show()

batch_size = 10
epochs = 100
inChannel = 1
x, y = 116, 116
input_img = Input(shape = (x, y, inChannel))



#%%
autoencoder = Model(input_img, unet2(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()

#%%
tensorboard = TensorBoard(log_dir="denoise_logs/{}".format(time()))
autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground), callbacks=[tensorboard])
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
#acc = autoencoder_train.history['acc']
#val_acc = autoencoder_train.history['val_acc']
epochs = range(100)
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
#plt.plot(epochs, acc, 'r+', label= 'Training accuracy')
#plt.plot(epochs, val_acc, 'r', label = 'Validation accuracy')
plt.title('Training and validation loss')
plt.legend()
plt.show()
#%%
test_X = open_images_add_corruption(filepath_test_X)
test_ground = open_images(filepath_test_ground)

#%%
pred = autoencoder.predict(test_X)
plt.figure(figsize=(20, 4))
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







