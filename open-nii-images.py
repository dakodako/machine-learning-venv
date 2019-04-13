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
#%matplotlib inline

#%%

ff = glob.glob('dataset/T1/*')





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
    a = a[:,43:94,:]
    for i in range(a.shape[1]):
        images.append((a[:,i,:]))




#extrace one slice out

a[:,0,:].shape
#%%
images = np.asarray(images)
images = images.reshape(-1,113,113,1)
images.shape
#%%
m = np.max(images)
mi = np.min(images)
images = (images - mi)/(m - mi)
#%%
temp = np.zeros([12*51,116,116,1])
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

# the convolutional autoencoder

batch_size = 40
epochs = 1
inChannel = 1
x, y = 116, 116
input_img = Input(shape = (x, y, inChannel))

# there are two parts in the autoencoder: encoder and decoder
#%%
def unet(input_img):
	s = Lambda(lambda x: x/255)(input_img)
	c1 = Conv2D(64,(3,3),activation = 'relu')(s)
	c1 = Dropout(0.1)(c1) # ????
	c1 = Conv2D(64,(3,3), activation = 'relu')(c1)
	p1 = MaxPooling2D((2,2), strides = (2,2))(c1)
	c2 = Conv2D(128, (3,3), activation = 'relu')(p1)
	c2 = Dropout(0.1)(c2) # ????
	c2 = Conv2D(128, (3,3))(c2)
	p2 = MaxPooling2D((2,2), strides = (2,2))(c2)

	c3 = Conv2D(256,(3,3), activation = 'relu')(p2)
	c3 = Dropout(0.1)(c3) # ????
	c3 = Conv2D(256,(3,3), activation = 'relu')(c3)
	p3 = MaxPooling2D((2,2), strides = (2,2))(c3)

	c4 = Conv2D(512, (3,3), activation = 'relu')(p3)
	c4 = Dropout(0.1)(c4) # ????
	c4 = Conv2D(512, (3,3), activation = 'relu')(c4)
	p4 = MaxPooling2D((2,2), strides = (2,2))(c3)

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
	return output
#%% encoder

# the encoder has three convolution layers
# each convolution layer is followed by a batch normalization layer
# max-pooling layer is used after the first and second convolution blocks
def autoencoder(input_img):
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
	conv1 = BatchNormalization()(conv1)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	conv1 = BatchNormalization()(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) 
	conv2 = BatchNormalization()(conv2)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	conv2 = BatchNormalization()(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) 
	conv3 = BatchNormalization()(conv3)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	conv3 = BatchNormalization()(conv3)


#%%decoder
	conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv3) 
	conv4 = BatchNormalization()(conv4)
	conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
	conv4 = BatchNormalization()(conv4)
	up1 = UpSampling2D((2,2))(conv4) 
	conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) 
	conv5 = BatchNormalization()(conv5)
	conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
	conv5 = BatchNormalization()(conv5)
	up2 = UpSampling2D((2,2))(conv5) 
	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) 
	return decoded


#%%
#autoencoder = Model(input_img, autoencoder(input_img))
autoencoder = Model(input_img, unet(input_img))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())

autoencoder.summary()


#autoencoder_train = autoencoder.fit(train_X, train_ground, batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, valid_ground))

#loss = autoencoder_train.history['loss']
#val_loss = autoencoder_train.history['val_loss']
'''
epochs = range(1)
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#autoencoder = autoencoder.save_weights('autoencoder_mri.h5')
#autoencoder = Model(input_img, autoencoder(input_img))
#autoencoder.load_weights('autoencoder_mri.h5')

pred = autoencoder.predict(valid_X)

plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(valid_ground[i, ..., 0], cmap='gray')
plt.show()    
plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(pred[i, ..., 0], cmap='gray')  
plt.show()
'''


