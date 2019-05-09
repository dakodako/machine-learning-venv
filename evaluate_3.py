#

#%%
#import pydicom
from keras import models
from keras.models import load_model
#from pydicom.data import get_testdata_files
import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import imresize, imrotate
from skimage.transform import rotate, resize
import nibabel as nib
import glob
import sys
#%%
def pad_zero_margins2(input_img, size):
    width = input_img.shape[1]
    start = int(np.floor((size - width)/2))
    output_img = np.zeros((size,size))
    output_img[:,start:(start + width)] = input_img
    return output_img
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
model = load_model('autoencoder2_petra4.h5')
model.summary()
#%%
print(len(model.layers))
#%%
filepath_test_X = 'PETRA2/*'
filepath_test_ground = 'MP2RAGE2/*'
test_X = open_images(filepath_test_X)
test_ground = open_images(filepath_test_ground)
print(test_X.shape)
pred = model.predict(test_X)

#%%
tst_idx = [0,10,20,30,49]
#%%
fig, (ax1, ax2, ax3,ax4,ax5) = plt.subplots(figsize=(13, 3), ncols=5)
pos = ax1.imshow(np.abs(pred[tst_idx[0], ..., 0]-test_ground[tst_idx[0], ..., 0]), interpolation='none')
fig.colorbar(pos, ax=ax1)
pos = ax2.imshow(np.abs(pred[tst_idx[1], ..., 0]-test_ground[tst_idx[1], ..., 0]), interpolation='none')
fig.colorbar(pos, ax=ax2)
pos = ax3.imshow(np.abs(pred[tst_idx[2], ..., 0]-test_ground[tst_idx[2], ..., 0]), interpolation='none')
fig.colorbar(pos, ax=ax3)
pos = ax4.imshow(np.abs(pred[tst_idx[3], ..., 0]-test_ground[tst_idx[3], ..., 0]), interpolation='none')
fig.colorbar(pos, ax=ax4)
pos = ax5.imshow(np.abs(pred[tst_idx[4], ..., 0]-test_ground[tst_idx[4], ..., 0]), interpolation='none')
fig.colorbar(pos, ax=ax5)
#%%
plt.figure(figsize=(20, 4))
print("Test Images inputs")
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_X[tst_idx[i], ..., 0], cmap = 'gray')
plt.show()

plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(test_ground[tst_idx[i], ..., 0], cmap='gray')
plt.show()    

plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(pred[tst_idx[i], ..., 0], cmap='gray')  
plt.show()

plt.figure(figsize=(20, 4))
print("Error of Test Images")
for i in range(5):
    fig = plt.subplot(1, 5, i+1)
    plt.imshow(np.abs(pred[tst_idx[i], ..., 0]-test_ground[tst_idx[i], ..., 0]))  
plt.show()

#%%
print(test_X.shape)
print(test_ground.shape)
#%%
plt.figure(figsize=(20, 4))
print("Test Images inputs")
for i in range(5):
    k = i + 4
    plt.subplot(1, 5, i + 1)
    plt.imshow(test_X[k, ..., 0], cmap = 'gray')
plt.show()

plt.figure(figsize=(20, 4))
print("Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(test_ground[k, ..., 0], cmap='gray')
plt.show()    

plt.figure(figsize=(20, 4))
print("Reconstruction of Test Images")
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(pred[k, ..., 0], cmap='gray')  
plt.show()


#%% 

loss = np.mean(np.square(test_ground - pred))
print(loss)

#%%
#plt.imshow(pred[0,:,:,0])
#%%
print(np.mean(pred[3,:,:,0]))
print(np.mean(test_ground[3,:,:,0]))
print(test_ground[3,128,128,0])