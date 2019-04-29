#

#%%
import pydicom
from keras import models
from keras.models import load_model
from pydicom.data import get_testdata_files
import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import imresize, imrotate
from skimage.transform import rotate, resize
import nibabel as nib
import glob
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
model = load_model('autoencoder2_petra.h5')
model.summary()

#%%
filepath_test_X = 'PETRA2/*'#sys.argv[3]
filepath_test_ground = 'MP2RAGE2/*'#sys.argv[4]
test_X = open_images(filepath_test_X)
test_ground = open_images(filepath_test_ground)

#%%
pred = model.predict(test_X)
print(pred.shape)
#%%
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