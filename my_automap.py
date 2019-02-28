#%%
import numpy as np 
import os

from matplotlib import pyplot as plt 
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import open_and_view_nii_images_functions
from open_and_view_nii_images_functions import extract_a_slice, open_image, open_images
#%%
filepath = '/Users/chid/machine-learning-venv/dataset/T1/*'
filename = '/Users/chid/machine-learning-venv/dataset/T1/943862_T1w_restore.1.60.nii.gz'
padding = True
pad_size = 3
img = open_image(filename, padding, pad_size)

print(img.shape)

s = extract_a_slice(0,img)
s = s.reshape((s.shape[0],s.shape[1]))
print(s.shape)
#%%
def to_freq_space_volume(volume):
    # a volume has a shape of [number of slices, x, y, channels]
    v_freq = []
    return v_freq

def to_freq_space_2d(img):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real ans imaginary part
    """
    
    img_f = np.fft.fft2(img)  # FFT
    img_fshift = np.fft.fftshift(img_f)  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_real_imag
#%%
img_freq = to_freq_space_2d(s)
print(img_freq.shape)

#%%

print(np.max(img_freq[:,:,0]))
print(np.max(img_freq[:,:,1]))
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img_real = np.reshape(img_freq[:,:,0], (img_freq.shape[0],img_freq.shape[1]))
plt.imshow(curr_img_real, cmap='gray')

plt.subplot(122)
curr_img_real = np.reshape(img_freq[:,:,1], (img_freq.shape[0],img_freq.shape[1]))
plt.imshow(curr_img_real, cmap='gray')
plt.show()

#%%

#if normalize:
#    x = x - np.mean(x)

images = open_images(filepath)
print(images.shape)
imgs_freq = []
for i in range(images.shape[0]):
    s = images[i,:,:,:]
    s = s.reshape([s.shape[0],s.shape[1]])
    s_freq = to_freq_space_2d(s)
    imgs_freq.append((s_freq))

imgs_freq = np.asarray(imgs_freq)
print(imgs_freq.shape) #(612, 113, 113, 2)

#%%
n = imgs_freq.shape[1]
inChannel = 2
m1 = 32
m2 = 64
input_img = Input(shape = (n*n*inChannel,)) #2*n^2
#%%
# simulating Fourier Transform: mapping from sensor domain to the image domain
fc1 = Dense(n*n, activation = 'tanh')(input_img)
fc2 = Dense(n*n, activation = 'tanh')(fc1)
fc2 = Reshape((n,n,1))(fc2)
#%%
# sparse autoencoder
conv1 = Conv2D(m1, (3, 3), activation = 'relu', padding = 'same')(fc2)
conv2 = Conv2D(m2, (3, 3), activation = 'relu', padding = 'same')(conv1)
deconv = Conv2DTranspose(1, (3, 3), padding = 'same')(conv2)

#%%
automap = Model(input_img, deconv)
automap.compile(loss='mean_squared_error', optimizer = RMSprop())

automap.summary()
