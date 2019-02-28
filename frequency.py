#%%
import numpy as np 
import os

from matplotlib import pyplot as plt 

import open_and_view_nii_images_functions
from open_and_view_nii_images_functions import extract_a_slice, open_image, open_images
#%%
filepath = '/Users/chid/machine-learning-venv/dataset/T1/*'
filename = '/Users/chid/machine-learning-venv/dataset/T1/943862_T1w_restore.1.60.nii.gz'
img = open_image(filename)

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

img_freq = to_freq_space_2d(s)
print(img_freq.shape)

#%%

print(np.max(img_freq[:,:,0]))
print(np.max(img_freq[:,:,1]))
plt.figure(figsize=[5,5])

# Display the first image in training data
plt.subplot(121)
curr_img_real = np.reshape(img_freq[:,:,0], (113,113))
plt.imshow(curr_img_real, cmap='gray')

plt.subplot(122)
curr_img_real = np.reshape(img_freq[:,:,1], (113,113))
plt.imshow(curr_img_real, cmap='gray')
plt.show()

#%%

#if normalize:
#    x = x - np.mean(x)

 

