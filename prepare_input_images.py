#%%
from skimage.io import imread
from skimage.color import rgb2grey
from skimage.transform import resize
#%%
import glob
import numpy as np # will use numpy for fft
from numpy import fft
import matplotlib.pyplot as plt
#%%
# read the image
filepath = '~/machine-learning-venv/imagenet/'
filename = 'ILSVRC2011_val_00000012.JPEG'
print(filepath + filename)
#%%
#for f in sorted(glob.glob(filepath)):
a = imread(filepath + filename)
# resize the image
a = resize(a, (128, 128))

plt.imshow(a)
#%%
# turn the image into grayscale
a = rgb2grey(a)
plt.imshow(a, cmap = 'gray')
#%%
# fourier transform the image
A = fft.fft2(a)
A = fft.fftshift(A)
plt.imshow(np.log(np.abs(A)), cmap='gray')

# undersample the k-space

# normailise the input ??
# return the full k-space and undersampled k-space (partial k-space)
