
#%%
from __future__ import print_function
from __future__ import division
#%%
import numpy as np 
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'
import nibabel as nib
from skimage.transform import rotate, resize
#%%
petra_filepath = '/Users/chid/Downloads/PETRA_compressed/petra_1312.nii.gz'
mp2rage_filepath = '/Users/chid/Downloads/MP2RAGE_compressed/mp2rage_1312.nii.gz'
#%%
a = nib.load(petra_filepath)
a = a.get_data()
s = a[:,144,:]
s_petra = rotate(s,90)
s_petra_resized = resize(s_petra, (256,256))
plt.imshow(s_petra)

#%%
a = nib.load(mp2rage_filepath)
a = a.get_data()
s_mp2 = a[:,:,96]
plt.imshow(s_mp2)

#%%
plt.imshow(np.hstack((s_petra_resized,s_mp2)))
#%%
fig, axes = plt.subplots(1,2)
axes[0].hist(s_petra_resized.ravel(),bins = 20)
axes[0].set_title('PETRA slice histogram')
axes[1].hist(s_mp2.ravel(), bins = 20)
axes[1].set_title('MP2RAGE slice histogram')

#%%
plt.plot(s_petra_resized.ravel(), s_mp2.ravel(),'.')

#%%
def mutual_information(hgram):
#Mutual information for joint histogram

# Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


