#%%
import numpy as np 
import nibabel as nib 
from keras.preprocessing.image import ImageDataGenerator
import os
path = '/Users/chid/Downloads/ds101/'
sub_num = 12
#%%
filenames = os.listdir(path)
result = []
for filename in filenames:
    if os.path.isdir(os.path.join(os.path.abspath(path),filename)):
        result.append(filename)
result.sort()
#%%
print(result[1:])
#%%
fname = '/anatomy/highres001.nii.gz'
for f in result[1:1]:
    filename = os.path.join(os.path.abspath(path), f)
    filename = filename + fname
    a = nib.load(filename)
    a = a.get_data()
    mid = int(a.shape[1]/2)
    