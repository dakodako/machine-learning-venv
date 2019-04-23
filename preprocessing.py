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
images = []
for f in result[1:2]:
	filename = os.path.join(os.path.abspath(path), f)
	filename = filename + fname
	#print(filename)
	a = nib.load(filename)
	a = a.get_data()
	mid = int(a.shape[1]/2)
	#print(mid)
	test_img = a[:,mid -25:mid + 25, 24:200]
	#print(test_img.shape)
	for i in range(test_img.shape[1]):
		images.append(test_img[:,i,:])
images = np.asarray(images)
images = images.reshape(-1,images.shape[1], images.shape[2],1)
m = np.max(images)
mi = np.min(images)

