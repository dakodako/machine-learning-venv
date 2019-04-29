#%%
import sys
import nibabel as nib
import numpy as np
import glob
from matplotlib import pyplot as plt
from skimage.transform import rotate, resize
#%%
def pad_zero_margins(input_img, size):
    width = input_img.shape[1]
    start = int(np.floor((size - width)/2))
    output_img = np.zeros((size,size))
    output_img[:,start:(start + width)] = input_img
    return output_img
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




print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))
print(sys.argv[1])

filepath_X = sys.argv[1]
filepath_ground = sys.argv[2]

b = nib.load(filepath_ground)
b = b.get_data()
print(b.shape)
images = []
mid = int(b.shape[1]/2)
b = b[:,:,mid-25:mid + 25]
for i in range(b.shape[2]):
    temp = b[:,:,i]
    temp = np.reshape(temp,[b.shape[0],b.shape[1]])
    temp = pad_zero_margins(temp,256)
    images.append(temp)
images = np.asarray(images)
images = images.reshape(-1,images.shape[1],images.shape[2],1)
m = np.max(images)
mi = np.min(images)
images = (images - mi)/(m - mi)

print(images.shape)
s = np.reshape(images[20,:,:,:],[256,256])
plt.imshow(s)
plt.show()

#%%
filename = '../Documents/MRI_data/dataset/X/100610_T1w_restore.1.60.nii.gz'
images = []
a = nib.load(filename)
a = a.get_data()
mid = int(a.shape[2]/2)
a = a[:,:,mid-25:mid+25]
for i in range(a.shape[2]):
    temp = rotate(a[:,:,i],90, resize = True)
    temp = pad_zero_margins(temp,136)
    temp = resize(temp, (256,256))
    images.append(temp)
images = np.asarray(images)
images = images.reshape(-1,images.shape[1],images.shape[2],1)
m = np.max(images)
mi = np.min(images)
images = (images - mi)/(m - mi)
#%%
print(images.shape)

s = images[25,:,:,0]
print(s.shape)
plt.imshow(s)
#%%
img = np.zeros((136,136))
plt.imshow(img)
img[:,12:125] = s
plt.imshow(img)
img = resize(img, (256,256))
plt.imshow(img)
#%%
img2 = pad_zero_margins(s, 136)
print(img2.shape)
plt.imshow(img2)