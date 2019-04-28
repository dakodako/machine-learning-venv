
import sys
import nibabel as nib
import numpy as np
import glob
from matplotlib import pyplot as plt
def pad_zero_margins(input_img):
    output_img = np.zeros((256,256))
    output_img[:,31:224] = input_img
    return output_img
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
'''
a = nib.load(filepath_X)
a = a.get_data()
print(a.shape)
images = []
mid = int(a.shape[1]/2)
a = a[:,:,mid-25:mid + 25]
for i in range(a.shape[2]):
    images.append((a[:,:,i]))
images = np.asarray(images)
images = images.reshape(-1,images.shape[1],images.shape[2],1)
m = np.max(images)
mi = np.min(images)
images = (images - mi)/(m - mi)

print(images.shape)
s = np.reshape(images[20,:,:,:],[256,193])
plt.imshow(s)
plt.show()

'''
b = nib.load(filepath_ground)
b = b.get_data()
print(b.shape)
images = []
mid = int(b.shape[1]/2)
b = b[:,:,mid-25:mid + 25]
for i in range(b.shape[2]):
    temp = b[:,:,i]
    temp = np.reshape(temp,[b.shape[0],b.shape[1]])
    temp = pad_zero_margins(temp)
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
'''
img = np.zeros((256,256))
plt.imshow(img)
plt.show()
img[:,31:224] = s
plt.imshow(img)
plt.show()
'''

