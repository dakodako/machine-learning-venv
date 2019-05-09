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
#%%
test_img = test_X[30,:,:,:]
plt.imshow(test_img[:,:,0], cmap = 'gray')
#%%
test_img_tensor = np.expand_dims(test_img, axis = 0)
print(test_img_tensor.shape)
layer_outputs = [layer.output for layer in model.layers[:59]]
print(len(layer_outputs))

#%%
pred = model.predict(test_img_tensor)
plt.imshow(pred[0,:,:,0], cmap = 'gray')
#%%

im = plt.imshow(np.abs(pred[0,:,:,0]-test_ground[30,:,:,0]))

#%%
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
activations = activation_model.predict(test_img_tensor)
print(len(activations))

#%%
layer_names = []
for layer in model.layers[:59]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    print(size)
   
    n_cols = n_features // images_per_row
    print(n_cols)
    display_grid = np.zeros((size*n_cols, images_per_row * size))
    print(display_grid.shape)
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:,:, col*images_per_row + row]
            #channel_image = np.log(channel_image)
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col*size : (col+1)*size,
                        row*size:(row+1)*size] = channel_image
    if n_cols != 0:
        scale = 1./size
        fig = plt.figure(figsize=(scale*display_grid.shape[1], scale * display_grid.shape[0]))
        print(display_grid.shape)
        plt.title(layer_name)
        plt.grid(False)
        figname = 'activations/p2m/'+layer_name + '.png'
        plt.imshow(display_grid, aspect='auto')
        plt.show()
        fig.savefig(figname, dpi=fig.dpi)
