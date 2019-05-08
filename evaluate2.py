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
#%%
model = load_model('autoencoder2_mri.h5')
model.summary()
#%%
fname = 'highres001.nii.gz'
a = nib.load(fname)
a = a.get_data()
print(a.shape)
#%%
mid = int(a.shape[1]/2)
print(mid)
test_img = a[:,127,24:200]
print(test_img.shape)
#plt.imshow(test_img, cmap = 'gray')
#%%
# reading dicom
#ds = pydicom.dcmread('test.dcm')
#plt.imshow(ds.pixel_array)
#test_img = ds.pixel_array
test_img = resize(test_img, (116,116))
#test_img = rotate(test_img, 180)
#%%
m = np.max(test_img)
mi = np.min(test_img)
test_img = (test_img - mi)/(m - mi)
plt.imshow(test_img, cmap = 'gray')
plt.show()
#%%
test_img_tensor = np.expand_dims(test_img, axis = 0)
test_img_tensor = np.expand_dims(test_img_tensor, axis = 3)
print(test_img_tensor.shape)


#%%

pred = model.predict(test_img_tensor)
print(pred.shape)
plt.imshow(pred[0,:,:,0], cmap = 'gray')
plt.show()

#%%
layer_outputs = [layer.output for layer in model.layers[:36]]
print(len(layer_outputs))
activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
'''
#%%
activations = activation_model.predict(test_img_tensor)
print(len(activations))
#%%
first_layer_activation = activations[35]
print(first_layer_activation.shape)

#plt.matshow(first_layer_activation[0,:,:,25], cmap = 'viridis')
#%%
print(len(model.layers))
#%%
layer_names = []
for layer in model.layers[:36]:
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
        figname = 'activations/'+layer_name + '.png'
        plt.imshow(display_grid, aspect='auto',cmap='gray')
        plt.show()
        fig.savefig(figname, dpi=fig.dpi)

'''