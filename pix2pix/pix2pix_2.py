#%%
from __future__ import print_function, division
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize, rotate
import nibabel as nib 
import scipy
from keras.optimizers import Adam, RMSprop
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, ReLU, MaxPooling2D
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.layers.merge import concatenate
#%%
#from data_loader import DataLoader
import sys
import os
import datetime
#%%
class DataLoader():
    def __init__(self, dataset_name, img_res = (128,128)):
        self.img_res = img_res
        self.dataset_name = dataset_name

    def load_data(self, batch_size = 1, is_testing = False, is_jitter = False):
        def randomCrop(img , mask, width, height):
            assert img.shape[0] >= height
            assert img.shape[1] >= width
            assert img.shape[0] == mask.shape[0]
            assert img.shape[1] == mask.shape[1]
            x = np.random.randint(0, img.shape[1] - width)
            y = np.random.randint(0, img.shape[0] - height)
            img = img[y:y+height, x:x+width]
            mask = mask[y:y+height, x:x+width]
            return img, mask
    
        data_type = "train" if not is_testing else "test"
        path = glob('/home/student.unimelb.edu.au/chid/Documents/MRI_data/MRI_data/Daris/%s/%s/*' %(self.dataset_name,data_type))
        #path = glob('/Users/didichi/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        #path = glob('/Users/chid/.keras/datasets/facades/train/*')
        batch_images = np.random.choice(path, size = batch_size)
        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = nib.load(img_path)
            img = img.get_data()
            _,_,w = img.shape
            _w = int(w/2)
            img_A, img_B = img[:,:,:_w], img[:,:,_w:]
            #img_A, img_B = img[:,:,_w:],img[:,:,:_w]
            img_A = np.squeeze(img_A)
            img_B = np.squeeze(img_B)
            img_A = resize(img_A, (self.img_res[0],self.img_res[1]))
            img_B = resize(img_B, (self.img_res[0],self.img_res[1]))
            if not is_testing and np.random.random() <0.5 and is_jitter:
                # 1. Resize an image to bigger height and width
                img_A = resize(img_A, (img_A.shape[0] + 64, img_A.shape[1] + 64))
                img_B = resize(img_B, (img_B.shape[0] + 64, img_B.shape[1] + 64))
                # 2. Randomly crop the image
                img_A, img_B = randomCrop(img_A, img_B, self.img_res[0], self.img_res[1])
                # 3. Randomly flip the image horizontally
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
            m_A = np.max(img_A)
            mi_A = np.min(img_A)
            img_A = 2* (img_A - mi_A)/(m_A - mi_A) - 1
            m_B = np.max(img_B)
            mi_B = np.min(img_B)
            img_B = 2* (img_B - mi_B)/(m_B - mi_B) -1 
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = np.asarray(imgs_A, dtype=float)
        imgs_A = np.reshape(imgs_A, (-1,imgs_A.shape[1], imgs_A.shape[2],1))
        imgs_B = np.asarray(imgs_B, dtype = float)
        imgs_B = np.reshape(imgs_B, (-1,imgs_B.shape[1],imgs_B.shape[2],1))
        return imgs_A, imgs_B
    
    def load_batch(self, batch_size = 1, is_testing = False, is_jitter = False):
        def randomCrop(img , mask, width, height):
            assert img.shape[0] >= height
            assert img.shape[1] >= width
            assert img.shape[0] == mask.shape[0]
            assert img.shape[1] == mask.shape[1]
            x = np.random.randint(0, img.shape[1] - width)
            y = np.random.randint(0, img.shape[0] - height)
            img = img[y:y+height, x:x+width]
            mask = mask[y:y+height, x:x+width]
            return img, mask
        data_type = "train" if not is_testing else "test"
        #path = glob('/Users/didichi/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        path = glob('/home/student.unimelb.edu.au/chid/Documents/MRI_data/MRI_data/Daris/%s/%s/*' % (self.dataset_name,data_type)) 
        self.n_batches = int(len(path) / batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = nib.load(img)
                img = img.get_data()
                _,_,w = img.shape
                _w = int(w/2)
                img_A, img_B = img[:,:,:_w], img[:,:,_w:]
                #img_A, img_B = img[:,:,_w:],img[:,:,:_w]
                img_A = np.squeeze(img_A)
                img_B = np.squeeze(img_B)
                img_A = resize(img_A, (self.img_res[0],self.img_res[1]))
                img_B = resize(img_B, (self.img_res[0],self.img_res[1]))
                #print(img_A.shape)
                #print(img_B.shape)
                if not is_testing and np.random.random() <0.5 and is_jitter:
                    # 1. Resize an image to bigger height and width
                    img_A = resize(img_A, (img_A.shape[0] + 64, img_A.shape[1] + 64))
                    img_B = resize(img_B, (img_B.shape[0] + 64, img_B.shape[1] + 64))
                    # 2. Randomly crop the image
                    img_A, img_B = randomCrop(img_A, img_B, self.img_res[0], self.img_res[1])
                    # 3. Randomly flip the image horizontally
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                m_A = np.max(img_A)
                mi_A = np.min(img_A)
                img_A = 2* (img_A - mi_A)/(m_A - mi_A) - 1
                m_B = np.max(img_B)
                mi_B = np.min(img_B)
                img_B = 2* (img_B - mi_B)/(m_B - mi_B) - 1
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = np.asarray(imgs_A, dtype=float)
            imgs_A = np.reshape(imgs_A, (-1,imgs_A.shape[1], imgs_A.shape[2],1))
            imgs_B = np.asarray(imgs_B, dtype = float)
            imgs_B = np.reshape(imgs_B, (-1,imgs_B.shape[1], imgs_B.shape[2],1))
            yield imgs_A, imgs_B
    
#%%
class Pix2Pix():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # configure data loader
        self.dataset_name = 'p2m'
        self.data_loader = DataLoader(dataset_name = self.dataset_name, img_res = (self.img_rows, self.img_cols))
        # calculate output shape of D (PatchGAN)
        patch = int(self.img_rows/2**4) # was 4 
        self.disc_patch = (patch, patch, 1)

        # number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(lr = 0.0002, beta_1= 0.5, beta_2=0.999)
        #optimizer = RMSprop(0.01)
        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss = 'mse', optimizer = optimizer, metrics = ['accuracy'])
        #self.discriminator.compile(loss = 'binary_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
        # build the generator
        self.generator = self.build_generator()
        self.generator.summary()
        # input images and their conditioning images
        img_mp2 = Input(shape = self.img_shape) # real image
        img_petra = Input(shape = self.img_shape) # input image
        # by conditioning on petra generate a fake version of mp2rage
        fake_mp2 = self.generator(img_petra)
        # for the combined model we will only train the generator
        self.discriminator.trainable = False
        # discriminators determines validity of translated images
        valid = self.discriminator([fake_mp2, img_petra])
        
        self.combined = Model(inputs = [img_mp2, img_petra], outputs = [valid, fake_mp2])
        self.combined.compile(loss = ['mse','mae'], loss_weights=[1,1000],optimizer = optimizer)
        #self.combined.compile(loss = ['binary_crossentropy','mae'], loss_weights=[1,100], optimizer = optimizer)

    def build_generator(self):
        '''u-net'''
        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            #d = Dropout(0.1)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            #u = Conv2DTranspose()
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        d1 = conv2d(d0, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)

        # Upsampling
        u1 = deconv2d(d7, d6, self.gf*8)
        u2 = deconv2d(u1, d5, self.gf*8)
        u3 = deconv2d(u2, d4, self.gf*8)
        u4 = deconv2d(u3, d3, self.gf*4)
        u5 = deconv2d(u4, d2, self.gf*2)
        u6 = deconv2d(u5, d1, self.gf)

        u7 = UpSampling2D(size=2)(u6)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model(d0, output_img)
    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        #input image
        inp = Input(shape = self.img_shape)
        #target image
        tar = Input(shape = self.img_shape)

        #Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis = -1)([inp, tar])

        '''
        Discriminator receives 2 inputs.
            Input image and the target image, which it should classify as real.
            Input image and the generated image (output of generator), which it should classify as fake.
            We concatenate these 2 inputs together in the code (tf.concat([inp, tar], axis=-1))
        '''

        d1 = d_layer(combined_imgs, self.df, bn = False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size = 4, strides = 1, padding = 'same')(d4)

        return Model([inp, tar], validity)
    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        #print(valid.shape)
        #print(fake.shape)
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size, is_jitter= False)):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                #print(imgs_B.shape)
                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)
                #print(fake_A.shape)
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # print(d_loss.shape)
                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])

                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
        self.discriminator.save('D.h5')
        self.generator.save('G.h5')
        self.combined.save('combined.h5')
    def sample_images(self, epoch, batch_i):
        os.makedirs('images_l1_1000/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3 # row and col

        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True, is_jitter=False)
        fake_A = self.generator.predict(imgs_B)
        #print(imgs_A.shape)
        #print(imgs_B.shape)
        #print(fake_A.shape)
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                #print(cnt)
                #print(gen_imgs.shape)
                #print(gen_imgs[cnt].shape)
                current_img = gen_imgs[cnt]
                current_img = np.reshape(current_img, (current_img.shape[0], current_img.shape[1]))
                #print(current_img.shape)
                axs[i,j].imshow(current_img, cmap = 'gray')
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images_l1_1000/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

#%%
if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=50, batch_size=1, sample_interval=200)


#%%

#D = DataLoader('p2m',(256,256))

#%%
#img_A,img_B = D.load_data(3)
#%%
#print(img_A.shape)
