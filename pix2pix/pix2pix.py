from __future__ import print_function, division
import scipy
from keras.optimizers import Adam
class Pix2Pix():
    def __init__(self):
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # configure data loader

        # calculate output shape of D (PatchGAN)
        patch = int(self.img_rows/2**4)
        self.disc_patch = (patch, patch, 1)

        # number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)