import scipy
from glob import glob
import numpy as numpy
from matplotlib import pyplot as pyplot

class DataLoader():
    def __init__(self, img_res = (128,128)):
        self.img_res = img_res
    def load_data(self, batch_size = 1, is_testing = False):

        imgs_A = []
        imgs_B = []

        return imgs_A, imgs_B