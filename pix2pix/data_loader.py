#%%
import scipy
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.transform import resize, rotate
import nibabel as nib 
#%%
'''
batch_size = 1
path = glob('/Users/chid/.keras/datasets/facades/train/*')
n_batches = int(len(path) / batch_size)
print(n_batches)
for i in range(n_batches-399):
    batch = path[i*batch_size:(i+1)*batch_size]
    imgs_A, imgs_B = [], []
    for img in batch:
        img = imread(img)
        h,w,_ = img.shape
        _w = int(w/2)
        img_A, img_B = img[:,:_w,:], img[:,_w:,:]
        img_A = resize(img_A, self.img_res)
        img_B = resize(img_B, self.img_res)
        if not is_testing and np.random.random() <0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
        m_A = np.max(img_A)
        mi_A = np.min(img_A)
        img_A = (img_A - mi_A)/(m_A - mi_A)
        m_B = np.max(img_B)
        mi_B = np.min(img_B)
        img_B = (img_B - mi_B)/(m_B - mi_B)
        imgs_A.append(img_A)
        imgs_B.append(img_B)

    
        #plt.imshow(img)
    #print(batch)
#print(batch_images)
#%%

#%%
path = glob('/Users/chid/.keras/datasets/p2m/train/*')
batch_images = np.random.choice(path, size = 1)
print(batch_images)
imgs_A = []
imgs_B = []
#%%
for img_path in batch_images:
    img = nib.load(img_path)
    img = img.get_data()
    h,w = img.shape
    _w = int(w/2)
    img_A, img_B = img[:,:_w], img[:,_w:]
    m_A = np.max(img_A)
    mi_A = np.min(img_A)
    img_A = (img_A - mi_A)/(m_A - mi_A)
    m_B = np.max(img_B)
    mi_B = np.min(img_B)
    img_B = (img_B - mi_B)/(m_B - mi_B)
    imgs_A.append(img_A)
    imgs_B.append(img_B)
#%%
plt.imshow(img_A,cmap = 'gray')
#%%
plt.imshow(img_B,cmap = 'gray')

#%%

#%%
path = glob('/Users/chid/.keras/datasets/facades/train/*')
batch_images = np.random.choice(path, size = 1)
print(batch_images)
test = imread(batch_images[0])
#%%
img = test[:,:int(test.shape[1]/2),:]
mask = test[:,int(test.shape[1]/2):,:]
print(img.shape)
print(mask.shape)
img,mask = randomCrop(img, mask, 128, 128)
#%%
plt.imshow(img)
#%%
plt.imshow(mask)
'''
#%%
class DataLoader():
    def __init__(self, dataset_name, img_res = (128,128)):
        self.img_res = img_res
        self.dataset_name = dataset_name
    
    def load_data(self, batch_size = 1, is_testing = False):
        data_type = "train" if not is_testing else "test"
        path = glob('/Users/chid/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        #path = glob('/Users/chid/.keras/datasets/facades/train/*')
        batch_images = np.random.choice(path, size = 1)
        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = nib.load(img_path)
            img = img.get_data()
            h,w = img.shape
            _w = int(w/2)
            img_A, img_B = img[:,:_w], img[:,_w:]
            img_A = resize(img_A, self.img_res)
            img_B = resize(img_B, self.img_res)
            if not is_testing and np.random.random() <0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)
            m_A = np.max(img_A)
            mi_A = np.min(img_A)
            img_A = (img_A - mi_A)/(m_A - mi_A)
            m_B = np.max(img_B)
            mi_B = np.min(img_B)
            img_B = (img_B - mi_B)/(m_B - mi_B)
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = np.asarray(imgs_A, dtype=float)
        imgs_A = np.reshape(imgs_A, (imgs_A.shape[1], imgs_A.shape[2]))
        imgs_B = np.asarray(imgs_B, dtype = float)
        imgs_B = np.reshape(imgs_B, (imgs_B.shape[1],imgs_B.shape[2]))
        return imgs_A, imgs_B
    def load_data_with_random_jitter(self, batch_size = 1, is_testing = False):
        def randomCrop(img , mask, width, height):
            print(img.shape[0])
            print(height)
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
        path = glob('/Users/chid/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        #path = glob('/Users/chid/.keras/datasets/facades/train/*')
        batch_images = np.random.choice(path, size = 1)
        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = nib.load(img_path)
            img = img.get_data()
            h,w = img.shape
            _w = int(w/2)
            img_A, img_B = img[:,:_w], img[:,_w:]
            img_A = resize(img_A, self.img_res)
            img_B = resize(img_B, self.img_res)
            if not is_testing and np.random.random() <0.5:
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
            img_A = (img_A - mi_A)/(m_A - mi_A)
            m_B = np.max(img_B)
            mi_B = np.min(img_B)
            img_B = (img_B - mi_B)/(m_B - mi_B)
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = np.asarray(imgs_A, dtype=float)
        imgs_A = np.reshape(imgs_A, (imgs_A.shape[1], imgs_A.shape[2]))
        imgs_B = np.asarray(imgs_B, dtype = float)
        imgs_B = np.reshape(imgs_B, (imgs_B.shape[1],imgs_B.shape[2]))
        return imgs_A, imgs_B
    def load_batch(self, batch_size = 1, is_testing = False):
        data_type = "train" if not is_testing else "test"
        path = glob('/Users/chid/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        self.n_batches = int(len(path) / batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = nib.load(img)
                img = img.get_data()
                h,w = img.shape
                _w = int(w/2)
                img_A, img_B = img[:,:_w], img[:,_w:]
                img_A = resize(img_A, self.img_res)
                img_B = resize(img_B, self.img_res)
                if not is_testing and np.random.random() <0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)
                m_A = np.max(img_A)
                mi_A = np.min(img_A)
                img_A = (img_A - mi_A)/(m_A - mi_A)
                m_B = np.max(img_B)
                mi_B = np.min(img_B)
                img_B = (img_B - mi_B)/(m_B - mi_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)
        imgs_A = np.asarray(imgs_A, dtype=float)
        imgs_A = np.reshape(imgs_A, (imgs_A.shape[1], imgs_A.shape[2]))
        imgs_B = np.asarray(imgs_B, dtype = float)
        imgs_B = np.reshape(imgs_B, (imgs_B.shape[1],imgs_B.shape[2]))
        yield imgs_A, imgs_B
    def load_batch_with_random_jitter(self, batch_size = 1, is_testing = False):
        def randomCrop(img , mask, width, height):
            print(img.shape[0])
            print(height)
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
        path = glob('/Users/chid/.keras/datasets/%s/%s/*' % (self.dataset_name, data_type))
        self.n_batches = int(len(path) / batch_size)
        for i in range(self.n_batches-1):
            batch = path[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = nib.load(img)
                img = img.get_data()
                h,w = img.shape
                _w = int(w/2)
                img_A, img_B = img[:,:_w], img[:,_w:]
                img_A = resize(img_A, self.img_res)
                img_B = resize(img_B, self.img_res)
                if not is_testing and np.random.random() <0.5:
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
                img_A = (img_A - mi_A)/(m_A - mi_A)
                m_B = np.max(img_B)
                mi_B = np.min(img_B)
                img_B = (img_B - mi_B)/(m_B - mi_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)
        imgs_A = np.asarray(imgs_A, dtype=float)
        imgs_A = np.reshape(imgs_A, (imgs_A.shape[1], imgs_A.shape[2]))
        imgs_B = np.asarray(imgs_B, dtype = float)
        imgs_B = np.reshape(imgs_B, (imgs_B.shape[1],imgs_B.shape[2]))
        yield imgs_A, imgs_B
    

#%%
D = DataLoader('p2m',(256,256))
#%%
test_A, test_B = D.load_data_with_random_jitter(1, False)
#%%
#test_A = np.asarray(test_A, dtype = float)
print(test_A.shape)
#%%
plt.imshow(test_A)
#%%
#test_B = np.asarray(test_B, dtype = float)
plt.imshow(test_B)
#%%
batch_test_A = D.load_batch(1, True)
#%%
batch_size = 1
for batch_i, (imgs_A, imgs_B) in enumerate(D.load_batch(batch_size, True)):
    print(imgs_A.shape)