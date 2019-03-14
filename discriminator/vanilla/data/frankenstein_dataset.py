import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
from PIL import Image
from scipy.misc import imresize, imsave
from torchvision.transforms import ToTensor, Compose
import torch
import cv2
from scipy.ndimage.filters import gaussian_filter as blur

import pdb

class FrankensteinDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, root, allrandom=False, return_idx=False, blur=False):
        self.root = root
        self.dir_A = os.path.join(root)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)
        self.allrandom = allrandom
        self.return_idx = return_idx
        self.blur = blur

    def get_path_name(self, idx):
        return self.A_paths[idx]

    def random_crop(self, im, size=500, resize=256):
        h,w,_ = im.shape
        h_start, w_start = np.random.randint(h-size), np.random.randint(w-size)
        crop = im[h_start:h_start+size, w_start:w_start+size, :].copy()
        if resize:
            return imresize(crop, (resize, resize))
        else:
            return crop

    def crop(self, im_l, im_r, same):
        '''
        Crop image to 341 x 500
        Starting image is assumed to be guranteed to be size 1024 x >500
        '''
        SIZE = 1024
        WIDTH = 341
        HEIGHT = 500

        SIZE = 512
        WIDTH = 170
        HEIGHT = 249

        # FOR CVCL
        SIZE = 256
        WIDTH = 120
        HEIGHT = 255

        yl = np.random.randint(0, im_l.shape[0]-HEIGHT)
        yr = np.random.randint(0, im_r.shape[0]-HEIGHT)

        if same:
            # we choose the actual left and right "half"
            xl = np.random.randint(0, SIZE//2 - WIDTH)
            xr = np.random.randint(SIZE//2, SIZE - WIDTH)

            # Set y cuts to be the same (we want horizons!)
            yl = yr
        else:
            # we choose the patch from anywhere
            xl = np.random.randint(0, SIZE - WIDTH)
            xr = np.random.randint(0, SIZE - WIDTH)

        return im_l[yl:yl+HEIGHT, xl:xl+WIDTH, :], im_r[yr:yr+HEIGHT, xr:xr+WIDTH, :]

    def get_deterministic(self, idx_l, idx_r):
        if idx_l==idx_r:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)

        A_path_l = self.A_paths[idx_l]
        A_path_r = self.A_paths[idx_r]
        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        A_img_l_crop, A_img_r_crop = self.crop(A_l, A_r, same)

        # Ensure commutativity, flip right side
        A_img_r_crop = A_img_r_crop[:,::-1,:]

        # concatenate on channel axis
        try:
            A_img = np.concatenate((A_img_l_crop, A_img_r_crop), 2) 
        except:
            return {}, {}

        transform = ToTensor()

        return transform(A_img), same

    def __getitem__(self, index):
        if self.allrandom:
            same = 0
        else:
            same = np.random.randint(2)

        idx_l = np.random.randint(len(self.A_paths))
        idx_r = idx_l
        if same==1:
            same = torch.ones(1)
        else:
            same = torch.zeros(1)
            while idx_r==idx_l:
                idx_r = np.random.randint(len(self.A_paths))

        A_path_l = self.A_paths[idx_l]
        A_path_r = self.A_paths[idx_r]
        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        #A_l, A_r = self.random_crop(A_l), self.random_crop(A_r) # CHANGE

        A_img_l_crop, A_img_r_crop = self.crop(A_l, A_r, same)

        # Ensure commutativity, flip right side
        A_img_r_crop = A_img_r_crop[:,::-1,:]

        if self.blur:
            A_img_l_crop = blur(A_img_l_crop, (2,2,0))
            A_img_r_crop = blur(A_img_r_crop, (2,2,0))

        '''
        A_img = np.hstack((A_img_l_crop, A_img_r_crop))
        imsave('test_ims/{}.jpg'.format(index), A_img)
        imsave('test_ims/{}-l.jpg'.format(index), A_l)
        imsave('test_ims/{}-r.jpg'.format(index), A_r)
        '''

        '''
        h,w,c = A_l.shape
        h = min(h, A_r.shape[0])    # Choose min height
        w = w//3

        A_img_l_crop = A_l[:h,:w,:]
        A_img_r_crop = A_r[:h,-w:,:]
        #A_img = np.hstack((A_img_l_crop, A_img_r_crop))
        '''
        # concatenate on channel axis
        try:
            A_img = np.concatenate((A_img_l_crop, A_img_r_crop), 2) 
        except:
            return {}, {}

        #pdb.set_trace()
        #A_img = cv2.resize(A_img, dsize=(170, 250), interpolation=cv2.INTER_CUBIC)

        transform = ToTensor()

        if self.return_idx:
            return transform(A_img), same, idx_l, idx_r
        else:
            return transform(A_img), same

    def __len__(self):
        # if we use length squared things break (?)
        return len(self.A_paths)

    def name(self):
        return 'FrankensteinImageDataset'
