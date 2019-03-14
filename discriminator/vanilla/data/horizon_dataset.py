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

class HorizonDataset(BaseDataset):
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

        self.idx_l = np.random.randint(len(self.A_paths))
        self.idx_r = self.idx_l

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

    def crop(self, im_l, im_r, offset):
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
        HEIGHT = 200

        # Interpolate between 0 and maxheight
        yl = (1-offset) * (im_l.shape[0]-HEIGHT)
        yr = offset * (im_r.shape[0]-HEIGHT)

        yl, yr = int(yl), int(yr)

        # we choose the actual left and right "half"
        xl = 0
        xr = SIZE - WIDTH

        return im_l[yl:yl+HEIGHT, xl:xl+WIDTH, :], im_r[yr:yr+HEIGHT, xr:xr+WIDTH, :]

    def get_deterministic(self, offset, idx_l, idx_r):
        same = torch.ones(1)

        A_path_l = self.A_paths[idx_l]
        A_path_r = self.A_paths[idx_r]
        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        A_img_l_crop, A_img_r_crop = self.crop(A_l, A_r, offset/100.)

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
        # Always same
        same = torch.ones(1)

        A_path_l = self.A_paths[self.idx_l]
        A_path_r = self.A_paths[self.idx_r]
        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        A_img_l_crop, A_img_r_crop = self.crop(A_l, A_r, index/100.)

        # Ensure commutativity, flip right side
        A_img_r_crop = A_img_r_crop[:,::-1,:]

        # concatenate on channel axis
        try:
            A_img = np.concatenate((A_img_l_crop, A_img_r_crop), 2) 
        except:
            return {}, {}

        transform = ToTensor()

        return transform(A_img), same, self.idx_l, self.idx_r

    def __len__(self):
        # if we use length squared things break (?)
        return len(self.A_paths)

    def name(self):
        return 'HorizonImageDataset'
