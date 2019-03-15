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

class EvalDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, root, allrandom=False, blur=False):
        self.root = root
        self.dir_A = os.path.join(root)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)
        self.allrandom = allrandom
        self.blur = blur
        #self.left_aux = {'idx'     : np.random.randint(len(self.A_paths)),
        self.left_aux = {'idx'     : 153,  # For mountains
                         'x_crop'  : 0, 
                         'y_crop'  : 0}

    def get_path_name(self, idx):
        return self.A_paths[idx]

    def set_left(self, aux):
        self.left_aux = aux

    def crop(self, im_l, im_r, aux=None):
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

        yl = self.left_aux['y_crop']
        yr = np.random.randint(0, im_r.shape[0]-HEIGHT)

        xl = self.left_aux['x_crop']
        xr = np.random.randint(0, SIZE - WIDTH)

        if aux:
            yr = aux['y_crop']
            xr = aux['x_crop']


        return im_l[yl:yl+HEIGHT, xl:xl+WIDTH, :], im_r[yr:yr+HEIGHT, xr:xr+WIDTH, :], {'x_crop': xr, 'y_crop': yr}
    
    def get_deterministic(self, index, crop_params):
        idx_l = self.left_aux['idx']
        idx_r = index

        assert idx_l != idx_r, "Indices cannot be equal!"

        A_path_l = self.A_paths[idx_l]
        A_path_r = self.A_paths[idx_r]
        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        A_img_l_crop, A_img_r_crop, crop_params = self.crop(A_l, A_r, aux=crop_params)

        # Ensure commutativity, flip right side
        A_img_r_crop = A_img_r_crop[:,::-1,:]

        if self.blur:
            A_img_l_crop = blur(A_img_l_crop, (2,2,0))
            A_img_r_crop = blur(A_img_r_crop, (2,2,0))

        # concatenate on channel axis
        try:
            A_img = np.concatenate((A_img_l_crop, A_img_r_crop), 2) 
        except:
            return {}, {}

        transform = ToTensor()

        crop_params['idx'] = index

        return transform(A_img), crop_params

    def __getitem__(self, index):
        idx_l = self.left_aux['idx']
        idx_r = index

        assert idx_l != idx_r, "Indices cannot be equal!"

        A_path_l = self.A_paths[idx_l]
        A_path_r = self.A_paths[idx_r]
        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        A_img_l_crop, A_img_r_crop, crop_params = self.crop(A_l, A_r)

        # Ensure commutativity, flip right side
        A_img_r_crop = A_img_r_crop[:,::-1,:]

        if self.blur:
            A_img_l_crop = blur(A_img_l_crop, (2,2,0))
            A_img_r_crop = blur(A_img_r_crop, (2,2,0))

        # concatenate on channel axis
        try:
            A_img = np.concatenate((A_img_l_crop, A_img_r_crop), 2) 
        except:
            return {}, {}

        transform = ToTensor()

        crop_params['idx'] = index

        return transform(A_img), crop_params

    def __len__(self):
        # if we use length squared things break (?)
        return len(self.A_paths)

    def name(self):
        return 'EvalImageDataset'
