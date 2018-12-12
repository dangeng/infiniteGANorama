import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
from PIL import Image
from scipy.misc import imresize

import pdb


class ManualDataset(BaseDataset):
    '''
    Manual dataset that selects only from the left side of the image
    This short coming is addressed in Manual2Dataset, so should probably use that
    '''

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

        self.curr_left = np.random.randint(len(self.A_paths))   # Current left image
        self.prev_right = None  # Previous getitem's right image

    def choose(self):
        # Called when user selects previous getitem output
        self.curr_left = self.prev_right

    def __getitem__(self, index):
        # Use previously chosen (current) left and random right image
        idx_l = self.curr_left
        idx_r = np.random.randint(len(self.A_paths))
        self.prev_right = idx_r

        A_path_l = self.A_paths[idx_l]
        A_path_r = self.A_paths[idx_r]
        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        A_l = A_l[:, 26:230, :]
        A_r = A_r[:, 25:230, :]

        A_l = imresize(A_l, (256, 256))
        A_r = imresize(A_r, (256, 256))

        h,w,c = A_l.shape

        A_img = np.zeros_like(A_l)
        #A_img[:,:w//2,:] = A_l[:,:w//2,:]
        #A_img[:,w//2:,:] = A_r[:,w//2:,:]
        A_img[:,:w//3,:] = A_l[:,:w//3,:]
        A_img[:,-w//3:,:] = A_r[:,:w//3+1,:]
        A_img = Image.fromarray(A_img)

        A = self.transform(A_img)
        if self.opt.direction == 'BtoA':
            input_nc = self.opt.output_nc
        else:
            input_nc = self.opt.input_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        return {'A': A, 'A_paths': '[{}]+[{}]]'.format(A_path_l, A_path_r)}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'ManualImageDataset'
