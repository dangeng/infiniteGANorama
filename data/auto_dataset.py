import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
from PIL import Image
from scipy.misc import imresize

import pdb

class AutoDataset(BaseDataset):
    '''
    Auto Dataset that retrieves a user defined image and side
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

        # Start with left side of random image
        self.curr = (np.random.randint(len(self.A_paths)), 0)   # Current image + side
        self.prev = None # Previous image + side

        # Public variables
        # To be set to dictate what the output of the dataset is
        self.idx = 0
        self.side = 0

    def stage_retrieve(self):
        '''
        Generates random idx and side for you and returns it
        '''
        self.idx = np.random.randint(len(self.A_paths))
        self.side = np.random.randint(2)
        self.prev = (self.idx, self.side)
        return self.prev

    def stage_request(self, slug):
        '''
        Generates random idx and side for you and returns it
        Slug should be tuple: (idx, side)
        '''
        self.idx = slug[0]
        self.side = slug[1]
        self.prev = (self.idx, self.side)
        return self.prev

    def choose(self):
        # Called when user selects previous getitem output
        self.curr = self.prev

    def __getitem__(self, index):
        # Use user defined (from stage_retrieve function call) image for "prev"

        A_path_l = self.A_paths[self.curr[0]]
        A_path_r = self.A_paths[self.prev[0]]

        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        # Crop and resize to square
        A_l = A_l[:, 26:230, :]
        A_r = A_r[:, 25:230, :]
        A_l = imresize(A_l, (256, 256))
        A_r = imresize(A_r, (256, 256))

        h,w,c = A_l.shape

        A_img = np.zeros_like(A_l)
        #A_img[:,:w//2,:] = A_l[:,:w//2,:]
        #A_img[:,w//2:,:] = A_r[:,w//2:,:]
        if self.curr[1] == 0:
            A_img[:,:w//3,:] = A_l[:,:w//3,:]
        else:
            A_img[:,:w//3,:] = A_l[:,-w//3+1:,:]

        if self.prev[1] == 0:
            A_img[:,-w//3:,:] = A_r[:,:w//3+1,:]
        else:
            A_img[:,-w//3:,:] = A_r[:,-w//3:,:]

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
        return 'AutoImageDataset'
