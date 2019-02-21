import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
from PIL import Image
from scipy.misc import imresize
from torchvision.transforms import ToTensor, Compose
import torch

class FrankensteinDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, root, allrandom=False):
        self.root = root
        self.dir_A = os.path.join(root)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)
        self.allrandom = allrandom

    def random_crop(self, im, size=500, resize=256):
        h,w,_ = im.shape
        h_start, w_start = np.random.randint(h-size), np.random.randint(w-size)
        crop = im[h_start:h_start+size, w_start:w_start+size, :].copy()
        if resize:
            return imresize(crop, (resize, resize))
        else:
            return crop

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

        h,w,c = A_l.shape

        A_img_l_crop = A_l[:,:w//3,:]
        A_img_r_crop = A_r[:,2*w//3:,:]
        A_img = np.hstack((A_img_l_crop, A_img_r_crop))

        transform = ToTensor()

        return transform(A_img), same

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'FrankensteinImageDataset'
