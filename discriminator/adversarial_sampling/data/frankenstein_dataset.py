import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
from PIL import Image
from scipy.misc import imresize
from torchvision.transforms import ToTensor, Compose, Normalize
import torch

import pdb

class FrankensteinDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, root, length=None):
        self.root = root
        self.dir_A = os.path.join(root)

        self.A_paths = make_dataset(self.dir_A)
        self.A_paths = sorted(self.A_paths)
        if length is not None:
            self.A_paths = self.A_paths[:length]

    def random_crop(self, im, size=500, resize=256):
        h,w,_ = im.shape
        h_start, w_start = np.random.randint(h-size), np.random.randint(w-size)
        crop = im[h_start:h_start+size, w_start:w_start+size, :].copy()
        if resize:
            return imresize(crop, (resize, resize))
        else:
            return crop

    def __getitem__(self, index):
        N = int(np.sqrt(len(self)))
        idx_l, idx_r = np.unravel_index(index, (N, N))

        same = torch.tensor(int(idx_l == idx_r))

        A_path_l = self.A_paths[idx_l]
        A_path_r = self.A_paths[idx_r]
        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        #A_l, A_r = self.random_crop(A_l), self.random_crop(A_r) # CHANGE

        h,w,c = A_l.shape
        h = min(h, A_r.shape[0])    # Choose min height
        w = w//3

        A_img_l_crop = A_l[:h,:w,:]
        A_img_r_crop = A_r[:h,-w:,:]
        #A_img = np.hstack((A_img_l_crop, A_img_r_crop))
        # concatenate on channel axis
        try:
            A_img = np.concatenate((A_img_l_crop, A_img_r_crop), 2) 
        except:
            return {}, {}

        transform = Compose([ToTensor(), 
            Normalize([0.4009, 0.3978, 0.3817, 0.4032, 0.3987, 0.3820],
                      [0.14384869, 0.1398176 , 0.16267562, 0.14757314, 0.14043665, 0.16479132])])

        return transform(A_img), same

    def __len__(self):
        return len(self.A_paths)**2

    def name(self):
        return 'FrankensteinImageDataset'
