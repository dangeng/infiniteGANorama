import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import numpy as np
from PIL import Image
from scipy.misc import imresize


class FrankensteinDataset(BaseDataset):
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

    def random_crop(self, im, size=500, resize=256):
        h,w,_ = im.shape
        h_start, w_start = np.random.randint(h-size), np.random.randint(w-size)
        crop = im[h_start:h_start+size, w_start:w_start+size, :].copy()
        if resize:
            return imresize(crop, (resize, resize))
        else:
            return crop

    def __getitem__(self, index):
        idx_l = np.random.randint(len(self.A_paths))
        idx_r = np.random.randint(len(self.A_paths))
        A_path_l = self.A_paths[idx_l]
        A_path_r = self.A_paths[idx_r]
        A_img_l = Image.open(A_path_l).convert('RGB')
        A_img_r = Image.open(A_path_r).convert('RGB')

        A_l = np.array(A_img_l)
        A_r = np.array(A_img_r)

        #A_l, A_r = self.random_crop(A_l), self.random_crop(A_r) # CHANGE

        h,w,c = A_l.shape

        A_img = np.zeros_like(A_l)
        A_img[:,:w//2,:] = A_l[:,:w//2,:]
        A_img[:,w//2:,:] = A_r[:,w//2:,:]
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
        return 'FrankensteinImageDataset'
