import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
#from util.visualizer import Visualizer
import matplotlib.pyplot as plt

import pdb
import torch
from collections import OrderedDict
from util import util
import numpy as np
from scipy.misc import imsave

# Generates a ton of images and ranks them in order of
# discriminator loss

def plotTensor(im):
    im = util.tensor2im(im)
    plotim(im)

def plotim(im):
    plt.imshow(im)
    plt.show()

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.no_flip = True
    opt.resize_or_crop='none'
    opt.dataset_mode='manual2'

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    dirname = '12_nol1'

    model = create_model(opt)
    model.setup(opt)
    total_steps = 0

    chkpt_D = torch.load('checkpoints/ranker/earliest_net_D.pth')
    #chkpt_G = torch.load('checkpoints/streetview_throttled_sidesonly/12_net_G.pth') # good generator
    chkpt_G = torch.load('checkpoints/streetview_nlayers5/30_net_G.pth') # best generator!!!
    #chkpt_G = torch.load('checkpoints/streetview_nol1/12_net_G.pth')

    new_chkpt_D = OrderedDict()
    new_chkpt_G = OrderedDict()
    for k, v in chkpt_D.items():
        name = 'module.' + k # add `module.`
        new_chkpt_D[name] = v
    for k, v in chkpt_G.items():
        name = 'module.' + k # add `module.`
        new_chkpt_G[name] = v

    model.netD.load_state_dict(new_chkpt_D)
    model.netG.load_state_dict(new_chkpt_G)

    generated = []

    #for i, data in enumerate(dataset):
    while True:
        data = data_loader.dataset[0]
        data['A'] = data['A'].unsqueeze(0)  # required because we're not using dataloader, which adds a batch dim

        model.set_input(data)

        # Generate image
        model.forward()

        # Create input to discriminator
        fake_AB = torch.cat((model.real_A, model.fake_B), 1)

        # Feed fake_AB to patchGAN discriminator
        pred = model.netD(fake_AB)

        # Get loss of discriminator
        loss = model.criterionGAN(pred, False)

        fake_B = util.tensor2im(model.fake_B)
        #samples.append([loss, model.fake_B[0].detach().cpu().numpy().transpose(1,2,0)])
        plotim(fake_B)
        print('Do you like it? (y/*)')
        like = input()
        if like == 'y':
            generated.append(fake_B)
            data_loader.dataset.choose()
        elif like=='done':
            break

    # concat generated into pano
    extend_length = int(2/3*256)
    pano = generated[0]/255.
    for i in range(1, len(generated)):
        pano_length = pano.shape[1]
        new_pano = np.zeros((256, pano_length + extend_length, 3))
        new_pano[:,:pano_length,:] = pano

        new_seg = np.zeros((256, pano_length + extend_length, 3))
        new_seg[:, -256:, :] = generated[i]/255.
        pano = np.maximum(new_pano, new_seg)

print('Save name: ')
name = input()
imsave('results/manual/sv/new_G/{}.jpg'.format(name), pano)
