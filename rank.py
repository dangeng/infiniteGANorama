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

# Generates a ton of images and ranks them in order of
# discriminator loss

if __name__ == '__main__':
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)

    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    total_steps = 0

    chkpt_D = torch.load('checkpoints/streetview_pangan_2/10_net_D.pth')
    chkpt_G = torch.load('checkpoints/streetview_pangan_2/10_net_G.pth')

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

    samples = []

    for i, data in enumerate(dataset):
        if i < 100:
            print(i)
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
            samples.append([loss, fake_B])
        else:
            break
    samples.sort(key=lambda x: x[0])

def plotim(im):
    plt.imshow(im)
    plt.show()
