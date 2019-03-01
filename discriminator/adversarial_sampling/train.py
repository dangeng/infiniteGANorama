import time
import networks
import pdb
from data.frankenstein_dataset import FrankensteinDataset
import matplotlib.pyplot as plt
from scipy.misc import imsave
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch
from tqdm import tqdm
import numpy as np

from tensorboardX import SummaryWriter

batch_size = 1
ds_size = 200

writer = SummaryWriter()

device = torch.device("cuda:6")

net_D = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False, out_channels=128)
net_D.to(device)
optimizer_D = optim.SGD(net_D.parameters(), lr=0.0001, momentum=0.5)
chkpt = torch.load('../vanilla/checkpoints_/63.pth')
net_D.load_state_dict(chkpt['state_dict'])

def softmax2d(tensor):
    '''
    Pytorch doesn't natively implement 2d softmax???
    I am saddened!
    '''
    return F.softmax(tensor.view(-1)).view_as(tensor)

def normalize_G(net_G):
    '''
    Normalizes the off diagonal and on diagonal entries of net_G 
    to sum to one respectively
    We need to return an optimizer as well, because when
    normalizing we effectively overwrite the original tensor
    '''
    # Turn off grad tracking (makes life easier)
    net_G.requires_grad = False

    # Normalize off/on diagonal parts of G
    eye = torch.eye(ds_size).cuda()
    norm = (net_G*(1 - eye)).sum()
    off_diagonal = net_G*(1-eye)/norm
    on_diagonal = eye/float(ds_size)
    net_G = off_diagonal + on_diagonal

    # Set new G to track grads, make new optimizer
    net_G.requires_grad = True
    opt_G = optim.SGD([net_G], lr=0.0001, momentum=0.5)

    return net_G, opt_G

net_G = torch.zeros((ds_size, ds_size), device=device)
net_G -= torch.eye(ds_size, device=device) * 1e6   # diagonals have no weight
optimizer_G = optim.SGD([net_G], lr=100, momentum=0.5)
net_G.requires_grad = True
#net_G, optimizer_G = normalize_G(net_G)

dataset = FrankensteinDataset()
# This dataset is sorted in lexiographic order, not numerical
dataset.initialize('../../../data/semanticLandscapes/train_img', length=ds_size)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_idx(idx):
    data, target = dataset[idx]

    if type(data) is type({}):
        return  # Abort if data is bad

    # Preprocess stuff
    target = target.float()
    data.unsqueeze_(0)

    # To gpu and get sample index
    data, target = data.to(device), target.to(device)
    idx_l, idx_r = np.unravel_index(idx, net_G.shape)

    # Optimize discriminator
    optimizer_D.zero_grad()
    pred = net_D(data)
    d_loss = F.binary_cross_entropy(pred, target)

    # If same image, weight by 1/ds_size
    # If fake image, weight by G
    if target == 1:
        d_loss *= 1/float(ds_size)
    else:
        d_loss *= softmax2d(net_G)[idx_l, idx_r].detach() # detach this?

    d_loss.backward(retain_graph=True)  # Need to compute another backward
    optimizer_D.step()

    # Optimize generator (if frankenstein image)
    if target == 0:
        optimizer_G.zero_grad()
        # calculate pred again?
        valid = 1 - target      # We want to trick D
        g_loss = F.binary_cross_entropy(pred, valid)    
        g_loss *= softmax2d(net_G)[idx_l, idx_r]   # Weight the loss
        g_loss.backward()
        optimizer_G.step()
        return d_loss.item(), g_loss.item()

    # Just return if we trained on real example
    return

total_steps = 0
prev_d_loss, prev_g_loss = 0,0
for epoch in range(100):
    epoch_iter = 0

    net_D.train()
    print('Epoch: {}'.format(epoch))
    for idx in tqdm(range(ds_size*ds_size)):
        total_steps += batch_size
        epoch_iter += batch_size

        # Train on random real image
        real_idx = np.random.randint(ds_size)
        train_idx(real_idx * ds_size + real_idx)

        # Train on random frankenstein pair
        losses = train_idx(idx)

        # Sometimes losses will be none (if we didn't train G,
        # idx is a single real image, or the dataloading failed)
        if losses is not None:
            prev_d_loss, prev_g_loss = losses

        if idx % 10==0:
            writer.add_scalar('d_loss', prev_d_loss, total_steps)
            writer.add_scalar('g_loss', prev_g_loss, total_steps)

    # Normalize G params
    #net_G, optimizer_G = normalize_G(net_G)

    torch.save({'discriminator_state_dict': net_D.state_dict(), 'generator_state': net_G}, 'checkpoints_200/{}.pth'.format(epoch))
