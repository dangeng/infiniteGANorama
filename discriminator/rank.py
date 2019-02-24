import time
import networks
import pdb
from data.frankenstein_dataset import FrankensteinDataset
import matplotlib.pyplot as plt
from scipy.misc import imsave
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np

batch_size = 1

device = torch.device("cuda")

model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False, out_channels=128)      # Shouldn't be patchGAN, add fc layer at end
chkpt = torch.load('checkpoints/30.pth')
model.load_state_dict(chkpt['state_dict'])
model.to(device)

total_steps = 0

dataset = FrankensteinDataset()
#dataset.initialize('../datasets/street_view/sides/', allrandom=True)
dataset.initialize('../../data/semanticLandscapes/train_img', allrandom=True)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

ims = []
preds = []


for i, (data, target) in enumerate(train_loader):
    print(i)
    data, target = data.to(device), target.to(device)

    total_steps += batch_size

    pred = model(data)
    loss = F.binary_cross_entropy(pred, target)

    ims.append(data.detach().cpu().numpy())
    preds.append(pred.item())

    if i == 10000:
        break

#ims = np.array(ims)
preds = np.array(preds)

ims = [ims[i] for i in np.argsort(preds)]
preds = preds[np.argsort(preds)]

def convertImage(im):
    im = np.concatenate((im[:3,:,:], im[3:,:,:]), 2)
    return im

np.savetxt('samples/preds.txt', preds)
for i in range(100):
    imsave('samples/{}.jpg'.format(i), convertImage(ims[-i][0]).transpose(1,2,0))
for i in range(9900,10000):
    imsave('samples/{}.jpg'.format(i), convertImage(ims[-i][0]).transpose(1,2,0))
