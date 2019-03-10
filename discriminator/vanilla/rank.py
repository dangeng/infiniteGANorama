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

#model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False, out_channels=256, glob=True)
model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False)
#model = networks.Siamese()
chkpt = torch.load('checkpoints_fast/133.pth')
model.load_state_dict(chkpt['state_dict'])
model.to(device)

patch_loss = networks.GANLoss()

total_steps = 0

dataset = FrankensteinDataset()
#dataset.initialize('../datasets/street_view/sides/', allrandom=True)
dataset.initialize('../../../data/semanticLandscapes512/train_img', allrandom=True)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

ims = []
preds = []


# 1000 ~ 25 sec
# 100000 ~ 2500 sec
for i in range(10000):
#for i in range(10000):
#for i, (data, target) in enumerate(train_loader):
    if i % 100 == 0:
        print(i)
    data, target = dataset[0]      # Samples random pair
    data = data.unsqueeze(0)
    data, target = data.to(device), target.to(device)

    total_steps += batch_size

    pred = model(data)
    loss = patch_loss(pred.cpu(), target)
    #loss = F.binary_cross_entropy(pred, target)

    ims.append(data.detach().cpu().numpy())
    preds.append(pred.mean().item())


#ims = np.array(ims)
preds = np.array(preds)

ims = [ims[i] for i in np.argsort(preds)]
preds = preds[np.argsort(preds)]

def convertImage(im):
    im = np.concatenate((im[:3,:,:], im[3:,:,:]), 2)
    return im

np.savetxt('samples/preds.txt', preds)
for i in range(100):
    imsave('samples/best/{}.jpg'.format(i), convertImage(ims[-i][0]).transpose(1,2,0))
for i in range(100):
    imsave('samples/worst/{}.jpg'.format(i), convertImage(ims[i][0]).transpose(1,2,0))
