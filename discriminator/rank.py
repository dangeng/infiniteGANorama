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

model = networks.define_D(3, 64, 'n_layers', n_layers_D=3, use_sigmoid=True)      # Shouldn't be patchGAN, add fc layer at end
chkpt = torch.load('checkpoints/99.pth')
model.load_state_dict(chkpt['state_dict'])
model.to(device)

total_steps = 0

dataset = FrankensteinDataset()
dataset.initialize('../datasets/street_view/sides/', allrandom=True)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

ims = []
preds = []

for epoch in range(100):
    epoch_iter = 0

    model.train()
    print('Epoch: {}'.format(epoch))
    for i, (data, target) in enumerate(train_loader):
        print(i)
        data, target = data.to(device), target.to(device)

        total_steps += batch_size
        epoch_iter += batch_size

        pred = model(data)
        loss = F.binary_cross_entropy(pred, target)

        ims.append(data.detach().cpu().numpy())
        preds.append(pred.item())

        if i == 10000:
            break

    ims = np.array(ims)
    preds = np.array(preds)

    ims = ims[np.argsort(preds)]
    preds = preds[np.argsort(preds)]

    break

for i in range(100):
    imsave('samples/{}.jpg'.format(i), ims[-i][0].transpose(1,2,0))
