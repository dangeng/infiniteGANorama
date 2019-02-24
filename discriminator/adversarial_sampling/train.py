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

from tensorboardX import SummaryWriter

batch_size = 1
ds_size = 1000

writer = SummaryWriter()

device = torch.device("cuda")

# Shouldn't be patchGAN, add fc layer at end
model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False, out_channels=128)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)

total_steps = 0

dataset = FrankensteinDataset()
# This dataset is sorted in lexiographic order, not numerical
dataset.initialize('../../../data/semanticLandscapes/train_img', length=ds_size)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_idx(idx):
    data,target = dataset(idx)
    data, target = data.to(device), target.to(device)

for epoch in range(100):
    epoch_iter = 0

    model.train()
    print('Epoch: {}'.format(epoch))
    for i, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if type(data) is type({}):
            continue

        data, target = data.to(device), target.to(device)

        total_steps += batch_size
        epoch_iter += batch_size

        optimizer.zero_grad()
        pred = model(data)
        loss = F.binary_cross_entropy(pred, target)
        loss.backward()
        optimizer.step()


        if i % 10==0:
            writer.add_scalar('loss', loss.item(),total_steps)

    torch.save({'state_dict': model.state_dict()}, 'checkpoints/{}.pth'.format(epoch))
