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

from tensorboardX import SummaryWriter

batch_size = 1

writer = SummaryWriter()

device = torch.device("cuda")

model = networks.define_D(3, 64, 'n_layers', n_layers_D=3, use_sigmoid=True)      # Shouldn't be patchGAN, add fc layer at end
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)

total_steps = 0

dataset = FrankensteinDataset()
dataset.initialize('../datasets/street_view/sides/')

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(100):
    epoch_iter = 0

    model.train()
    print('Epoch: {}'.format(epoch))
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        print(data.shape)
        raise Exception

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
