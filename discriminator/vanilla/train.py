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

# Must be one for patch_loss
batch_size = 32

writer = SummaryWriter(comment="CorrectBatchNorm")

device = torch.device("cuda:4")

# Shouldn't be patchGAN, add fc layer at end
#model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False, out_channels=256)
model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False)
#model = networks.Siamese()
model.to(device)
#model = torch.nn.DataParallel(model, device_ids=(0,1,3,4))

patch_loss = networks.GANLoss(use_lsgan=False).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

total_steps = 0

dataset = FrankensteinDataset()
#dataset.initialize('../datasets/street_view/sides/')
dataset.initialize('../../../data/semanticLandscapes512/train_img')

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Just train forever
for epoch in range(100000000000):
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
        #pdb.set_trace()

        #loss = F.binary_cross_entropy(pred, target)
        loss = patch_loss(pred, target)
        loss.backward()
        optimizer.step()

        if i % 10==0:
            writer.add_scalar('loss', loss.item(),total_steps)

    torch.save({'state_dict': model.state_dict()}, 'checkpoints_bn/{}.pth'.format(epoch))
