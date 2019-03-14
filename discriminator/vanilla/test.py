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

# Must be one for patch_loss
batch_size = 1

device = torch.device("cuda:3")

# Shouldn't be patchGAN, add fc layer at end
#model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False, out_channels=256, glob=True)
#model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False)
#model = networks.Siamese()
model = networks.GlobalLocal()
#model = torch.nn.DataParallel(model, device_ids=(0,1,3,4))
#model = networks.SiameseResnet()
#chkpt = torch.load('checkpoints/patch/179.pth')
chkpt = torch.load('checkpoints/localGlobal/190.pth')
#chkpt = torch.load('checkpoints/SiameseResnet/17.pth')
model.load_state_dict(chkpt['state_dict'])
model.to(device)

patch_loss = networks.GANLoss(use_lsgan=False).to(device)

total_steps = 0

dataset = FrankensteinDataset()
#dataset.initialize('../datasets/street_view/sides/')
#dataset.initialize('../../../data/semanticLandscapes512/train_img')
dataset.initialize('../../../data/MITCVCL/imgs')

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

total = 0
correct = 0
real = []
fake = []
losses = []

model.eval()
#for i, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
for i, (data, target) in enumerate(train_loader):
    if i % 100 == 0:
        print(i)
    if i == 1000:
        break
    if type(data) is type({}):
        continue

    data, target = data.to(device), target.to(device)

    pred = model(data)

    loss = F.binary_cross_entropy_with_logits(pred, target)
    #pdb.set_trace()
    #loss = patch_loss(pred, target)

    pred = pred.mean() > .5
    if pred.item() == target.item():
        correct += 1
    total += 1

    '''
    pred = F.sigmoid(pred).mean(dim=(2,3)) > .5
    pred = pred > 0
    correct += (pred.float() == target).sum().item()
    '''

    if target == 0:
        fake.append((pred.float() == target).sum().item())
    else:
        real.append((pred.float() == target).sum().item())

    total += batch_size
    losses.append(loss.item())

    #pdb.set_trace()

print('Accuracy: {}%'.format(correct/float(total)))
print('Real acc: {}%'.format(np.mean(real)*100))
print('Fake acc: {}%'.format(np.mean(fake)*100))
