import time
import networks
import pdb
from data.frankenstein_dataset import FrankensteinDataset
from data.horizon_dataset import HorizonDataset
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
#model = networks.GlobalLocal()
#model = networks.SiameseResnet()
#chkpt = torch.load('checkpoints/localGlobalBlur2/65.pth')
chkpt = torch.load('checkpoints/patch/179.pth')
#chkpt = torch.load('checkpoints/SiameseResnet/17.pth')
model.load_state_dict(chkpt['state_dict'])
model.to(device)

patch_loss = networks.GANLoss()

total_steps = 0

#dataset = FrankensteinDataset()
dataset = HorizonDataset()
#dataset.initialize('../datasets/street_view/sides/', allrandom=True)
#dataset.initialize('../../../data/semanticLandscapes512/train_img', allrandom=True, return_idx=True)
#dataset.initialize('../../../data/MITCVCL/imgs', allrandom=True, return_idx=True)
dataset.initialize('../../../data/MITCVCL/coast', allrandom=True, return_idx=True)
#dataset.initialize('../../../data/MITCVCL/mountain', allrandom=True, return_idx=True)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

indices = []
preds = []

# 1000 ~ 25 sec
# 100000 ~ 2500 sec
model.eval()
for i in range(100):
#for i, (data, target) in enumerate(train_loader):
    if i % 100 == 0:
        print(i)
    data, target, idx_l, idx_r = dataset[i]      # Samples random pair
    data = data.unsqueeze(0)
    data, target = data.to(device), target.to(device)

    total_steps += batch_size

    pred = model(data)
    #pred = F.sigmoid(pred).mean(dim=(2,3))
    #loss = patch_loss(pred.cpu(), target)
    #loss = F.binary_cross_entropy(pred, target)

    indices.append((idx_l, idx_r))
    preds.append(pred.mean().item())


#ims = np.array(ims)
preds = np.array(preds)
preds = 1 / (1 + np.exp(-preds))

indices = [indices[i] for i in np.argsort(preds)]
preds = preds[np.argsort(preds)]

def convertImage(im):
    # undo right image flip when cat-ing
    im = np.concatenate((im[:3,:,:], im[3:,:,::-1]), 2)
    return im

def savePaths(indices):
    paths = ''
    for idx_l, idx_r in indices:
        paths += dataset.get_path_name(idx_l).split('/')[-1]
        paths += ', '
        paths += dataset.get_path_name(idx_r).split('/')[-1]
        paths += '\n'

    f = open("samples/indices.txt", "w")
    f.write(paths)
    f.close()

np.savetxt('samples/preds.txt', preds)
savePaths(indices)

for i in range(1,101):
    data, target = dataset.get_deterministic(indices[-i][0], indices[-i][1])
    im = data.numpy()
    imsave('samples/best/{}.jpg'.format(i), convertImage(im).transpose(1,2,0))
for i in range(100):
    data, target = dataset.get_deterministic(indices[i][0], indices[i][1])
    im = data.numpy()
    imsave('samples/worst/{}.jpg'.format(i), convertImage(im).transpose(1,2,0))
