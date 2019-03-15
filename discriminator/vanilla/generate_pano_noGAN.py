import time
import networks
import pdb
from data.frankenstein_dataset import FrankensteinDataset
from data.horizon_dataset import HorizonDataset
from data.eval_dataset import EvalDataset
import matplotlib.pyplot as plt
from scipy.misc import imsave
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import numpy as np

batch_size = 1

device = torch.device("cuda")

model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False)
chkpt = torch.load('checkpoints/patch/179.pth')
model.load_state_dict(chkpt['state_dict'])
model.to(device)

patch_loss = networks.GANLoss()

dataset = EvalDataset()
#dataset.initialize('../../../data/semanticLandscapes512/train_img', allrandom=True, return_idx=True)
dataset.initialize('../../../data/MITCVCL/mountain', allrandom=True)


def convertImage(im):
    # undo right image flip when cat-ing
    im = np.concatenate((im[:3,:,:], im[3:,:,::-1]), 2)
    return im.transpose(1,2,0)

def convertPano(ims):
    pano = np.concatenate((ims[0][:3,:,:], ims[0][3:,:,::-1]), 2)
    for i in range(1, len(ims)):
        pano = np.concatenate((pano, ims[i][3:,:,::-1]), 2)
    return pano.transpose(1,2,0)

ims = []    # Dangerous! Don't overload RAM!
used = []

model.eval()
for im_idx in range(10):
    preds = []
    crop_params = []

    # Look at all pairs in dataset
    for i in range(len(dataset)):
        if i % 100 == 0:
            print(i)

        # Skips same image
        if i == dataset.left_aux['idx'] or i in used:
            preds.append(0)
            crop_params.append({})
            continue

        data, aux = dataset[i]
        data = data.unsqueeze(0)
        data = data.to(device)

        pred = model(data)

        preds.append(pred.mean().item())
        crop_params.append(aux)

    # Softmax preds
    preds = np.array(preds)
    preds = 1 / (1 + np.exp(-preds))

    # Argsort image pairs
    crop_params = [crop_params[i] for i in np.argsort(preds)]
    indices = np.arange(len(dataset))[np.argsort(preds)]
    preds = preds[np.argsort(preds)]

    # Get best pair
    best_index = indices[-1]
    best_params = crop_params[-1]
    data, aux = dataset.get_deterministic(best_index, best_params)

    # Save stuff
    used.append(best_index)
    ims.append(data.numpy())
    imsave('pano/test_{}.jpg'.format(im_idx), convertImage(data.numpy()))
    np.save('pano/test_{}.npy'.format(im_idx), data.numpy())

    # Set left params for next round
    dataset.set_left(best_params)

imsave('pano/pano.jpg'.format(im_idx), convertPano(ims))

