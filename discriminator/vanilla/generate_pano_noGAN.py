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
NUM_SLICES = 10

device = torch.device("cuda")

model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False)
chkpt = torch.load('checkpoints/patch/179.pth')
model.load_state_dict(chkpt['state_dict'])
model.to(device)

patch_loss = networks.GANLoss()

dataset = EvalDataset()
#dataset.initialize('../../../data/semanticLandscapes512/train_img', allrandom=True, return_idx=True)
dataset.initialize('../../../data/MITCVCL/coast', allrandom=True)


def convertImage(im):
    # undo right image flip when cat-ing
    im = np.concatenate((im[:3,:,:], im[3:,:,::-1]), 2)
    return im.transpose(1,2,0)

def convertPano(best_params):
    # Set left_aux to original left_aux
    dataset.left_aux = best_params[0]

    # Get original pair and use it to start pano
    data, aux = dataset.get_deterministic(best_params[1])
    pair = data.numpy()
    pano = np.concatenate((pair[:3,:,:], pair[3:,:,::-1]), 2)

    # Iterate through rest of the pairs
    for i in range(2, len(best_params)):
        data, aux = dataset.get_deterministic(best_params[i])
        pair = data.numpy()
        pano = np.concatenate((pano, pair[3:,:,::-1]), 2)
    return pano.transpose(1,2,0)

best_params = [dataset.left_aux]
used = [dataset.left_aux['idx']]

model.eval()
for im_idx in range(NUM_SLICES):
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

    # Softmax preds (not exactly necessary here actually)
    preds = np.array(preds)
    preds = 1 / (1 + np.exp(-preds))

    # Argsort image pairs
    crop_params = [crop_params[i] for i in np.argsort(preds)]
    indices = np.arange(len(dataset))[np.argsort(preds)]
    preds = preds[np.argsort(preds)]

    # Get best pair
    best_index = indices[-1]
    best_param = crop_params[-1]
    data, aux = dataset.get_deterministic(best_param)

    # Save stuff
    used.append(best_index)
    best_params.append(best_param)
    #imsave('pano/test_{}.jpg'.format(im_idx), convertImage(data.numpy()))
    #np.save('pano/test_{}.npy'.format(im_idx), data.numpy())

    # Set left params for next round
    dataset.set_left(best_param)

# Fine tune horizons
# `best_params` contains all info needed for each image slice
dataset.left_aux = best_params[0]
best_params_horizons = [best_params[0]]
best_preds = []

# Iterate through all slices
for i in range(1, len(best_params)):
    preds = []
    crop_params = []

    # Iterate through all possible y_crops
    for params in dataset.y_offsets(best_params[i]):
        data, aux = dataset.get_deterministic(params)
        data = data.unsqueeze(0)
        data = data.to(device)

        pred = model(data)

        preds.append(pred.mean().item())
        crop_params.append(aux)
        imsave('pano/{}_{}.jpg'.format(i, params['y_crop']), convertImage(data.detach().cpu().numpy()[0]))

    # Softmax preds (not exactly necessary)
    preds = np.array(preds)
    preds = 1 / (1 + np.exp(-preds))

    np.save('pano/preds_{}.npy'.format(i), preds)

    # Argsort image pairs
    crop_params = [crop_params[i] for i in np.argsort(preds)]
    indices = np.arange(len(dataset))[np.argsort(preds)]
    preds = preds[np.argsort(preds)]

    # Replace best_params[i]
    best_param = crop_params[-1]
    best_params_horizons.append(best_param)
    dataset.left_aux = best_param
    best_preds.append(preds[-1])

print(best_params)
print(best_params_horizons)
print(best_preds)

imsave('pano/pano.jpg'.format(im_idx), convertPano(best_params_horizons))
