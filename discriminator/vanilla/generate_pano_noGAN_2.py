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

# Panorama parameters
batch_size = 1
NUM_SLICES = 3
SAMPLES=5           # Number of horizon offsets to sample before finetuning

# Set up deep learning stuff
device = torch.device("cuda")
model = networks.define_D(6, 64, 'n_layers', n_layers_D=3, use_sigmoid=False)
chkpt = torch.load('checkpoints/patch_horizon/49.pth')
model.load_state_dict(chkpt['state_dict'])
model.to(device)
patch_loss = networks.GANLoss()

# Create dataset
dataset = EvalDataset()
#dataset.initialize('../../../data/semanticLandscapes512/train_img', allrandom=True, return_idx=True)
#dataset.initialize('../../../data/MITCVCL/coast', allrandom=True)
dataset.initialize('../../../data/instagram_landscapes/anne_karin_69', allrandom=True)

def convertImage(im):
    '''
    Takes output of dataset and returns a np image
    '''
    # undo right image flip when cat-ing
    im = np.concatenate((im[:3,:,:], im[3:,:,::-1]), 2)
    return im.transpose(1,2,0)

def convertPano(best_params):
    '''
    Takes in a list of slice parameters (dicts) and returns a panorama (np image)
    '''
    # Save array of images too (for inpainting)
    ims = []

    # Set left_aux to original left_aux
    dataset.left_aux = best_params[0]

    # Get original pair and use it to start pano
    data, aux = dataset.get_deterministic(best_params[1])
    pair = data.numpy()
    ims.append(pair)
    pano = np.concatenate((pair[:3,:,:], pair[3:,:,::-1]), 2)

    # Iterate through rest of the pairs
    for i in range(2, len(best_params)):
        dataset.left_aux = best_params[i-1]
        data, aux = dataset.get_deterministic(best_params[i])
        pair = data.numpy()
        ims.append(pair)
        pano = np.concatenate((pano, pair[3:,:,::-1]), 2)

    # Save ims
    np.save('pano/ims.npy', np.array(ims))

    return pano.transpose(1,2,0)

def get_best_offsets(left_params, right_params, samples=None):
    '''
    Takes in two slice parameters (dicts), return the best offsets given a pair
    '''
    # Set left image
    dataset.left_aux = left_params

    preds = []
    crop_params = []

    # This doesn't exactly give `samples` number of indices, but close enough...
    if samples is not None:
        num_y_offsets = dataset.get_num_y_offsets(right_params)
        step = num_y_offsets // samples
        test_indices = np.arange(num_y_offsets)[step::step]

    # Iterate through all possible y_crops of right image
    for i, params in enumerate(dataset.y_offsets(right_params)):
        if samples is not None:
            if i not in test_indices:
                continue

        data, aux = dataset.get_deterministic(params)
        data = data.unsqueeze(0)
        data = data.to(device)

        pred = model(data)

        preds.append(pred.mean().item())
        crop_params.append(aux)
        #imsave('pano/{}_{}.jpg'.format(i, params['y_crop']), convertImage(data.detach().cpu().numpy()[0]))

    # Softmax preds (not exactly necessary)
    preds = np.array(preds)
    preds = 1 / (1 + np.exp(-preds))

    #np.save('pano/preds_{}.npy'.format(i), preds)

    # Argsort image pairs
    crop_params = [crop_params[i] for i in np.argsort(preds)]
    indices = np.arange(len(dataset))[np.argsort(preds)]
    preds = preds[np.argsort(preds)]

    # Replace best_params[i]
    best_param = crop_params[-1]
    best_pred = preds[-1]

    return best_pred, best_param


best_params = [dataset.left_aux]
used = [dataset.left_aux['idx']]

model.eval()
for im_idx in range(NUM_SLICES):
    preds = []
    crop_params = []

    # Look at all pairs in dataset
    for i in range(len(dataset)):
        if i % 100 == 0:
            #print(i)
            pass

        # Skips same image or image if already used
        if i == dataset.left_aux['idx'] or i in used:
            preds.append(0)
            crop_params.append({})
            continue

        right_params = {'idx': i,
                        'x_crop': 100, # arbitrary, should change later
                        'y_crop': 0}

        best_pair_pred, best_pair_params = \
                get_best_offsets(dataset.left_aux, right_params, samples=SAMPLES)

        preds.append(best_pair_pred)
        crop_params.append(best_pair_params)

    # Argsort image pairs
    preds = np.array(preds)
    crop_params = [crop_params[i] for i in np.argsort(preds)]
    indices = np.arange(len(dataset))[np.argsort(preds)]
    preds = preds[np.argsort(preds)]

    print('Prefintune:')
    print(preds[-5:])
    print(crop_params[-1])

    # Finetune each pair above .75 or just first one
    to_fine_tune = np.array(crop_params)[preds > .75]
    if len(to_fine_tune) == 0:
        to_fine_tune = crop_params[-1:]
    else:
        to_fine_tune = to_fine_tune[-5:]

    preds = []
    crop_params = []
    for params in to_fine_tune:
        best_pair_pred, best_pair_params = \
                get_best_offsets(dataset.left_aux, params)
        preds.append(best_pair_pred)
        crop_params.append(best_pair_params)

    # Argsort image pairs
    preds = np.array(preds)
    crop_params = [crop_params[i] for i in np.argsort(preds)]
    indices = np.arange(len(dataset))[np.argsort(preds)]
    preds = preds[np.argsort(preds)]

    # Get best pair
    candidates = np.array(crop_params)[preds > .95]
    if len(candidates) == 0:
        candidates = [crop_params[-1]]
    best_param = np.random.choice(candidates)
    best_index = best_param['idx']
    data, aux = dataset.get_deterministic(best_param)

    print('Finetune:')
    print(preds)
    print(best_param)
    print('\n')

    # Save stuff
    used.append(best_index)
    best_params.append(best_param)
    #imsave('pano/prefinetune_{}.jpg'.format(im_idx), convertImage(data.numpy()))
    #np.save('pano/test_{}.npy'.format(im_idx), data.numpy())

    # Set left params for next round
    dataset.set_left(best_param)

'''
for param in best_params:
    print(param)

# Finetune panorama
best_params_finetuned = [best_params[0]]
best_preds_finetuned = []

for i in range(len(best_params) - 1):
    best_pair_pred, best_pair_params = \
            get_best_offsets(best_params[i], best_params[i+1])
    best_params_finetuned.append(best_pair_params)
    best_preds_finetuned.append(best_pair_pred)

    data, aux = dataset.get_deterministic(best_pair_params)
    imsave('pano/finetune_{}.jpg'.format(im_idx), convertImage(data.numpy()))

for param in best_params_finetuned:
    print(param)
'''

imsave('pano/pano.jpg'.format(im_idx), convertPano(best_params))
#print(best_param)
