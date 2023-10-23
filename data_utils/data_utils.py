from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
import random
from numpy.random import randint
import numpy as np
import sys
import math
sys.path.append('..')

class ResizeDataset(Dataset):
    def __init__(self, parent_dataset, resolution):
        self.parent_dataset = parent_dataset
        self.resolution = resolution
    def __len__(self,):
        return self.parent_dataset.__len__()
    
    def __getitem__(self, global_idx):
        x, y = self.parent_dataset.__getitem__(global_idx)
        xp = F.interpolate(x[None,...] if len(x.shape) < 4 else x, size= self.resolution,mode = 'bicubic',align_corners=True)
        yp = F.interpolate(y[None,...] if len(y.shape) < 4 else y, size= self.resolution,mode = 'bicubic',align_corners=True)
        return torch.squeeze(xp), torch.squeeze(yp)
        

def mask_patches(size, drop_type='zeros', max_block=0.7, drop_pix=0.3,\
                 channel_per = 0.5, channel_drop_per = 0.2, device='cpu', min_block=10):
    #######################
    # max_block_sz: percentage of the maximum block to be dropped
    # 
    # funtion returns a mask with 0,1. Which is multiplied with the data tensor 
    # To generate masked sample
    # 
    #######################
    
    np.random.seed()    
    C, H, W = size
    mask = torch.ones(size, device = device)
    drop_t = drop_type
    augmented_channels = np.random.choice(C, math.ceil(C*channel_per))
    #print(augmented_channels)
    drop_len = int(channel_drop_per* math.ceil(C*channel_per))
    mask[augmented_channels[:drop_len], :, :] = 0.0
    for i in augmented_channels[drop_len:]:
        #print("Masking")
        n_drop_pix = drop_pix*H*W
        mx_blk_height = int(H*max_block)
        mx_blk_width = int(W*max_block)


        while n_drop_pix >0:
            rnd_r = random.randint(0, H-2)
            rnd_c = random.randint(0, W-2)

            rnd_h = min(random.randint(min_block, mx_blk_height), H-rnd_r)
            rnd_w = min(random.randint(min_block, mx_blk_width), W-rnd_c)
            mask[i, rnd_r:rnd_r+rnd_h, rnd_c:rnd_c+rnd_w] = 0
            n_drop_pix -= rnd_h*rnd_c
    #print("One data Done")   
    return None, mask   
def batched_masker(data_i, aug):
    data = torch.zeros_like(data_i)
    data.copy_(data_i)
    mask = []
    b,c,h,w = data.shape
    #print(data_i.device)
    for i in range(data.shape[0]):
        _,n = aug((c,h,w), device=data_i.device)
        mask.append(n)
    #print("loop done")
    masks = torch.stack(mask, dim = 0)
    #print("returning from augmenter")
    return data*masks, masks


class MakserNonuniformMest(object):
    def __init__(self, grid_non_uni, gird_uni, radius):
        self.grid_non_uni = grid_non_uni
        self.grid_uni = gird_uni
        dists = torch.cdist(gird_uni, grid_non_uni).to(gird_uni.device) # shaped num query points x num data points
        self.in_nbr = torch.where(dists <= radius, 1., 0.)

    def __call__(self, data_i, aug):
        print(self.in_nbr.shape)
        
