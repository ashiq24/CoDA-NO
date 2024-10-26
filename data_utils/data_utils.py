import math
import random
import sys
from typing import Optional, Tuple
from utils import *
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

sys.path.append('..')


class ResizeDataset(Dataset):
    def __init__(self, parent_dataset, resolution):
        self.parent_dataset = parent_dataset
        self.resolution = resolution

    def __len__(self,):
        return self.parent_dataset.__len__()

    def __getitem__(self, global_idx):
        x, y = self.parent_dataset.__getitem__(global_idx)
        xp = F.interpolate(
            x[None, ...] if len(x.shape) < 4 else x,
            size=self.resolution,
            mode='bicubic',
            align_corners=True,
        )
        yp = F.interpolate(
            y[None, ...] if len(y.shape) < 4 else y,
            size=self.resolution,
            mode='bicubic',
            align_corners=True,
        )
        return torch.squeeze(xp), torch.squeeze(yp)


class MaskerUniform:
    """Performs masking on datasets with regular meshes.

    For masking with data points from data sets with irregular meshes,
    use ``MaskerNonuniformMesh`` (below).
    """

    def __init__(
        self,
        drop_type='zeros',
        max_block=0.7,
        drop_pix=0.3,
        channel_per=0.5,
        channel_drop_per=0.2,
        device='cpu',
        min_block=30,
    ):
        self.drop_type = drop_type
        self.max_block = max_block
        self.drop_pix = drop_pix
        self.channel_per = channel_per
        self.channel_drop_per = channel_drop_per
        self.device = device
        self.min_block = min_block

    def __call__(self, size):
        """Returns a mask to be multiplied into a data tensor.

        Generates a binary mask of 0s and 1s to be point-wise multiplied into a
        data tensor to create a masked sample. By training on masked data, we
        expect the model to be resilient to missing data.

        TODO arg ``max_block_sz`` is outdated
        Parameters
        ---
        max_block_sz: percentage of the maximum block to be dropped
        """

        np.random.seed()
        C, H, W = size
        mask = torch.ones(size, device=self.device)
        drop_t = self.drop_type
        if self.channel_per == 1.0:
            augmented_channels = [i for i in range(C)]
        else:
            augmented_channels = np.random.choice(
                C, math.ceil(C * self.channel_per))
        drop_len = int(self.channel_drop_per * math.ceil(C * self.channel_per))
        mask[augmented_channels[:drop_len], :, :] = 0.0
        for i in augmented_channels[drop_len:]:
            n_drop_pix = self.drop_pix * H * W
            mx_blk_height = int(H * self.max_block)
            mx_blk_width = int(W * self.max_block)

            repetation = 3
            while n_drop_pix > 0 and repetation > 0:
                rnd_r = random.randint(0, H - 2)
                rnd_c = random.randint(0, W - 2)

                rnd_h = min(
                    random.randint(self.min_block, mx_blk_height),
                    H - rnd_r
                )
                rnd_w = min(
                    random.randint(self.min_block, mx_blk_width),
                    W - rnd_c
                )
                mask[i, rnd_r:rnd_r + rnd_h, rnd_c:rnd_c + rnd_w] = 0
                n_drop_pix -= rnd_h * rnd_c
                repetation -= 1
        return None, mask


def batched_masker(
    data: torch.Tensor,  # with a batch dimension
    aug,  # instance of a Masker like above
    batched_channels: Optional[Tuple[Tuple[int]]] = None,
):
    mask = []
    aug.device = data.device
    for i in range(data.shape[0]):
        _, m = aug(data[i].shape)
        mask.append(m)

    masks = torch.stack(mask, dim=0)
    return data * masks, masks


# Nearest neighbour type masking for non-uniform 
# mesh

class MaskerNonuniformMesh(object):
    def __init__(
        self,
        grid_non_uni,
        gird_uni,
        radius,
        drop_type='zeros',
        drop_pix=0.3,
        channel_aug_rate=0.7,
        channel_drop_rate=0.2,
        device='cpu',
        max_block=10,
        verbose=None
    ):
        """
        Parameters
        ---
        drop_type : dropped pixels are filled with zeros
        drop_pix: Percentage of pixels to be dropped
        channel_aug_rate: Percentage of channels to be augmented
        channel_drop_rate: Percentage of (channel_aug_rate) channels to be masked completely (drop all pxiels)
        max_block: Maximum number of regions to be dropped.
        """
        self.grid_non_uni = grid_non_uni
        self.grid_uni = gird_uni
        dists = torch.cdist(gird_uni, grid_non_uni).to(
            gird_uni.device)  # shaped num query points x num data points
        self.in_nbr = torch.where(dists <= radius, 1., 0.).long()

        self.drop_type = drop_type
        self.drop_pix = drop_pix
        self.channel_aug_rate = channel_aug_rate
        self.channel_drop_rate = channel_drop_rate
        self.device = device
        self.max_block = max_block
        self.verbose = verbose

    def __call__(self, size):

        L, C = size
        mask = torch.ones(size, device=self.device)
        drop_t = self.drop_type  # no effect now

        augmented_channels = np.random.choice(
            C, math.ceil(C * self.channel_aug_rate))

        p = random.random()
        # with 50% probability, drop all pixels in very few channels
        # or doing masking to many channels
        if p < 0.5:
            drop_len = int(
                self.channel_drop_rate *
                math.ceil(
                    C *
                    self.channel_aug_rate))
            mask[:, augmented_channels[:drop_len]] = 0.0
            return None, mask
        else:
            drop_len = 0
        # print('masking', augmented_channels[drop_len:])
        for i in augmented_channels[drop_len:]:
            n_drop_pix = self.drop_pix * L
            max_location = self.max_block
            while n_drop_pix > 0:
                # python random is inclusive of low and high
                j = random.randint(0, self.in_nbr.shape[0] - 1)
                mask[self.in_nbr[j] == 1, i] = 0
                n_drop_pix -= sum(self.in_nbr[j]).float()
                max_location -= 1
                if max_location == 0:
                    break
        return None, mask


def get_meshes(params, grid_size):
    mesh = get_mesh(params)
    input_mesh = torch.from_numpy(mesh).type(torch.float).cuda()

    minx, maxx = np.min(mesh[:, 0]), np.max(mesh[:, 0])
    miny, maxy = np.min(mesh[:, 1]), np.max(mesh[:, 1])

    size_x, size_y = grid_size
    idx_x = torch.arange(start=minx,
                         end=maxx + (maxx - minx) / size_x - 1e-5,
                         step=(maxx - minx) / (size_x - 1))
    idx_y = torch.arange(start=miny,
                         end=maxy + (maxy - miny) / size_y - 1e-5,
                         step=(maxy - miny) / (size_y - 1))
    x, y = torch.meshgrid(idx_x, idx_y, indexing='ij')
    output_mesh = torch.transpose(torch.stack(
        [x.flatten(), y.flatten()]), 0, 1).type(torch.float).cuda()

    return input_mesh, output_mesh

def get_mesh_displacement(x):
    """
    Only returns the displacement field.
    """
    return x[:, -2:].clone().detach()
