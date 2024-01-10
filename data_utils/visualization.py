import gc
import logging
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data

from data_utils.hdf5_datasets import Equation
from models.codano import CodANO, CoDANOTemporal
from models.get_models import StageEnum

N_ROWS = 3  # ground_truth, prediction, and error

def show_data_diff(
    ground_truth: torch.Tensor(),
    prediction: torch.Tensor(),
    channel: int,
    n_cols=5,  # used as time axis
    logger: Optional[logging.Logger] = None
):
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(logging.WARNING)  # plt is noisy on [DEBUG]

    fig, axs = plt.subplots(
        N_ROWS,
        n_cols + 1,
        figsize=(13, 8),  # (width, height)
        subplot_kw={'xticks': [], 'yticks': []},
    )

    x = ground_truth[channel].cpu()
    v_min = x.min()
    v_max = x.max()
    row = 0
    for c in range(n_cols):
        ax = axs[row, c]
        _x = x[c]
        im = ax.imshow(_x, vmin=v_min, vmax=v_max)
    fig.colorbar(im, ax=axs[row, n_cols], location='left')

    y = prediction[channel].cpu().detach().numpy()
    v_min = y.min()
    v_max = y.max()
    row = 1
    for c in range(n_cols):
        ax = axs[row, c]
        _y = y[c]
        im = ax.imshow(_y, vmin=v_min, vmax=v_max)
    fig.colorbar(im, ax=axs[row, n_cols], location='left')

    error2 = np.square(x - y)
    v_min = error2.min()
    v_max = error2.max()
    row = 2
    for c in range(n_cols):
        ax = axs[row, c]
        _error2 = error2[c].cpu().detach().numpy()
        im = ax.imshow(_error2, vmin=v_min, vmax=v_max)
    fig.colorbar(im, ax=axs[row, n_cols], location='left')
    # TODO this makes the color last ax column be awkwardly half empty. fix.

    plt.tight_layout()
    plt.show()


def show_multi_physics_data_diffs(
    model: Union[CodANO, CoDANOTemporal],
    data_loader: data.DataLoader,
    stage: Optional[StageEnum] = None,
    logger: Optional[logging.Logger] = None,
):
    swe_loss, diff_loss, ns_loss = get_multi_physics_data_losses(
        model,
        data_loader,
        # TODO add these indexes to config
        ((0, 125), (125, 250), (250, 300)),
        stage=stage,
        logger=logger,
    )

    print("WATER DEPTH (SHALLOW WATER)")
    (depth, _), _ = data_loader.dataset[swe_loss[0]]
    show_data_diff(
        depth,
        swe_loss[2],
        channel=0,
        logger=logger,
    )

    print("ACTIVATOR (DIFFUSION-REACTION)")
    (activator, _), _ = data_loader.dataset[diff_loss[0]]
    show_data_diff(
        activator,
        diff_loss[2],
        channel=1,
        logger=logger,
    )

    print("INHIBITOR (DIFFUSION-REACTION)")
    (inhibitor, _), _ = data_loader.dataset[diff_loss[0]]
    show_data_diff(
        inhibitor,
        diff_loss[2],
        channel=2,
        logger=logger,
    )

    print("PARTICLE DENSITY (NAVIER-STOKES)")
    (particles, _), _ = data_loader.dataset[ns_loss[0]]
    show_data_diff(
        particles,
        ns_loss[2],
        channel=3,
        logger=logger,
    )

    print("X-VELOCITY (NAVIER-STOKES)")
    (velocity_x, _), _ = data_loader.dataset[ns_loss[0]]
    show_data_diff(
        velocity_x,
        ns_loss[2],
        channel=4,
        logger=logger,
    )

    print("Y-VELOCITY (NAVIER-STOKES)")
    (velocity_y, _), _ = data_loader.dataset[ns_loss[0]]
    show_data_diff(
        velocity_y,
        ns_loss[2],
        channel=5,
        logger=logger,
    )
