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
from train.trainer import multi_physics_loss, MAP_EQUATION_TO_CHANNELS


# When the model performs badly, HOW is it performing badly?
# because of masking, I need to remember the model output, which usually still
# has edges in it from being masked.
# TODO get masks from model wrapper and save those during train/test (opt)
def get_multi_physics_data_losses(
    model: Union[CodANO, CoDANOTemporal],
    data_loader: data.DataLoader,
    domains: Tuple[Tuple[int, int], ...],
    stage: Optional[StageEnum] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[int, float, torch.Tensor]]:
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # If `stage is None`, assume `model` has been set to the apt stage
    # by the caller.
    if stage is not None:
        model.stage = stage
    model.eval()

    # Initialize losses.
    domain_losses: List[Tuple[int, float, Optional[torch.Tensor]]] = [
        (-1, -1, None) for _ in domains
    ]

    def update_losses(data_idx: int, item_loss: float, pred: torch.Tensor):
        for domain_idx, (lo, hi) in enumerate(domains):
            if lo <= data_idx < hi and item_loss > domain_losses[domain_idx][1]:
                domain_losses[domain_idx] = (data_idx, item_loss, pred)


    # TODO consider using tqdm for larger datasets for visibility
    for j, (x, y) in enumerate(data_loader):
        j_outer = j * data_loader.batch_size
        equations = x[1]
        x = x[0].cuda()
        y = y[0].cuda()
        model.next_channels = tuple([
            MAP_EQUATION_TO_CHANNELS[Equation(eq.item())] for eq in equations
        ])
        out, *_ = model(x)
        # capture all but the first of the returned tuple in underscore.

        # TODO add reporting for total/avg losses in each dataset
        # since I'm running the model this whole time anyways.
        for k, eq in enumerate(equations):
            loss = multi_physics_loss(
                y[k],
                out[k],
                nn.MSELoss(),
                Equation(eq.item()),
            )
            update_losses(j_outer + k, loss.item(), out[k])

        del x, y, out, loss
        gc.collect()
        torch.cuda.empty_cache()

    return domain_losses


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

