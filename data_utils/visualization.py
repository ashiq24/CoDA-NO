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
    logger: Optional[logging.Logger] = None,
):
    """
    Display the difference between ground truth and prediction data.

    Args:
        ground_truth (torch.Tensor): The ground truth data.
        prediction (torch.Tensor): The predicted data.
        channel (int): The channel index to visualize.
        n_cols (int, optional): The number of columns in the visualization grid. Defaults to 5.
        logger (Optional[logging.Logger], optional): The logger object for logging messages. Defaults to None.
    """
    if logger is None:
        logger = logging.getLogger()
    logger.setLevel(logging.WARNING)  # plt is noisy on [DEBUG]

    fig, axs = plt.subplots(
        N_ROWS,
        n_cols + 1,
        figsize=(13, 8),  # (width, height)
        subplot_kw={"xticks": [], "yticks": []},
    )

    x = ground_truth[channel].cpu()
    v_min = x.min()
    v_max = x.max()
    row = 0
    for c in range(n_cols):
        ax = axs[row, c]
        _x = x[c]
        im = ax.imshow(_x, vmin=v_min, vmax=v_max)
    fig.colorbar(im, ax=axs[row, n_cols], location="left")

    y = prediction[channel].cpu().detach().numpy()
    v_min = y.min()
    v_max = y.max()
    row = 1
    for c in range(n_cols):
        ax = axs[row, c]
        _y = y[c]
        im = ax.imshow(_y, vmin=v_min, vmax=v_max)
    fig.colorbar(im, ax=axs[row, n_cols], location="left")

    error2 = np.square(x - y)
    v_min = error2.min()
    v_max = error2.max()
    row = 2
    for c in range(n_cols):
        ax = axs[row, c]
        _error2 = error2[c].cpu().detach().numpy()
        im = ax.imshow(_error2, vmin=v_min, vmax=v_max)
    fig.colorbar(im, ax=axs[row, n_cols], location="left")

    plt.tight_layout()
    plt.show()


def show_multi_physics_data_diffs(
    model: Union[CodANO, CoDANOTemporal],
    data_loader: data.DataLoader,
    swe_index: int = 0,
    diff_index: int = 10,
    ns_index: int = 20,
    stage: Optional[StageEnum] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Displays the differences between the predicted and ground truth data for multiple physics equations.

    Args:
        model (Union[CodANO, CoDANOTemporal]): The trained model used for prediction.
        data_loader (data.DataLoader): The data loader containing the dataset.
        swe_index (int, optional): The index of the data sample for the shallow water equation. Defaults to 0.
        diff_index (int, optional): The index of the data sample for the diffusion-reaction equation. Defaults to 10.
        ns_index (int, optional): The index of the data sample for the Navier-Stokes equation. Defaults to 20.
        stage (Optional[StageEnum], optional): The stage of the model. Defaults to None.
        logger (Optional[logging.Logger], optional): The logger object for logging. Defaults to None.
    """
    model.eval()
    model.stage = stage
    (depth_in, _), (depth_out, _) = data_loader.dataset[swe_index]
    pred, *_ = model(
        depth_in.unsqueeze(0).cuda(),
        equations=torch.Tensor([Equation.SWE.value]).cuda(),
    )

    print("WATER DEPTH (SHALLOW WATER)")
    show_data_diff(
        depth_out, pred.squeeze(0), channel=0, logger=logger,
    )

    (reaction_in, _), (reaction_out, _) = data_loader.dataset[diff_index]
    pred, *_ = model(
        reaction_in.unsqueeze(0).cuda(),
        equations=torch.Tensor([Equation.DIFF.value]).cuda(),
    )

    print("ACTIVATOR (DIFFUSION-REACTION)")
    show_data_diff(
        reaction_out, pred.squeeze(0), channel=0, logger=logger,
    )

    print("INHIBITOR (DIFFUSION-REACTION)")
    show_data_diff(
        reaction_out, pred.squeeze(0), channel=1, logger=logger,
    )

    (ns_in, _), (ns_out, _) = data_loader.dataset[ns_index]
    pred, *_ = model(
        ns_in.unsqueeze(0).cuda(), equations=torch.Tensor([Equation.NS.value]).cuda(),
    )

    print("PARTICLE DENSITY (NAVIER-STOKES)")
    show_data_diff(
        ns_out, pred.squeeze(0), channel=0, logger=logger,
    )

    print("X-VELOCITY (NAVIER-STOKES)")
    show_data_diff(
        ns_out, pred.squeeze(0), channel=1, logger=logger,
    )

    print("Y-VELOCITY (NAVIER-STOKES)")
    show_data_diff(
        ns_out, pred.squeeze(0), channel=2, logger=logger,
    )
