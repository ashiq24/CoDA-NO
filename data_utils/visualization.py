import gc
import logging
from typing import Optional, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils import data

from data_utils.hdf5_datasets import Equation
from models.codano import CodANO, CoDANOTemporal
from models.get_models import StageEnum
from train.trainer import multi_physics_loss


MAP_EQUATION_TO_CHANNELS = {
    Equation.SWE: (0,),
    Equation.DIFF: (1, 2),
    Equation.NS: (3, 4, 5,),
}


# When the model performs badly, HOW is it performing badly?
# because of masking, I need to remember the model output, which usually still
# has edges in it from being masked.
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
                domain_losses[domain_idx] = (data_idx, item_loss, pred.clone())


    # TODO consider using tqdm for larger datasets for visibility
    for j, (x, y) in enumerate(data_loader):
        j_outer = j * data_loader.batch_size
        equations = x[1]
        x = x[0].cuda()
        y = y[0].cuda()
        model.next_channels = tuple([
            MAP_EQUATION_TO_CHANNELS[Equation(eq.item())] for eq in equations
        ])
        out, *_ = model(x.clone())
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

