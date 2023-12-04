# +
import logging
import pprint
import sys

import matplotlib.pyplot as plt
import wandb

from torch import nn
from torch.utils import data

# +
from data_utils.hdf5_datasets import MultiPhysicsDataset, Equation
from data_utils.visualization import (
    get_multi_physics_data_losses,
    show_data_diff,
    show_multi_physics_data_diffs,
)

from layers.attention import TNOBlock3D
from layers.fino import SpectralConvKernel2d, SpectralConvolutionKernel3D

from models.codano import CoDANOTemporal
from models.get_models import get_ssl_models_codaNo, SSLWrapper, StageEnum
from train.trainer import (
    multi_physics_trainer,
    test_single_physics,
)
from utils import get_wandb_api_key
from YParams import YParams


# +
## SSL model 
params = YParams('./config/pdebench_overfit.yaml', 'codano_gino', print_params=False)
verbose = True

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True)
logger = logging.getLogger()  # get the root logger
# -

# Set up WandB logging
if params.wandb['log']:
    wandb.login(key=get_wandb_api_key())
    wandb.init(
        config=params,
        name=params.wandb['name'],
        group=params.wandb['group'],
        project=params.wandb['project'],
        entity=params.wandb['entity'],
    )

# +
if verbose:
    logger.debug(f"{params.nettype=}")
if params.nettype == 'transformer':
    if verbose:
        logger.info(f"{params.grid_type=}")
    if params.grid_type == 'uniform':
        encoder, decoder, contrastive, predictor = get_ssl_models_codaNo(
            params,
            CoDANOTemporal,
            TNOBlock3D,
            SpectralConvolutionKernel3D,
            logger=logger,
        )
    else:
        encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)
    
    if verbose:
        logger.info(f"{params.pretrain_ssl=}")
    variables = {
        Equation(eq): v for eq, v
        in params.variables_per_equation.items()
    }
    if params.add_static_feature:
        raise NotImplementedError("Static features not currently handled.")
    else:
        n_static_channels = 0
    
    model = SSLWrapper(
        params,
        encoder,
        decoder,
        contrastive,
        predictor,
        variables_per_equations=variables,
        # extra features per variable:
        n_encoding_channels=params.n_encoding_channels,
        n_static_channels=n_static_channels,
        stage=StageEnum.RECONSTRUCTIVE,
        logger=logger.getChild("Adjoint"),
    )

elif params.nettype == 'simple':
    model = get_model_fno(params)

model = model.cuda()


# +
# print(model)

print(sum([p.numel() for p in model.parameters()]))
# -

# In this exercise of overfitting to one sample (from each equation), don't bother with a separate test set. We just want to demonstrate the model has sufficient expressive power to drive the (training) loss to `0.0`.

# +
# Datasets to be used in reconstructive (i.e. SSL) learning:
train_reconstructive = MultiPhysicsDataset(
    swe_args=params.shallow_water,
    diff_args=params.diffusion_reaction,
    ns_args=params.navier_stokes,
    predictive=False,  # Target output will be the same as the input.
    strides_off=0,
)

# Datasets to be used in predictive (i.e. SL) learning:
train_predictive = MultiPhysicsDataset(
    swe_args=params.shallow_water,
    diff_args=params.diffusion_reaction,
    ns_args=params.navier_stokes,
    predictive=True,  # Target output will be the next time trajectory.
    strides_off=0,
)

# +
train_reconstructive_loader = data.DataLoader(
    train_reconstructive,
    batch_size=params.batch_size,
    shuffle=False,
)

train_predictive_loader = data.DataLoader(
    train_predictive,
    batch_size=params.batch_size,
    shuffle=False,
)
# -

# Train/test for the reconstructive (i.e. SSL) task:
model.train()
model.stage = StageEnum.RECONSTRUCTIVE
# import pdb; pdb.set_trace()
multi_physics_trainer(
    model,
    train_reconstructive_loader,
    # test_reconstructive_loader,
    [],  # test_loader
    nn.MSELoss(),  # training loss_fn
    params,
    # epochs=params.epochs // 10,
    # wandb_log=params.wandb['log'],
    wandb_log=False,  # debug
    log_interval=params.wandb['log_interval'],
    # log_interval=1,
    script=False,
)

# +
# assert False
# model.stage = StageEnum.RECONSTRUCTIVE
# # "midpoint" 1 - end SWE; begin diffusion-reaction:
# m1 = len(test_predictive.swe_dataset)
# # "midpoint" 2 - end diffusion-reaction; begin Navier-Stokes:
# m2 = m1 + len(test_predictive.diff_dataset)
# n = len(test_predictive_loader.dataset)

# print("Test on the Shallow Water Equation dataset:")
# test_single_physics(
#     model,
#     test_reconstructive_loader,
#     0,   # start
#     m1,  # stop
#     script=False,
# )
    
# print("Test on the Diffusion-Reaction dataset:")
# test_single_physics(
#     model,
#     test_reconstructive_loader,
#     m1,  # start
#     m2,  # stop
#     script=False,
# )

# print("Test on the Navier-Stokes dataset:")
# test_single_physics(
#     model,
#     test_reconstructive_loader,
#     m2,  # start
#     n,  # stop
#     script=False,
# )
# # print("Test on the mixed dataset:")
# # test_single_physics(0, 4400)

# # What is the loss for each channel?
# -

# *Investigate and visualize performance of encoder/decoder reconstruction.*


# +
# show_multi_physics_data_diffs(
#     model,
#     train_reconstructive_loader,
#     StageEnum.RECONSTRUCTIVE,
# )

# +
# show_multi_physics_data_diffs(
#     model,
#     test_reconstructive_loader,
#     StageEnum.RECONSTRUCTIVE,
# )
# -

# TODO make a small helper to (uniquely) name models
#
# `torch.save(model, 'weights/model_4_ssl.pth')`

params.freeze_encoder = True
print(f"{params.freeze_encoder=}")

model.train()
logger.setLevel(logging.DEBUG)
print(f"{params.pretrain_ssl=}")
if params.pretrain_ssl:
    # Now train/test for the predictive (i.e. SL) task:
    model.stage = StageEnum.PREDICTIVE
    # import pdb; pdb.set_trace()
    multi_physics_trainer(
        model,
        train_predictive_loader,
        # test_predictive_loader,
        None,
        nn.MSELoss(),  # training loss_fn
        params,
        # epochs=200,
        wandb_log=params.wandb['log'],
        log_interval=params.wandb['log_interval'],
    )

# +
# # "midpoint" 1 - end SWE; begin diffusion-reaction:
# m1 = len(train_predictive.swe_dataset)
# # "midpoint" 2 - end diffusion-reaction; begin Navier-Stokes:
# m2 = m1 + len(train_predictive.diff_dataset)
# n = len(train_predictive_loader.dataset)

# print("Test on the Shallow Water Equation dataset:")
# test_single_physics(
#     model,
#     train_predictive_loader,
#     nn.MSELoss(),
#     0,   # start
#     m1,  # stop
#     script=False,
# )

# print("Test on the Diffusion-Reaction dataset:")
# test_single_physics(
#     model,
#     train_predictive_loader,
#     nn.MSELoss(),
#     m1,  # start
#     m2,  # stop
#     script=False,
# )

# print("Test on the Navier-Stokes dataset:")
# test_single_physics(
#     model,
#     train_predictive_loader,
#     nn.MSELoss(),
#     m2,  # start
#     n,  # stop
#     script=False,
# )
# # print("Test on the mixed dataset:")
# # test_single_physics(0, 4400)

# +
# show_multi_physics_data_diffs(
#     model,
#     train_predictive_loader,
#     StageEnum.PREDICTIVE,
# )

# +
# show_multi_physics_data_diffs(
#     model,
#     test_predictive_loader,
#     StageEnum.PREDICTIVE,
# )
# -

# torch.save(model, 'weights/model_4_sl.pth')

if params.wandb['log']:
    wandb.finish()
