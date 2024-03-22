import gc
import logging
import pathlib
import sys
import wandb
import torch
from torch import nn
from torch.utils import data
import data_utils
from layers.attention import TNOBlock3D
from layers.fino import SpectralConvolutionKernel3D
from models.codano import CoDANOTemporal
from neuraloperator.models import FNO
from models.get_models import get_ssl_models_codaNo, SSLWrapper, StageEnum
from train import multi_physics_trainer, test_single_physics
from utils import get_wandb_api_key, save_model
import YParams
from data_utils.visualization import (
        show_data_diff,
        show_multi_physics_data_diffs,
    )

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, force=True)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# SSL model
params = YParams.YParams("./config/pdebench.yaml", "codano", print_params=False,)
verbose = True

# Set up WandB logging
if params.wandb["log"]:
    wandb.login(key=get_wandb_api_key())
    wandb.init(
        config=params,
        name=params.wandb["name"],
        group=params.wandb["group"],
        project=params.wandb["project"],
        entity=params.wandb["entity"],
    )

# +
if verbose:
    logger.debug(f"{params.nettype=}")
if params.nettype == "transformer":
    if verbose:
        logger.info(f"{params.grid_type=}")
    if params.grid_type == "uniform":
        encoder, decoder, contrastive, predictor = get_ssl_models_codaNo(
            params,
            CoDANOTemporal,
            TNOBlock3D,
            SpectralConvolutionKernel3D,
        )

    if verbose:
        logger.info(f"{params.pretrain_ssl=}")
    model = SSLWrapper(
        params,
        encoder,
        decoder,
        contrastive,
        predictor,
        stage=("ssl" if params.pretrain_ssl else "sl"),
    )
    
model = model.cuda()

Equation = data_utils.Equation
MultiPhysicsDataset = data_utils.MultiPhysicsDataset
NSIncompressibleDataset = data_utils.NSIncompressibleDataset

# Datasets to be used in reconstructive (i.e. SSL) learning:
train_reconstructive = MultiPhysicsDataset(
    swe_args=params.shallow_water,
    diff_args=params.diffusion_reaction,
    ns_args=params.navier_stokes,
    predictive=False,  # Target output will be the same as the input.
)
test_reconstructive = MultiPhysicsDataset(
    swe_args=params.shallow_water,
    diff_args=params.diffusion_reaction,
    ns_args=params.navier_stokes,
    offset=10,
    predictive=False,  # Target output will be the same as the input.
)
# Datasets to be used in predictive (i.e. SL) learning:
train_predictive = MultiPhysicsDataset(
    swe_args=params.shallow_water,
    diff_args=params.diffusion_reaction,
    ns_args=params.navier_stokes,
    predictive=True,  # Target output will be the next time trajectory.
)
test_predictive = MultiPhysicsDataset(
    swe_args=params.shallow_water,
    diff_args=params.diffusion_reaction,
    ns_args=params.navier_stokes,
    offset=10,
    predictive=True,  # Target output will be the next time trajectory.
)

# +
train_reconstructive_loader = data.DataLoader(
    train_reconstructive, batch_size=params.batch_size, shuffle=False,
)
test_reconstructive_loader = data.DataLoader(
    test_reconstructive, batch_size=params.batch_size, shuffle=False,
)

train_predictive_loader = data.DataLoader(
    train_predictive, batch_size=params.batch_size, shuffle=False,
)
test_predictive_loader = data.DataLoader(
    test_predictive, batch_size=params.batch_size, shuffle=False,
)
# -

# Train/test for the reconstructive (i.e. SSL) task:
model.train()
model.stage = StageEnum.RECONSTRUCTIVE
# import pdb; pdb.set_trace()
multi_physics_trainer(
    model,
    train_reconstructive_loader,
    test_reconstructive_loader,
    nn.MSELoss(),
    params,
    epochs=params.epochs,
    wandb_log=params.wandb["log"],
    log_interval=params.wandb["log_interval"],
    script=True,
)

gc.collect()
torch.cuda.empty_cache()

show_multi_physics_data_diffs(
    model,
    train_reconstructive_loader,
    swe_index=0,
    diff_index=50,
    ns_index=100,
    stage=StageEnum.RECONSTRUCTIVE,
    logger=logger,
)

show_multi_physics_data_diffs(
    model,
    test_reconstructive_loader,
    swe_index=0,
    diff_index=50,
    ns_index=100,
    stage=StageEnum.RECONSTRUCTIVE,
    logger=logger,
)

save_model(
    model,
    directory=pathlib.Path("/weight"),
    stage=StageEnum.RECONSTRUCTIVE,
)

model.train()
logger.setLevel(logging.DEBUG)
params["gradient"]["threshold"] = 0.1

print(f"{params.pretrain_ssl=}")
if params.pretrain_ssl:
    # Now train/test for the predictive (i.e. SL) task:
    model.stage = StageEnum.PREDICTIVE
    multi_physics_trainer(
        model,
        train_predictive_loader,
        test_predictive_loader,
        nn.MSELoss(),  # training loss_fn
        params,
        epochs=10,
        wandb_log=params.wandb["log"],
        log_interval=params.wandb["log_interval"],
    )
# -

test_single_physics(
    model, test_predictive_loader, nn.MSELoss(), start=200, stop=2_000, script=False,
)

show_multi_physics_data_diffs(
    model,
    train_predictive_loader,
    swe_index=0,
    diff_index=50,
    ns_index=100,
    stage=StageEnum.PREDICTIVE,
    logger=logger,
)

show_multi_physics_data_diffs(
    model,
    test_predictive_loader,
    swe_index=0,
    diff_index=50,
    ns_index=100,
    stage=StageEnum.PREDICTIVE,
    logger=logger,
)

save_model(
    model, directory=pathlib.Path("/weight"), stage=StageEnum.PREDICTIVE,
)

if params.wandb["log"]:
    wandb.finish()
