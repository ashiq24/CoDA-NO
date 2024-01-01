# +
import sys

import wandb
# -

from data_utils.hdf5_datasets import *
from data_utils.visualization import show_multi_physics_data_diffs
from layers.attention import TNOBlock3D
from models.codano import CoDANOTemporal
from models.get_models import *
from train.trainer import (
    multi_physics_trainer,
    test_single_physics,
)
from utils import get_wandb_api_key
from YParams import YParams


# +
# SSL model
# params = YParams('./config/ssl.yaml', 'codano_gino', print_params=True)
# params = YParams('./config/test.yaml', 'codano_gino', print_params=True)
params = YParams('./config/pdebench_overfit.yaml',
                 'codano_gino', print_params=False)
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
        encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(
            params)

    if verbose:
        logger.info(f"{params.pretrain_ssl=}")
    model = SSLWrapper(
        params,
        encoder,
        decoder,
        contrastive,
        predictor,
        # variables_per_equation,
        stage=('ssl' if params.pretrain_ssl else 'sl'),
    )

elif params.nettype == 'simple':
    model = get_model_fno(params)

model = model.cuda()


# +
# print(model)

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
    train_reconstructive,
    batch_size=params.batch_size,
    shuffle=False,
    # sampler=data.WeightedRandomSampler(
    #     train.get_sampler_weights(),
    #     100,  # num_samples :TODO: parameterize in config
    #     replacement=False,
    # ),
)
test_reconstructive_loader = data.DataLoader(
    test_reconstructive,
    batch_size=params.batch_size,
    shuffle=False,
)

train_predictive_loader = data.DataLoader(
    train_predictive,
    batch_size=params.batch_size,
    shuffle=False,
    # sampler=data.WeightedRandomSampler(
    #     train.get_sampler_weights(),
    #     100,  # num_samples :TODO: parameterize in config
    #     replacement=False,
    # ),
)
test_predictive_loader = data.DataLoader(
    test_predictive,
    batch_size=params.batch_size,
    shuffle=False,
)

# +
# import pdb; pdb.set_trace()
# x, y = train_reconstructive_loader.dataset[150]
# -

# Train/test for the reconstructive (i.e. SSL) task:
model.train()
model.stage = StageEnum.RECONSTRUCTIVE
# import pdb; pdb.set_trace()
multi_physics_trainer(
    model,
    train_reconstructive_loader,
    test_reconstructive_loader,
    nn.MSELoss(),  # training loss_fn
    params,
    # epochs=params.epochs // 10,
    wandb_log=params.wandb['log'],
    # wandb_log=False,  # debug
    log_interval=params.wandb['log_interval'],
    # log_interval=1,
    script=True,
)

# +
model.stage = StageEnum.RECONSTRUCTIVE
# "midpoint" 1 - end SWE; begin diffusion-reaction:
m1 = len(test_predictive.swe_dataset)
# "midpoint" 2 - end diffusion-reaction; begin Navier-Stokes:
m2 = m1 + len(test_predictive.diff_dataset)
n = len(test_predictive_loader.dataset)
# Can't use `len(test_predictive_loader)`
# because this returns the length in units of batches.

print("Test on the Shallow Water Equation dataset:")
test_single_physics(
    model,
    test_reconstructive_loader,
    nn.MSELoss(),
    start=0,
    stop=m1,
    script=False,
)

print("Test on the Diffusion-Reaction dataset:")
test_single_physics(
    model,
    test_reconstructive_loader,
    nn.MSELoss(),
    start=m1,
    stop=m2,
    script=False,
)

print("Test on the Navier-Stokes dataset:")
test_single_physics(
    model,
    test_reconstructive_loader,
    nn.MSELoss(),
    start=m2,
    stop=n,
    script=False,
)
# print("Test on the mixed dataset:")
# test_single_physics(0, 4400)

# What is the loss for each channel?
# -

# *Investigate and visualize performance of encoder/decoder reconstruction.*


show_multi_physics_data_diffs(
    model,
    train_reconstructive_loader,
    StageEnum.RECONSTRUCTIVE,
    logger=logger,
)

show_multi_physics_data_diffs(
    model,
    test_reconstructive_loader,
    StageEnum.RECONSTRUCTIVE,
    logger=logger,
)

# TODO make a small helper to (uniquely) name models
#
# `torch.save(model, 'weights/model_4_ssl.pth')`

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
        test_predictive_loader,
        nn.MSELoss(),  # training loss_fn
        params,
        # epochs=1, # use default epochs from params
        wandb_log=params.wandb['log'],
        log_interval=params.wandb['log_interval'],
    )

# +
# "midpoint" 1 - end SWE; begin diffusion-reaction:
m1 = len(test_predictive.swe_dataset)
# "midpoint" 2 - end diffusion-reaction; begin Navier-Stokes:
m2 = m1 + len(test_predictive.diff_dataset)
n = len(test_predictive_loader.dataset)

print("Test on the Shallow Water Equation dataset:")
test_single_physics(
    model,
    test_predictive_loader,
    nn.MSELoss(),
    start=0,
    stop=m1,
    script=False,
)

print("Test on the Diffusion-Reaction dataset:")
test_single_physics(
    model,
    test_predictive_loader,
    nn.MSELoss(),
    start=m1,
    stop=m2,
    script=False,
)

print("Test on the Navier-Stokes dataset:")
test_single_physics(
    model,
    test_predictive_loader,
    nn.MSELoss(),
    start=m2,
    stop=n,
    script=False,
)
# print("Test on the mixed dataset:")
# test_single_physics(0, 4400)
# -

show_multi_physics_data_diffs(
    model,
    train_predictive_loader,
    StageEnum.PREDICTIVE,
    logger=logger,
)

# +
# show_multi_physics_data_diffs(
#     model,
#     test_predictive_loader,
#     StageEnum.PREDICTIVE,
#     logger=logger,
# )
# -

# torch.save(model, 'weights/model_4_sl.pth')

if params.wandb['log']:
    wandb.finish()
