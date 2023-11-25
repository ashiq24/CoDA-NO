# +
import sys

import wandb
# -

from data_utils.hdf5_datasets import *
from layers.attention import TNOBlock3D
from models.codano import CoDANOTemporal
from models.get_models import *
from train.trainer import multi_physics_trainer, test_single_physics
from utils import get_wandb_api_key
from YParams import YParams


# +
## SSL model 
# params = YParams('./config/ssl.yaml', 'codano_gino', print_params=True)
# params = YParams('./config/test.yaml', 'codano_gino', print_params=True)
params = YParams('./config/pdebench.yaml', 'codano_gino', print_params=True)
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
    model = SSLWrapper(
        params,
        encoder,
        decoder,
        contrastive,
        predictor,
        stage=('ssl' if params.pretrain_ssl else 'sl'),
    )

elif params.nettype == 'simple':
    model = get_model_fno(params)


# +
# print(model)

# +
# make it smol
opts = {
    'swe_opts': {
        'subsampling_rate': 2,
        'sample_size': 25,
    },
    'diff_opts': {
        'subsampling_rate': 2,
        'sample_size': 25,
    },
    'ns_opts': {
        # NS dataset is at 4x higher resolution,
        # so it must be subsampled at a 4x lower rate:
        'subsampling_rate': 8,
        'sample_size': 1,
    },
}

# Datasets to be used in reconstructive (i.e. SSL) learning:
train_reconstructive = MultiPhysicsDataset(
    params.filepath_swe,
    params.filepath_diff,
    params.filepaths_ns,
    predictive=False,  # Target output will be the same as the input.
    **opts,
)
test_reconstructive  = MultiPhysicsDataset(
    params.filepath_swe,
    params.filepath_diff,
    params.filepaths_ns,
    offset=10,
    predictive=False,  # Target output will be the same as the input.
    **opts,
)

# Datasets to be used in predictive (i.e. SL) learning:
train_predictive = MultiPhysicsDataset(
    params.filepath_swe,
    params.filepath_diff,
    params.filepaths_ns,
    predictive=True,  # Target output will be the next time trajectory.
    **opts,
)
test_predictive  = MultiPhysicsDataset(
    params.filepath_swe,
    params.filepath_diff,
    params.filepaths_ns,
    offset=10,
    predictive=True,  # Target output will be the next time trajectory.
    **opts,
)
# -

print(
    f"{len(train_reconstructive.swe_dataset)=}",
    f"{len(train_reconstructive.diff_dataset)=}",
    f"{len(train_reconstructive.ns_dataset)=}",
    sep='\n',
)
print(
    f"{len(test_predictive.swe_dataset)=}",
    f"{len(test_predictive.diff_dataset)=}",
    f"{len(test_predictive.ns_dataset)=}",
    sep='\n',
)

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

# Train/test for the reconstructive (i.e. SSL) task:
model = model.cuda()
model.stage = StageEnum.RECONSTRUCTIVE
multi_physics_trainer(
    model,
    train_reconstructive_loader,
    test_reconstructive_loader,
    nn.MSELoss(),  # training loss_fn
    params,
    wandb_log=params.wandb['log'],
    log_interval=params.wandb['log_interval'],
    script=False,
)

# TODO make a small helper to (uniquely) name models
# torch.save(model, 'weights/model_4_ssl.pth')

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
    0,   # start
    m1,  # stop
    script=False,
)

print("Test on the Diffusion-Reaction dataset:")
test_single_physics(
    model,
    test_predictive_loader,
    m1,  # start
    m2,  # stop
    script=False,
)

print("Test on the Navier-Stokes dataset:")
test_single_physics(
    model,
    test_predictive_loader,
    m2,  # start
    n,  # stop
    script=False,
)
# print("Test on the mixed dataset:")
# test_single_physics(0, 4400)
# -

# torch.save(model, 'weights/model_4_sl.pth')

if params.wandb['log']:
    wandb.finish()
