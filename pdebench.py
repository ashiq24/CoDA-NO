# +
import pprint
import sys

import matplotlib.pyplot as plt
import wandb
# -

from data_utils.hdf5_datasets import *
from layers.attention import TNOBlock3D
from models.codano import CoDANOTemporal
from models.get_models import *
from train.trainer import (
    multi_physics_trainer,
    multi_physics_loss,
    test_single_physics,
)
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

model = model.cuda()
# -


print(model)

# +
# make it smol
opts = {
    # 5 * 25 a/u data pairs, 64x64 resolution
    'swe_opts': {
        'subsampling_rate': 2,
        'sample_size': 25,
    },
    # 5 * 25 a/u data pairs, 64x64 resolution
    'diff_opts': {
        'subsampling_rate': 2,
        'sample_size': 25,
    },
    # 50 * 1 a/u data pairs, 64x64 resolution
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
# -

# Train/test for the reconstructive (i.e. SSL) task:
model.stage = StageEnum.RECONSTRUCTIVE
# import pdb; pdb.set_trace()
multi_physics_trainer(
    model,
    train_reconstructive_loader,
    test_reconstructive_loader,
    nn.MSELoss(),  # training loss_fn
    params,
    wandb_log=params.wandb['log'],
    # wandb_log=False,  # debug
    log_interval=params.wandb['log_interval'],
    script=False,
)

# +
model.stage = StageEnum.RECONSTRUCTIVE
# "midpoint" 1 - end SWE; begin diffusion-reaction:
m1 = len(test_predictive.swe_dataset)
# "midpoint" 2 - end diffusion-reaction; begin Navier-Stokes:
m2 = m1 + len(test_predictive.diff_dataset)
n = len(test_predictive_loader.dataset)

print("Test on the Shallow Water Equation dataset:")
test_single_physics(
    model,
    test_reconstructive_loader,
    0,   # start
    m1,  # stop
    script=False,
)
    
print("Test on the Diffusion-Reaction dataset:")
test_single_physics(
    model,
    test_reconstructive_loader,
    m1,  # start
    m2,  # stop
    script=False,
)

print("Test on the Navier-Stokes dataset:")
test_single_physics(
    model,
    test_reconstructive_loader,
    m2,  # start
    n,  # stop
    script=False,
)
# print("Test on the mixed dataset:")
# test_single_physics(0, 4400)

# What is the loss for each channel?
# -

# *Investigate and visualize performance of encoder/decoder reconstruction.*

# +
import gc

gc.collect()
torch.cuda.empty_cache()


# +
# When the model performs badly, HOW is it performing badly?
def get_multi_physics_data_losses(model, data_loader, domains):
    """Assumes `model` has been set to the apt stage by the caller."""
    logger.setLevel(logging.DEBUG)
    model.eval()
    ys = [None for _ in range(len(data_loader.dataset) // data_loader.batch_size)]
    losses = [0.0 for _ in range(len(data_loader.dataset))]

    # TODO consider using tqdm for larger datasets for visibility
    for j, (x, _) in enumerate(data_loader):
        equations = x[1]
        x = x[0].cuda()
        out, *_ = model(x.clone())  # capture all but the first of the returned tuple in under

        # We want to "remember" the worst-performing prediction
        loss_min = np.inf
        k_min = -1
        # losses = torch.tensor(0.0, dtype=torch.float).cuda()
        for k, eq in enumerate(equations):
            loss = multi_physics_loss(
                x.clone(),
                out.clone(),
                nn.MSELoss(),
                Equation(eq.item()),
                batch_index=k,
            )
            if loss < loss_min:
                loss_min = loss
                k_min = k
            # losses += loss
            idx = j * data_loader.batch_size + k
            losses[idx] = loss.item()
        
        ys[j] = out[k_min].clone()
        del x, out
        gc.collect()
        torch.cuda.empty_cache()        

    # What are the worst learned data points?
    # batch_size = data_loader.batch_size
    # swe_losses = {
    #     k: v for k, v in losses.items() 
    #     if k[1] + batch_size * k[0] < 125
    # }
    # ns_losses = {
    #     k: v for k, v in ns_losses.items()
    #     if v > list(sorted(ns_losses.values(), reverse=True))[cutoff]
    # }

    # What are the worst learned data points for each domain?
    domain_losses = []
    for lo, hi in domains:
        idx, _loss = max(enumerate(losses[lo:hi]), key=lambda ix: ix[1])
        domain_losses.append((idx, ys[idx // data_loader.batch_size]))

    return domain_losses

# pprint.pprint(swe_losses)
# pprint.pprint(diff_losses)
# pprint.pprint(ns_losses)
swe2_losses, diff2_losses, ns2_losses = get_multi_physics_data_losses(
    model,
    train_reconstructive_loader,
    ((0, 125), (125, 250), (250, 300)),
)

# +
N_ROWS = 3  # ground_truth, prediction, and error
def show_data_diff(
    data_loader,
    model_stage,  # XXX
    channel,
    index=0, 
    # n_rows=4,
    n_cols=5,  # used as time axis
):
    model.stage = model_stage
    logger.setLevel(logging.WARNING)  # plt is noisy on [DEBUG]
    fig, axs = plt.subplots(
        N_ROWS,
        n_cols,
        figsize=(13, 8),  # (width, height)
        subplot_kw={'xticks': [], 'yticks': []},
    )
    (x, _), _ = data_loader.dataset[index]
    x = x.unsqueeze(0).cuda()  # batch index
    out, *_ = model(x.clone())  # XXX

    x = x[0, channel].cpu()
    vmin = x.min()
    vmax = x.max()
    row = 0
    for c in range(n_cols):
        ax = axs[row, c]
        _x = x[c]
        im = ax.imshow(_x, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, location='right')
    
    out = out[0, channel].cpu().detach().numpy()
    vmin = out.min()
    vmax = out.max()
    row = 1
    for c in range(n_cols):
        ax = axs[row, c]
        _out = out[c]
        im = ax.imshow(_out, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, location='right')
        
    error2 = np.square(x - out)
    vmin = error2.min()
    vmax = error2.max()
    row = 2
    for c in range(n_cols):
        ax = axs[row, c]
        _error2 = error2[c].cpu().detach().numpy()
        im = ax.imshow(_error2, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, location='right')

    plt.tight_layout()
    plt.show()


# swe_indices = [k[1] + 10*k[0] for k in swe_losses.keys()]

print("WATER DEPTH (SHALLOW WATER)")
show_data_diff(
    train_reconstructive_loader,
    model_stage=StageEnum.RECONSTRUCTIVE,
    channel=0,
    index=swe_indices[0],
    # n_rows=2,
    # n_cols=2,
)

# +
N_ROWS = 3  # ground_truth, prediction, and error
def show_data_diff(
    ground_truth,
    prediction,
    channel,
    n_cols=5,  # used as time axis
):
    logger.setLevel(logging.WARNING)  # plt is noisy on [DEBUG]
    fig, axs = plt.subplots(
        N_ROWS,
        n_cols + 1,
        figsize=(13, 8),  # (width, height)
        subplot_kw={'xticks': [], 'yticks': []},
    )

    x = ground_truth[channel].cpu()
    vmin = x.min()
    vmax = x.max()
    row = 0
    for c in range(n_cols):
        ax = axs[row, c]
        _x = x[c]
        im = ax.imshow(_x, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[row, n_cols], location='left')
    
    y = prediction[channel].cpu().detach().numpy()
    vmin = y.min()
    vmax = y.max()
    row = 1
    for c in range(n_cols):
        ax = axs[row, c]
        _y = y[c]
        im = ax.imshow(_y, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[row, n_cols], location='left')
        
    error2 = np.square(x - y)
    vmin = error2.min()
    vmax = error2.max()
    row = 2
    for c in range(n_cols):
        ax = axs[row, c]
        _error2 = error2[c].cpu().detach().numpy()
        im = ax.imshow(_error2, vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=axs[row, n_cols], location='left')

    plt.tight_layout()
    plt.show()


# swe_indices = [k[1] + 10*k[0] for k in swe_losses.keys()]

print("WATER DEPTH (SHALLOW WATER)")
show_data_diff(
    train_reconstructive_loader,
    model_stage=StageEnum.RECONSTRUCTIVE,
    channel=0,
    index=swe_indices[0],
    # n_rows=2,
    # n_cols=2,
)


# +
# logger.setLevel(logging.WARNING)
# diff_indices = [k[1] + 10*k[0] for k in diff_losses.keys()]

# print("ACTIVATOR (DIFFUSION-REACTION)")
# show_data_diffs(
#     diff_indices,
#     channel=1,
#     n_rows=4,
#     n_cols=2,  # Number of columns as measured in A(x), U(x) pairs
# )

# print("INHIBITOR (DIFFUSION-REACTION)")
# show_data_diffs(
#     diff_indices,
#     channel=2,
#     n_rows=4,
#     n_cols=2,  # Number of columns as measured in A(x), U(x) pairs
# )

# +
# logger.setLevel(logging.WARNING)
# ns_indices = [k[1] + 10*k[0] for k in ns_losses.keys()]

# print("PARTICLE DENSITY (NAVIER-STOKES)")
# show_data_diffs(
#     ns_indices,
#     channel=3,
#     n_rows=4,
#     n_cols=2,
# )

# print("X-VELOCITY (NAVIER-STOKES)")
# show_data_diffs(
#     ns_indices,
#     channel=4,
#     n_rows=4,
#     n_cols=2,
# )

# print("Y-VELOCITY (NAVIER-STOKES)")
# show_data_diffs(
#     ns_indices,
#     channel=5,
#     n_rows=4,
#     n_cols=2,
# )

# +
def show_multi_physics_data_diffs(
    data_loader,
    model_stage,
    n_rows=2,
    n_cols=2,
):
    swe_losses, diff_losses, ns_losses = get_data_losses(model, data_loader, cutoff=4)
    show_data_args = dict(
        data_loader=data_loader,
        model_stage=model_stage,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    
    logger.setLevel(logging.WARNING)  # plt is noisy
    swe_indices = [k[1] + data_loader.batch_size*k[0] for k in swe_losses.keys()]
    
    print("WATER DEPTH (SHALLOW WATER)")
    show_data_diffs(
        indices=swe_indices,
        channel=0,
        **show_data_args,
    )
    
    diff_indices = [k[1] + data_loader.batch_size*k[0] for k in diff_losses.keys()]
    print("ACTIVATOR (DIFFUSION-REACTION)")
    show_data_diffs(
        indices=diff_indices,
        channel=1,
        **show_data_args,
    )
    
    print("INHIBITOR (DIFFUSION-REACTION)")
    show_data_diffs(
        indices=diff_indices,
        channel=2,
        **show_data_args,
    )
    
    ns_indices = [k[1] + data_loader.batch_size*k[0] for k in ns_losses.keys()]    
    print("PARTICLE DENSITY (NAVIER-STOKES)")
    show_data_diffs(
        indices=ns_indices,
        channel=3,
        **show_data_args,
    )
    
    print("X-VELOCITY (NAVIER-STOKES)")
    show_data_diffs(
        indices=ns_indices,
        channel=4,
        **show_data_args,
    )
    
    print("Y-VELOCITY (NAVIER-STOKES)")
    show_data_diffs(
        indices=ns_indices,
        channel=5,
        **show_data_args,
    )

show_multi_physics_data_diffs(train_reconstructive_loader, StageEnum.RECONSTRUCTIVE)
# -

show_multi_physics_data_diffs(test_reconstructive_loader, StageEnum.RECONSTRUCTIVE)

# TODO make a small helper to (uniquely) name models
#
# `torch.save(model, 'weights/model_4_ssl.pth')`

logger.setLevel(logging.DEBUG)
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

# +
show_multi_physics_data_diffs(train_predictive_loader, StageEnum.PREDICTIVE)

show_multi_physics_data_diffs(test_predictive_loader, StageEnum.PREDICTIVE)
# -

# torch.save(model, 'weights/model_4_sl.pth')

if params.wandb['log']:
    wandb.finish()
