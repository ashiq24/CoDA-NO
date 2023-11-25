# +
import logging
import sys

from timeit import default_timer
from tqdm.notebook import tqdm, trange
import wandb

from torch.utils import data
# -

from data_utils.hdf5_datasets import *
from layers.attention import TNOBlock3D
from layers.fino import SpectralConvolutionKernel3D
from models.codano import CoDANOTemporal
from models.get_models import *
from train.trainer import multi_physics_trainer, multi_physics_loss
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
            # logger=logger.getChild("get_SSLWrapper"),
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
        # logger=logger.getChild("SSLWrapper"),
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

train = MultiPhysicsDataset(
    params.filepath_swe,
    params.filepath_diff,
    params.filepaths_ns,
    # strides_on=1,
    # strides_off=1,
    # offset=0,
    **opts,
)
test  = MultiPhysicsDataset(
    params.filepath_swe,
    params.filepath_diff,
    params.filepaths_ns,
    # strides_on=1,
    # strides_off=1,
    **opts,
    offset=10,
)
# -

print(
    # f"{len(test_loader.dataset)=}\n"
    f"{len(train.swe_dataset)=}",
    f"{len(train.diff_dataset)=}",
    f"{len(train.ns_dataset)=}",
    sep='\n',
)
print(
    # f"{len(test_loader.dataset)=}\n"
    f"{len(test.swe_dataset)=}",
    f"{len(test.diff_dataset)=}",
    f"{len(test.ns_dataset)=}",
    sep='\n',
)

train_loader = data.DataLoader(
    train,
    batch_size=params.batch_size,
    shuffle=False,
    # sampler=data.WeightedRandomSampler(
    #     train.get_sampler_weights(),
    #     100,  # num_samples :TODO: parameterize in config
    #     replacement=False,
    # ),
)
test_loader = data.DataLoader(
    test,
    batch_size=params.batch_size,
    shuffle=False,
)

model = model.cuda()
multi_physics_trainer(
    model,
    train_loader,
    test_loader,
    params,
    wandb_log=params.wandb['log'],
    log_interval=params.wandb['log_interval'],
    stage=('ssl' if params.pretrain_ssl else 'sl'),
    notebook=True,
)

# +
# run testing epoch ('cause it was buggy on file)
model.eval()
# TODO confirm MSE is the best loss to train on and/or report.
loss_fn = nn.MSELoss()

t1 = default_timer()
test_l2 = 0.0
n_test = 0

with torch.no_grad():
    for x, y in test_loader:
        eqns = x[1]
        x = x[0].cuda()
        y = y[0].cuda()

        out, _, _, _ = model(x)
        n_test += 1
        target = x.clone()

        for k, eq in enumerate(eqns):
            loss = multi_physics_loss(
                target,
                out,
                loss_fn,
                Equation(eq.item()),
                batch_index=k
            )
            test_l2 += loss.item()

test_l2 /= n_test
t2 = default_timer()
test_time = t2 - t1
print(f"Time: {test_time:.2f}s\n"
      f"Loss: {test_l2:.6f}")
# -

torch.save(model, 'weights/model_4_ssl.pth')

params.pretrain_ssl = False
print(f"{params.pretrain_ssl=}")

print(f"{params.pretrain_ssl=}")
if params.pretrain_ssl:
    # if we were pre-training (ssl), then we will train (sl)
    model.stage = 'sl'
    multi_physics_trainer(
        model,
        train_loader,
        test_loader,
        params,
        # epochs=1, # use default epochs from params
        wandb_log=params.wandb['log'],
        log_interval=params.wandb['log_interval'],
        stage=model.stage
    )

# +
# run testing epoch ('cause it was buggy on file)
model.eval()
loss_fn = nn.MSELoss()

t1 = default_timer()
test_l2 = 0.0
n_test = 0

with torch.no_grad():
    for x, y in test_loader:
        eqns = y[1]
        x = x[0].cuda()
        y = y[0].cuda()

        out, _, _, _ = model(x)
        n_test += 1
        target = y.clone()

        for k, eq in enumerate(eqns):
            loss = multi_physics_loss(
                target,
                out,
                loss_fn,
                Equation(eq.item()),
                batch_index=k
            )
            test_l2 += loss.item()

t2 = default_timer()
test_time = t2 - t1
print(f"Time: {test_time:.2f}s\n"
      f"Loss: {test_l2 / n_test:.6f}")


# +
# model.eval()
# loss_fn = nn.MSELoss()

def test_single_physics(start: int, stop: int):    
    t1 = default_timer()
    test_l2 = 0.0
    ntest = 0
    with torch.no_grad():
        test_loader_trange = trange(
            # test_loader.dataset[start:stop],
            start, 
            stop,
            desc=f'Testing [{start} : {stop}]',
            leave=False,
            ncols=120,
        )

        for i in test_loader_trange:
            x, y = test_loader.dataset[i]
            eq = y[1]
            x = x[0].unsqueeze(0).cuda()
            y = y[0].unsqueeze(0).cuda()
            batch_size = y.shape[0]
    
            out, _, _, _ = model(x)
            ntest += 1
            target = y.clone()
    
            loss = multi_physics_loss(
                target,
                out,
                loss_fn,
                Equation(eq),
                batch_index=0,
            )
            test_l2 += loss.item()
    
    t2 = default_timer()
    test_time = t2 - t1
    print(f"Time: {test_time:.2f}s\n"
          f"Loss: {test_l2/ntest:.6f}")

print(
    f"{len(test_loader.dataset)=}\n"
    f"{len(test.swe_dataset)=}\n"
    f"{len(test.diff_dataset)=}\n"
    f"{len(test.ns_dataset)=}\n"
)

# +
m1 = len(test.swe_dataset)  # "midpoint" 1 - end SWE; begin diffusion-reaction
m2 = m1 + len(test.diff_dataset)  # "midpoint" 1 - end diffusion-reaction; begin Navier-Stokes
n = len(test_loader.dataset)

print("Test on the Shallow Water Equation dataset:")
test_single_physics(0, m1)
print("Test on the Diffusion-Reaction dataset:")
test_single_physics(m1, m2)
print("Test on the Navier-Stokes dataset:")
test_single_physics(m2, n)
# print("Test on the mixed dataset:")
# test_single_physics(0, 4400)
# -

torch.save(model, 'weights/model_4_sl.pth')

if params.wandb['log']:
    wandb.finish()
