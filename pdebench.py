# Generated with jupytext
# +
import wandb

# +
from data_utils.hdf5_datasets import *

from layers.attention import TNOBlock3D
from layers.fino import SpectralConvolutionKernel3D
from models.codano import CoDANOTemporal
from models.get_models import *
from train.trainer import multi_physics_trainer
from utils import get_wandb_api_key
from YParams import YParams


# +
# # !jupyter-notebook nbextension enable --py widgetsnbextension
# # !jupyter labextension enable widgetsnbextension
# # %load_ext widgetsnbextension
# # !nbextension enable --py widgetsnbextension
# -

## SSL model 
# params = YParams('./config/ssl.yaml', 'codano_gino', print_params=True)
# params = YParams('./config/test.yaml', 'codano_gino', print_params=True)
params = YParams('./config/pdebench.yaml', 'codano_gino', print_params=True)
verbose = True

# Set up WandB logging
if params.wandb.log:
    wandb.login(key=get_wandb_api_key())
    wandb.init(
        config=params,
        name=params.wandb.name,
        group=params.wandb.group,
        project=params.wandb.project,
        entity=params.wandb.entity,
    )

# +
if verbose:
    print(f"{params.nettype=}")
if params.nettype == 'transformer':
    if verbose:
        print(f"{params.grid_type=}")
    if params.grid_type == 'uniform':
        # import pdb; pdb.set_trace()
        encoder, decoder, contrastive, predictor = get_ssl_models_codaNo(
            params,
            CoDANOTemporal,
            TNOBlock3D,
            SpectralConvolutionKernel3D
        )
    else:
        encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)
    
    if verbose:
        print(f"{params.pretrain_ssl=}")
    model = SslWrapper(
        params,
        encoder,
        decoder,
        contrastive,
        predictor,
        stage=('ssl' if params.pretrain_ssl else 'sl')
    )

elif params.nettype == 'simple':
    model = get_model_fno(params)
# -


train = MultiPhysicsDataset(
    params.filepath_swe,
    params.filepath_diff,
    params.filepaths_ns,
    strides_on=3,
    strides_off=1,
    offset=0,
)
test  = MultiPhysicsDataset(
    params.filepath_swe,
    params.filepath_diff,
    params.filepaths_ns,
    strides_on=1,
    strides_off=3,
    offset=30,
)
# -

train_loader = torch.utils.data.DataLoader(
    train,
    batch_size=params.batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    test,
    batch_size=params.batch_size,
    shuffle=False,
)

for i, dataset in enumerate((train, test,)):
    print(
        f"datasets[{i}]:\n"
        f"{len(dataset.swe_dataset)=}\n"
        f"{len(dataset.diff_dataset)=}\n"
        f"{len(dataset.ns_dataset)=}\n"
    )


model = model.cuda()
multi_physics_trainer(
    model,
    train_loader,
    test_loader,
    params,
    wandb_log=params.wandb.log,
    log_interval=params.wandb.log_test_interval,
    stage=('ssl' if params.pretrain_ssl else 'sl')
)

print(model)

# +
# torch.save(model, 'out/pdebench00/model.pt')
# [WARNING] AttributeError: Can't pickle local object 'TNOBlock.__init__.<locals>.<lambda>'
# -

print(f"{params.pretrain_ssl=}")
if params.pretrain_ssl:
    # if we were pre-training (ssl), then we will train (sl)
    model.stage = 'sl'
    multi_physics_trainer(
        model,
        train_loader,
        test_loader,
        params,
        epochs=1,
        wandb_log=params.wandb.log,
        log_interval=params.wandb.log_test_interval,
        stage=model.stage
    )

# +
model.eval()

if params.wandb.log:
    wandb.finish()
