# Generated with jupytext
# +
import gc
import os
import enum
import pdb

import wandb
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from timeit import default_timer
from tqdm.notebook import tqdm

# +
from data_utils.data_loaders import *
from data_utils.hdf5_datasets import *

from layers.attention import TnoBlock2d, TNOBlock3D
from layers.fino import SpectralConvKernel2d, SpectralConvolutionKernel3D
from models.codano import CodANO, CoDANOTemporal
from models.get_models import *
from train.trainer import simple_trainer
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
#params = YParams('./config/test.yaml', 'codano_gino', print_params=True)
params = YParams('./config/pdebench.yaml', 'codano_gino', print_params=True)
verbose = True

# Set up WandB logging
if params.wandb_log:
    wandb.login(key=get_wandb_api_key())
    wandb.init(config=params, name=params.wandb_name, group=params.wandb_group,
               project=params.wandb_project, entity=params.wandb_entity)

# +
if params.pretrain_ssl:
    stage = 'ssl'
else:
    stage = 'sl'
    
if verbose:
    print(f"{stage=}")

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
    if params.pretrain_ssl:
        model = SslWrapper(params, encoder, decoder, contrastive, predictor, stage='ssl')
    else:
        model = SslWrapper(params, encoder, decoder, contrastive, predictor, stage='sl')
elif params.nettype == 'simple':
    model = get_model_fno(params)
# -


class Equation(enum.Enum):
    SWE = 0
    DIFF = 1
    NS = 2


# +
CHANNEL_DIM = 0
class MultiPhysicsDataset(torch.utils.data.Dataset):
    """
    First approximation of a multiphysics dataset
    combining the Shallow Water equations, Diffusion Reaction,
    and (incompressible) Navier-Stokes.
    
    "First approximation" because each field is represented
    in its own input and output channel. Schematically:
            +--------+
    ---h--- |        | ------- water height
            |        |
    ---a--- |        | ------- activator
    ---i--- |    GNO | ------- inhibitor
            | CODANO |
    ---p--- |    GNO | ------- particle density
    --v_x-- |        | ------- velocity_x
    --v_x-- |        | ------- velocity_y
            +--------+
            
    Further refinements should allow for constructive mixing
    of all channels.
    """
    def __init__(
        self,
        swe_file,
        diff_file,
        ns_files,
        stride,
        offset
    ):
        self.swe_dataset = SWEDataset(
            swe_file, 
            stride=stride, 
            offset=offset
        )
        self.diff_dataset = DiffusionReaction2DDataset(
            diff_file, 
            stride=stride, 
            offset=offset
        )
        self.ns_dataset = NSIncompressibleDataset(
            ns_files, 
            #subsampling_rate=7, # For a resolution of 512, this maintains (some of) the endpoints of the domain.
            # Make the grid sizes of all inputs all be the same, (i.e. 128x128)
            # although this does break the property of having points lie exactly on the boundary.
            subsampling_rate=4,
            stride=stride, 
            offset=offset
        )

    def __len__(self):
        return len(self.swe_dataset) + len(self.diff_dataset) + len(self.ns_dataset)

    # TODO positional encoding
    # TODO normalization - so all equations take place on the same scale
    # MPP points out the model doesn't learn dynamics well across different scales.
    def __getitem__(self, idx):
        """
        Returns fields A(x), U(x) across 6 variables.
        
        Shape: (T, W, H, C)
        Channels:
            height, (shallow water eqn)
            activator, inhibitor, (diffusion-reaction eqn)
            particle density, Vx, Vy (Navier-Stokes)
            
        "Pad" unused channels with Gaussian noise. This should teach
        the model to ignore these noisy channels without structure.
        """
        if idx < len(self.swe_dataset):
            # old shapes were like (T, W, H, C)
            swe_data = self.swe_dataset[int(idx)]
            # new shapes will be like (C, T, W, H)
            swe_in = torch.permute(swe_data.input, (3, 0, 1, 2))
            swe_out = torch.permute(swe_data.output, (3, 0, 1, 2))

            padding_shape = list(swe_in.shape)
            padding_shape[CHANNEL_DIM] = 5
            ###
            # Pad data with Gaussian noise [i.e. N(0, 1)] 
            first_padding = torch.randn(*padding_shape)
            second_padding = torch.randn(*padding_shape)
            ### 
            return (
                torch.cat([swe_in, first_padding], dim=CHANNEL_DIM),
                (
                    torch.cat([swe_out, second_padding], dim=CHANNEL_DIM),
                    Equation.SWE.value,  # mark this datum as Shallow Water eqn
                ),
            )
        
        idx -= len(self.swe_dataset)
        if idx < len(self.diff_dataset):
            # old shape was like (T, W, H, C)
            diff_data = self.diff_dataset[int(idx)]
            # new shape will be like (C, T, W, H)
            diff_in = torch.permute(diff_data.input, (3, 0, 1, 2))
            diff_out = torch.permute(diff_data.output, (3, 0, 1, 2))
            
            # Try to avoid slicing data if we don't need to
            # by creating correctly sized Gaussian noise tensors
            # to be pre- and post-pended to the pricipal data:
            padding0_shape = list(diff_in.shape)
            padding0_shape[CHANNEL_DIM] = 1
            padding1_shape = list(diff_out.shape)
            padding1_shape[CHANNEL_DIM] = 3
            ###
            # Pad data with Gaussian noise [i.e. N(0, 1)] 
            first_padding0 = torch.randn(*padding0_shape)
            first_padding1 = torch.randn(*padding1_shape)
            second_padding0 = torch.randn(*padding0_shape)
            second_padding1 = torch.randn(*padding1_shape)
            ###
            return (
                torch.cat([first_padding0, diff_in, first_padding1], dim=CHANNEL_DIM),
                (
                    torch.cat([second_padding0, diff_out, second_padding1], dim=CHANNEL_DIM),
                    Equation.DIFF.value,  # mark this datum as Diffusion-Reaction eqn
                ),
            )

        idx -= len(self.diff_dataset)
        if idx < len(self.ns_dataset):
            # old shape was like (T, W, H, C)
            ns_data = self.ns_dataset[int(idx)]
            # new shape will be like (C, T, W, H)
            ns_in = torch.permute(ns_data.input, (3, 0, 1, 2))
            ns_out = torch.permute(ns_data.output, (3, 0, 1, 2))
            
            padding_shape = list(ns_in.shape)
            padding_shape[CHANNEL_DIM] = 3
            ###
            # Pad data with Gaussian noise [i.e. N(0, 1)] 
            first_padding = torch.randn(*padding_shape)
            second_padding = torch.randn(*padding_shape)
            ###
            return (
                torch.cat([first_padding, ns_in], dim=CHANNEL_DIM),
                (
                    torch.cat([second_padding, ns_out], dim=CHANNEL_DIM),
                    Equation.NS.value,  # mark this datum as Navier-Stokes eqn
                ),
            )
        
        idx -= len(self.ns_dataset)
        raise IndexError(f"Cannot access item {idx + len(self)} of {len(self)}")


filepath_swe = "/mnt/mogab/data/pdebench/2D/shallow-water/2D_rdb_NA_NA.h5"
filepath_diff = "/mnt/mogab/data/pdebench/2D/diffusion-reaction/2D_diff-react_NA_NA.h5"
filepaths_ns = [
    "/mnt/mogab/data/pdebench/2D/NS_incom/ns_incom_inhom_2d_512-90.h5",
    "/mnt/mogab/data/pdebench/2D/NS_incom/ns_incom_inhom_2d_512-91.h5",
    "/mnt/mogab/data/pdebench/2D/NS_incom/ns_incom_inhom_2d_512-92.h5",
    "/mnt/mogab/data/pdebench/2D/NS_incom/ns_incom_inhom_2d_512-93.h5",
    "/mnt/mogab/data/pdebench/2D/NS_incom/ns_incom_inhom_2d_512-94.h5",
]

# Interleave 4 datasets for a 75-25% train-test split.
# HACK: what's a better way to do this on the class level?
train0 = MultiPhysicsDataset(filepath_swe, filepath_diff, filepaths_ns, stride=30, offset=0)
train1 = MultiPhysicsDataset(filepath_swe, filepath_diff, filepaths_ns, stride=30, offset=10)
train2 = MultiPhysicsDataset(filepath_swe, filepath_diff, filepaths_ns, stride=30, offset=20)
test   = MultiPhysicsDataset(filepath_swe, filepath_diff, filepaths_ns, stride=30, offset=30)
# -

train_loader0 = torch.utils.data.DataLoader(train0, batch_size=params.batch_size, shuffle=True)
train_loader1 = torch.utils.data.DataLoader(train1, batch_size=params.batch_size, shuffle=True)
train_loader2 = torch.utils.data.DataLoader(train2, batch_size=params.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=params.batch_size, shuffle=False)

# for i, loader in enumerate((train_loader0, train_loader1, train_loader2, test_loader,)):
for i, dataset in enumerate((train0, train1, train2, test,)):
    print(
        f"datasets[{i}]:\n"
        f"{len(dataset.swe_dataset)=}\n"
        f"{len(dataset.diff_dataset)=}\n"
        f"{len(dataset.ns_dataset)=}\n"
    )


def _train_on_loader(train_loader, loss_fn, optimizer, scheduler, stats, params, epochs):
    train_loader_tqdm = tqdm(
        train_loader,
        desc=f'Epoch {stats["epoch"]}/{epochs}',
        leave=False,
        ncols=100
    )
    size = len(train_loader)
    _t1 = default_timer()
    # for x, y in train_loader:
    for x, y in train_loader_tqdm:
        x = x.cuda()
        eqns = y[1]
        y = y[0].cuda()
        batch_size = y.shape[0]
        optimizer.zero_grad()

        # import pdb; pdb.set_trace()
        out = model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        stats["train_count"] += 1

        if stage == 'ssl':
            target = x.clone()
        else:
            target = y.clone()

        # Could this be made more efficient by collecting
        # the same equations in one "mini-batch?"
        for k, eq in enumerate(eqns):
            # import pdb; pdb.set_trace()
            # flat "projection" to attend to only the fields relevant
            # to the equation under study.
            if Equation(eq.item()) == Equation.SWE:
                _target = target[k, :, :, :, 0]
                _out = out[k, :, :, :, 0]

            elif Equation(eq.item()) == Equation.DIFF:
                _target = target[k, :, :, :, 1:3]
                _out = out[k, :, :, :, 1:3]

            elif Equation(eq.item()) == Equation.NS:
                _target = target[k, :, :, :, 3:]
                _out = out[k, :, :, :, 3:]

            else:
                # How did this happen?
                import pdb; pdb.set_trace()
                continue

            loss = loss_fn(
                _target.reshape(1, -1),
                _out.reshape(1, -1),
            )
            loss.backward()
            stats["train_l2"] += loss.item()

        optimizer.step()
        del x, y, out, loss
        gc.collect()

        count = stats["train_count"]
        if (count + 1) % 100 == 0:
            _t2 = default_timer()
            avg_train_l2 = stats['train_l2'] / count
            print(f"Train step {count:04d}: "
                  f"Time: {_t2 - _t1:.2f}s, "
                  f"Loss: {avg_train_l2:.4f}")
            _t1 = default_timer()


def multiphysics_trainer(
    model,
    train_loaders,
    test_loader,
    params,
    epochs=None,
    wandb_log=False,
    log_test_interval=1,
    stage='ssl',
):
    lr = params.lr
    weight_decay = params.weight_decay
    scheduler_step = params.scheduler_step
    scheduler_gamma = params.scheduler_gamma
    if epochs is None:
        epochs = params.epochs
    weight_path = params.weight_path
    optimizer = Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        amsgrad=False
    )
    scheduler = StepLR(
        optimizer,
        step_size=scheduler_step,
        gamma=scheduler_gamma)
    loss_p = nn.MSELoss()
                
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        stats = {
            "epoch": ep,
            "train_l2": 0,
            "train_count": 0,
        }
        for trainer in train_loaders:
            _train_on_loader(trainer, loss_p, optimizer, scheduler, stats, params, epochs)

        torch.cuda.empty_cache()
        scheduler.step()
        t2 = default_timer()
        epoch_train_time = t2 - t1
        avg_train_l2 = stats['train_l2'] / stats["train_count"]

        if ep % log_test_interval == 0:

            values_to_log = dict(train_err=avg_train_l2, time=epoch_train_time)
            print(f"Epoch {ep}: "
                  f"Time: {epoch_train_time:.2f}s, "
                  f"Loss: {avg_train_l2:.4f}")

            if wandb_log:
                wandb.log(values_to_log, step=ep, commit=True)

    # torch.save(model.state_dict(), weight_path)

    model.eval()
    t1 = default_timer()
    test_l2 = 0.0
    ntest = 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.cuda()
            eqns = [_y[0] for _y in y]
            y = torch.as_tensor([_y[1].unsqueeze(0) for _y in y]).cuda()
            batch_size = y.shape[0]

            out, _, _, _ = model(x)
            ntest += 1
            if stage == 'ssl':
                target = x.clone()
            else:
                target = y.clone()

            # TODO refactor
            for k, eq in enumerate(eqns):
                if eq.item() == Equation.SWE:
                    _target = target[k, :, :, :, 0]
                    _out = out[k, :, :, :, 0]
                
                if eq.item() == Equation.DIFF:
                    _target = target[k, :, :, :, 1:3]
                    _out = out[k, :, :, :, 1:3]

                if eq.item() == Equation.NS:
                    _target = target[k, :, :, :, 3:]
                    _out = out[k, :, :, :, 3:]

                _loss = loss_p(
                    _target.reshape(1, -1),
                    _out.reshape(1, -1),
                )
                _loss.backward()
                test_l2 += _loss.item()

    test_l2 /= ntest
    t2 = default_timer()

    if wandb_log:
        wandb.log({'test_error': test_l2}, commit=True)
    print("Test Error : ", test_l2)



model = model.cuda()
multiphysics_trainer(
    model,
    [train_loader0, train_loader1, train_loader2],
    test_loader,
    params,
    wandb_log=params.wandb_log, 
    log_test_interval=params.wandb_log_test_interval,
    stage=stage
)

print(model)

print(torch.save.__doc__)

# +
# torch.save(model, 'out/pdebench00/model.pt')
# [WARNING] AttributeError: Can't pickle local object 'TNOBlock.__init__.<locals>.<lambda>'
# -

print(f"{params.pretrain_ssl=}")
if params.pretrain_ssl:
    # if we were pre-training (ssl), then we will train (sl)
    model.stage = 'sl'
    multiphysics_trainer(
        model,
        [train_loader0],
        test_loader,
        params,
        epochs=1,
        wandb_log=params.wandb_log, 
        log_test_interval=params.wandb_log_test_interval,
        stage=model.stage
    )

# +
model.eval()
loss_p = nn.MSELoss()

t1 = default_timer()
test_l2 = 0.0
ntest = 0
with torch.no_grad():
    for x, y in test_loader:
        x = x.cuda()
        eqns = y[1]
        y = y[0].cuda()
        batch_size = y.shape[0]

        out, _, _, _ = model(x)
        ntest += 1
        if stage == 'ssl':
            target = x.clone()
        else:
            target = y.clone()

        # TODO refactor
        for k, eq in enumerate(eqns):
            if eq.item() == Equation.SWE:
                _target = target[k, :, :, :, 0]
                _out = out[k, :, :, :, 0]
                _loss = loss_p(
                    _target.reshape(1, -1),
                    _out.reshape(1, -1),
                )
                _loss.backward()
                test_l2 += _loss.item()
        
            if eq.item() == Equation.DIFF:
                _target = target[k, :, :, :, 1:3]
                _out = out[k, :, :, :, 1:3]
                _loss = loss_p(
                    _target.reshape(1, -1),
                    _out.reshape(1, -1),
                )
                _loss.backward()
                test_l2 += _loss.item()

            if eq.item() == Equation.NS:
                _target = target[k, :, :, :, 3:]
                _out = out[k, :, :, :, 3:]
                _loss = loss_p(
                    _target.reshape(1, -1),
                    _out.reshape(1, -1),
                )
                _loss.backward()
                test_l2 += _loss.item()

test_l2 /= ntest
t2 = default_timer()
# -

test_time = t2 - t1
print(f"Time: {test_time:.2f}s\n"
      f"Loss: {test_l2:.4f}")

if params.wandb_log:
    wandb.finish()
