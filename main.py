from YParams import YParams
import os
import wandb
import sys
import torch
from data_utils.data_loaders import *
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from data_utils.data_utils import MaskerNonuniformMesh, batched_masker, MaskerUniform, get_meshes
from models.codano import CodANO
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

from layers.variable_encoding import *
from models.get_models import *
from train.trainer import nonuniform_mesh_trainer
from utils import get_wandb_api_key, TokenExpansion
from models.model_helpers import count_parameters
from test.evaluations import missing_variable_testing
from torchsummary import summary
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from neuralop.training import setup
import neuralop.mpu.comm as comm

import random
def ddp_setup(rank: int, world_size: int):
    """
    Args:
    rank: Unique identifier of each process
    world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
        
#ddp_setup(rank, world_size)
torch.manual_seed(42)
random.seed(42)
config = "codano_test" #sys.argv[1]
#print("Loading config", config)
params = YParams('./config/ssl_ns_elastic.yaml', config, print_params=True)
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./config/ssl_ns_elastic.yaml", config_name = config
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()
config_name = pipe.steps[-1].config_name

params.config = config
# Set up WandB logging
# Set-up distributed communication, if using
device, is_logger = setup(config)
print("Device", device)

params.wandb_name = config
params.wandb_group = params.nettype
if params.wandb_log:
    wandb.login(key=get_wandb_api_key())
    wandb.init(
        config=config,
        name="codano",
        group="neuraloperator",
        project="CoDA-NO",
        entity="robert18")

if params.pretrain_ssl:
    stage = StageEnum.RECONSTRUCTIVE
else:
    stage = StageEnum.PREDICTIVE

if params.nettype == 'transformer':
    if params.grid_type == 'uniform':
        encoder, decoder, contrastive, predictor = get_ssl_models_codaNo(
            params)
    else:
        encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(
            params)

        variable_encoder = get_variable_encoder(params)
        k = variable_encoder(torch.randn(1317, 2), equation=['NS'])
        print(k.shape)
        k = variable_encoder(torch.randn(1317, 2))
        print(k.shape)
        token_expander = TokenExpansion(sum([params.equation_dict[i] for i in params.equation_dict.keys()]), params.n_encoding_channels, params.n_static_channels)

        variable_encoder.to(device)
        token_expander.to(device)

    print("Parameters Encoder", count_parameters(encoder), "x10^6")
    print("Parameters Decoder", count_parameters(decoder), "x10^6")
    print("Parameters Perdictor", count_parameters(predictor), "x10^6")
        
        

    model = SSLWrapper(
        params,
        encoder,
        decoder,
        contrastive,
        predictor,
        stage=stage).to(device)
    if params.grid_type != 'uniform':
        print("Setting the Grid")
        mesh = np.loadtxt(params.input_mesh_location, delimiter=',')
        input_mesh = torch.transpose(torch.stack([torch.tensor(
            mesh[0, :]), torch.tensor(mesh[1, :])]), 0, 1).type(torch.float).cuda()
        model.set_initial_mesh(input_mesh)
    
elif params.nettype in ['simple', 'gnn']:
    model = get_model_fno(params)
    print("Parameters Model", count_parameters(model), "x10^6")
    mesh = None
    variable_encoder = None
    token_expander = None

#device = torch.device("cuda" if torch.cuda.is_available() else 'cpu') ## specify the GPU id's, GPU id's start from 0.
# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    ).to(device)
#model = DDP(model, device_ids = [1,2]).to(device)
#model = model.cuda()
# non-uniform dataset
print(list(params.equation_dict.keys()))
dataset = NsElasticDataset(params.data_location, equation=list(params.equation_dict.keys()))
# train, test = dataset.get_onestep_dataloader(location=params.data_location, dt=params.dt, ntrain=params.get('ntrain'),
#                                              ntest=params.get('ntest'))

train, test = dataset.get_dataloader(params.mu_list, params.dt, ntrain=params.get(
    'ntrain'), ntest=params.get('ntest'), sample_per_inlet=params.sample_per_inlet)

normalizer = dataset.normalizer
normalizer.cuda()

# uniform dataset dummy
# train, test = get_dummy_dataloaders()
if params.training_stage == 'fine_tune':
    print(f"Loading Pretrained weights from {params.pretrain_weight}")
    model.load_state_dict(torch.load(params.pretrain_weight))
    print(f"Loading Pretrained weights from {params.NS_variable_encoder_path}")
    variable_encoder.load_encoder("NS", params.NS_variable_encoder_path)

nonuniform_mesh_trainer(
    model,
    train,
    test,
    params,
    wandb_log=params.wandb_log,
    log_test_interval=params.wandb_log_test_interval,
    normalizer=normalizer,
    stage=stage,
    variable_encoder=variable_encoder,
    token_expander=token_expander,
    initial_mesh=input_mesh)

if params.pretrain_ssl and not params.ssl_only:
    # if we were pre-training (ssl), then we will train (sl)
    model.stage = StageEnum.PREDICTIVE
    nonuniform_mesh_trainer(
        model,
        train,
        test,
        params,
        wandb_log=params.wandb_log,
        log_test_interval=params.wandb_log_test_interval,
        normalizer=normalizer,
        stage=model.stage,
        variable_encoder=variable_encoder,
        token_expander=token_expander,
        initial_mesh=input_mesh)

grid_non, grid_uni = get_meshes(
    params.input_mesh_location, params.grid_size)

test_augmenter = MaskerNonuniformMesh(
    grid_non_uni=grid_non.clone().detach(),
    gird_uni=grid_uni.clone().detach(),
    radius=params.masking_radius,
    drop_type=params.drop_type,
    drop_pix=params.drop_pix_val,
    channel_aug_rate=params.channel_per_val,
    channel_drop_rate=params.channel_drop_per_val,
    verbose=True)

missing_variable_testing(
    model,
    test,
    test_augmenter,
    normalizer,
    'sl',
    params,
    variable_encoder=variable_encoder,
    token_expander=token_expander,
    initial_mesh=input_mesh)

if params.wandb_log:
    wandb.finish()

#destroy_process_group()

#if __name__ == "__main__":
    #device = 0
    #world_size = 2
    #mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)
#main()