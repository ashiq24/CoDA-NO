from YParams import YParams
import os
import wandb

from data_utils.data_loaders import *
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from models.codano import CodANO
from models.get_models import *
from train.trainer import simple_trainer
from utils import get_wandb_api_key


## SSL model 
params = YParams('./config/ssl.yaml', 'codano_gino', print_params=True)
#params = YParams('./config/test.yaml', 'codano_gino', print_params=True)

#params = YParams('./config/ssl.yaml', 'base_config', print_params=True)

# Set up WandB logging
if params.wandb_log:
    wandb.login(key=get_wandb_api_key())
    wandb.init(config=params, name=params.wandb_name, group=params.wandb_group,
               project=params.wandb_project, entity=params.wandb_entity)

if params.pretrain_ssl:
    stage = 'ssl'
else:
    stage = 'sl'
    
if params.nettype == 'transformer':
    if params.grid_type == 'uniform':
        encoder, decoder, contrastive, predictor = get_ssl_models_codaNo(params)
    else:
        encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)
    
    if params.pretrain_ssl:
        model = SslWrapper(params, encoder, decoder, contrastive, predictor, stage='ssl')
    else:
        model = SslWrapper(params, encoder, decoder, contrastive, predictor, stage='sl')
elif params.nettype == 'simple':
    model = get_model_fno(params)


model = model.cuda()
# non-uniform dataset
train, test = get_onestep_dataloader(ntrain=params.get('ntrain'),
                                     ntest=params.get('ntest'))

# uniform dataset dummy
#train, test = get_dummy_dataloaders()
simple_trainer(model, train, test, params, wandb_log=params.wandb_log, 
               log_test_interval=params.wandb_log_test_interval, stage=stage)

if params.pretrain_ssl:
    # if we were pre-training (ssl), then we will train (sl)
    model.stage = 'sl'
    simple_trainer(model, train, test, params, wandb_log=params.wandb_log,
                   log_test_interval=params.wandb_log_test_interval, stage=model.stage)

if params.wandb_log:
    wandb.finish()