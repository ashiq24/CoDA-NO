from models.codano import CodANO
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from torchsummary import summary
from functools import partial
from YParams import YParams
from models.get_models import *
import os
import torch
from train.trainer import simple_trainer
from data_utils.data_loaders import get_onestep_dataloader

## SSL model 
params = YParams('./config/ssl.yaml', 'fno_gino', print_params=True)

if params.nettype == 'transformer':
    encoder, decoder, contrastive, predictor = get_ssl_models_codano_gino(params)
elif params.nettype == 'simple':
    encoder, decoder, contrastive, predictor = get_model_fno_gino(params)

if params.pretrain_ssl:
    model = SslWrapper(params, encoder, decoder, contrastive, predictor, stage='ssl')
else:
    model = SslWrapper(params, encoder, decoder, contrastive, predictor, stage='sl')

model = model.cuda()
train, test = get_onestep_dataloader()
simple_trainer(model.cuda(), train, test, params, stage=model.stage)

if params.pretrain_ssl:
    # if we were pre-training (ssl), then we will train (sl)
    model.stage = 'sl'
    simple_trainer(model.cuda(), train, test, params, stage=model.stage)