from models.codano import CodANO
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from torchsummary import summary
from functools import partial
from YParams import YParams
from models.get_models import get_ssl_models_Gino, SslWrapper
import os
import torch
from train.trainer import simple_trainer
from data_utils.data_loaders import get_onestep_dataloader

## SSL model 
params = YParams('./config/ssl.yaml', 'gnofno', print_params=True)
encoder, decoder, contrastive, predictor = get_ssl_models_Gino(params)

model = SslWrapper(params, encoder, decoder, contrastive, predictor, stage='ssl')

model = model.cuda()
train, test = get_onestep_dataloader()
simple_trainer(model.cuda(), train, test, params)

model.stage = 'sl'
simple_trainer(model.cuda(), train, test, params)