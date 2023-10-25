from models.codano import CodANO
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from torchsummary import summary
from functools import partial
from YParams import YParams
from models.get_models import get_ssl_models_Gino, SslWrapper
import os
import torch


## SSL model 
params = YParams('./config/ssl.yaml', 'base_config', print_params=True)
encoder, decoder, contrastive, predictor = get_ssl_models_Gino(params)

ssl_model = SslWrapper(params, encoder, decoder, contrastive, predictor, stage='ssl')

ssl_model = ssl_model.cuda()
y,_,_,_ = ssl_model(torch.randn(2,params.var_num*params.in_token_codim_en, 100, 100).cuda())

print(y.shape)