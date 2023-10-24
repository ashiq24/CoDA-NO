from models.codano import CodANO
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from torchsummary import summary
from functools import partial
from YParams import YParams
from models.get_models import get_ssl_models_Gino, SslWrapper
import os
import torch


# example model

token_codim = 1
out_token_coken = 1
hidden_token_codim = 4
lifting_token_codim = 4
n_heads = [2,2,2,2]
scaling = [[1,1],[1,1],[1,1],[1,1]]
modes = [[100,100],[100,100],[100,100],[100,100]]
lifting = True
projection = True
operator_block = TnoBlock2d
int_op = partial(SpectralConvKernel2d, frequency_mixer = False, fft_type='fft')
int_op_top = int_op
int_op_top = int_op

var_encoding=True #b
var_num=10 # denotes the number of varibales
var_enco_basis='fft'
var_enco_channels=1
enable_cls_token=True

model = CodANO(in_token_codim=token_codim, hidden_token_codim=hidden_token_codim, lifting_token_codim=lifting_token_codim,\
                n_layers=4, n_heads=n_heads, n_modes=modes, scalings=scaling, integral_operator=int_op,\
                integral_operator_top=int_op_top,integral_operator_bottom=int_op_top,\
                var_encoding=var_encoding, var_enco_channels = var_enco_channels,\
                var_num = var_num, enable_cls_token = enable_cls_token)

summary(model, (var_num*token_codim, 100, 100)) 

y = model(torch.randn(2,var_num*token_codim, 100, 100).cuda())

print("Output shape",y.shape)


## SSL model 
params = YParams('./config/ssl.yaml', 'base_config', print_params=True)
encoder, decoder, contrastive, predictor = get_ssl_models(params)

summary(model, (params.var_num*params.in_token_codim_en, 100, 100)) 

ssl_model = SslWrapper(params, encoder, decoder, contrastive, predictor, stage='ssl')

ssl_model = ssl_model.cuda()
y,_,_,_ = ssl_model(torch.randn(2,params.var_num*params.in_token_codim_en, 100, 100).cuda())

print(y.shape)