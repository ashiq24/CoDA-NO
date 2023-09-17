from models.codano import CodANO
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from torchsummary import summary
from functools import partial


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

var_encoding=True, #b
var_num=10, # denotes the number of varibales
var_enco_basis='fft',
var_enco_channels=1,
enable_cls_token=True,

model = CodANO(in_token_codim=token_codim, hidden_token_codim=hidden_token_codim, lifting_token_codim=lifting_token_codim,\
                n_layers=4, n_heads=n_heads, n_modes=modes, scalings=scaling, integral_operator=int_op,\
                integral_operator_top=int_op_top,integral_operator_bottom=int_op_top,\
                var_encoding=var_encoding,
                var_enco_channels = var_enco_channels,
                var_num = var_num,
                enable_cls_token = enable_cls_token)

summary(model, (var_num*token_codim, 100, 100)) 


