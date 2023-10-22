#from neuralop.models.tfno import Projection
import torch.nn as nn
import torch.nn.functional as F
from functools import partialmethod
import torch
from einops import rearrange
from functools import partial
from neuralop.layers.mlp import MLP
from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.padding import DomainPadding
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from layers.regrider import Regird
import numpy as np
from layers.variable_encoding import VaribaleEncoding2d


class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None,\
                 n_dim=2, non_linearity=F.gelu, permutation_invariant = False):
        '''
        Permutation invariant projection layer.
        Performs linear projections on each channel separately.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = in_channels if hidden_channels is None else hidden_channels 
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')
            
        self.permutation_invariant = permutation_invariant

        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.norm = nn.InstanceNorm2d(hidden_channels, affine=True)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        batch = x.shape[0]
        if self.permutation_invariant:
            assert x.shape[1]%self.in_channels == 0; "Total Number of Channels is not divisible by number of tokens"
            x  = rearrange(x, 'b (g c) h w -> (b g) c h w', c = self.in_channels)
        x = self.fc1(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        if self.permutation_invariant:
            x  = rearrange(x, '(b g) c h w -> b (g c) h w', b = batch)
        return x

class CodANO(nn.Module):
    """
    in_token_codim, input token codim/number of channel per input token
    out_token_codim=None, output token codim/number of channel per output token
    hidden_token_codim=None, 
    lifting_token_codim=None,
    var_encoding=False, boolen- if true then it adds varibale encoding with each channel
    var_num=None,  denotes the number of varibales
    var_enco_basis='sht',  specify the basis funtion for varibale encodings
    var_enco_channels=1, number of channels for each varibale encodings
    var_enco_mode_x=50, number of x modes for each varibale encodinngs
    var_enco_mode_y=50, number of y models for each
    enable_cls_token=False, if true, learnable cls token will be added
    static_channels_num=0, Number of static channels to be conacatenated (xy grid, land/sea mask etc)
    static_features=None, The static feature (it will be taken from the Proprecssor while initializing the model)
    integral_operator_top, it is require to re-grid opration (for example: from equiangular to LG grid.)
    integral_operator_bottom,  it is require to re-grid opration (for example: from LG grid ot equiangular)
    """
    def __init__(self,
                 in_token_codim,
                 out_token_codim=None,
                 hidden_token_codim=None, 
                 lifting_token_codim=None,
                 n_layers=4,
                 n_modes = None,
                 scalings = None,
                 n_heads = 1,
                 non_linearity = F.gelu,
                 layer_kwargs = {'incremental_n_modes':None, 'use_mlp':False, 'mlp_dropout':0, 'mlp_expansion':1.0,
                 'non_linearity':F.gelu,
                 'norm':None, 'preactivation':False,
                 'fno_skip':'linear',
                 'horizontal_skip' : 'linear',
                 'mlp_skip':'linear',
                 'separable':False,
                 'factorization':None,
                 'rank':1.0,
                 'fft_norm':'forward',
                 'normalizer' : 'instance_norm',
                 'joint_factorization':False, 
                 'fixed_rank_modes':False,
                 'implementation':'factorized',
                 'decomposition_kwargs':dict(),
                 'normalizer': False},
                 per_channel_attention=False,
                 operator_block=TnoBlock2d,
                 integral_operator=SpectralConvKernel2d,
                 integral_operator_top=partial(SpectralConvKernel2d, sht_grid="legendre-gauss"), 
                 integral_operator_bottom=partial(SpectralConvKernel2d, isht_grid="legendre-gauss"),
                 re_grid_input=False,
                 re_grid_output=False,
                 projection=True,
                 lifting=True,
                 domain_padding=None,
                 domain_padding_mode='one-sided',
                 var_encoding=False, #b
                 var_num=None, # denotes the number of varibales
                 var_enco_basis='fft',
                 var_enco_channels=1,
                 var_enco_mode_x=20,
                 var_enco_mode_y=40,
                 enable_cls_token=False,
                 static_channels_num=0,
                 static_features=None,
                ):
        super().__init__()
        self.n_layers = n_layers
        assert len(n_modes) == n_layers, "number of modes for all layers are not given"
        assert len(n_heads) == n_layers, "number of Attention head for all layers are not given"
        if integral_operator_bottom is None:
            integral_operator_bottom = integral_operator
        if integral_operator_top is None:
            integral_operator_top = integral_operator
        self.n_dim = len(n_modes[0])
        self.in_token_codim = in_token_codim
        self.var_num = var_num
        if hidden_token_codim is None:
            hidden_token_codim = in_token_codim
        if lifting_token_codim is None:
            lifting_token_codim = in_token_codim
        if out_token_codim is None:
            out_token_codim = in_token_codim
        self.re_grid_input = re_grid_input
        self.re_grid_output = re_grid_output
        
        if self.re_grid_input:
            self.input_regrider = Regird("equiangular","legendre-gauss")
        if self.re_grid_output:
            self.output_regrider = Regird("legendre-gauss","equiangular")
            
        self.hidden_token_codim = hidden_token_codim
        self.n_modes = n_modes
        self.scalings = scalings
        self.var_enco_channels = var_enco_channels
        self.n_heads = n_heads
        self.integral_operator = integral_operator
        self.layer_kwargs = layer_kwargs
        self.operator_block = operator_block
        self.lifting = lifting
        self.projection = projection

        self.register_buffer("static_features", static_features)
        self.static_channels_num = static_channels_num
        ## calculating scaling
        if self.scalings is not None:
            self.end_to_end_scaling = self.get_output_scaling_factor(np.ones_like(self.scalings[0]),self.scalings)
            print("End to End Scaling", self.end_to_end_scaling)
        else:
            self.end_to_end_scaling = 1
        if isinstance(self.end_to_end_scaling, (float, int)):
            self.end_to_end_scaling = [self.end_to_end_scaling]*self.n_dim

        ## Setting up domain padding for encoder and reconstructor
        
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(domain_padding=domain_padding, padding_mode=domain_padding_mode\
            , output_scaling_factor = self.end_to_end_scaling)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        ## Code for varibale encoding
        self.var_encoding = var_encoding
        if var_encoding:
            ###
            # Each variable along with its varibale concoding should remain consecutively to be considered a single token
            # for varibale_encoding with codim = 2
            # the channels can be [Variable1, varibale_encoding1,varibale_encoding1, static_channel, varibale2, varibale_encoding2,varibale_encoding2, static_channel, .....]
            # Each token is extracted accordingly in the attention module
            ###
            
            assert var_num is not None
            print("Using Variable encoding")

            self.var_encoding_funtions = VaribaleEncoding2d(var_num*self.var_enco_channels, mode_x=var_enco_mode_x,\
                                                            mode_y=var_enco_mode_y, basis=var_enco_basis)
            self.variable_channels = [i*(self.static_channels_num + self.var_enco_channels+1) for i in range(var_num)]
            self.static_cahnnels = []
            
            if self.static_channels_num != 0:
                for i in self.variable_channels:
                    for j in range(i+1,i+self.static_channels_num+1,1):
                        self.static_cahnnels.append(j)
                
            self.encoding_channels = list(set([i for i in range((self.static_channels_num +self.var_enco_channels+1)*var_num)])\
                                          - set( self.variable_channels) - set(self.static_cahnnels))
            
            
        else:
            var_enco_channels = 0
        
        ## initializing Components
        if self.lifting:
            print('Using lifing Layer')
            
            # a varibale + it's varibale encoding + the static channen together constitute a token
            
            self.lifting = Projection(in_channels=self.in_token_codim+var_enco_channels+self.static_channels_num, out_channels=hidden_token_codim,\
                                    hidden_channels=lifting_token_codim, n_dim=self.n_dim, permutation_invariant=True) #Permutation
        elif var_encoding:
            hidden_token_codim = self.in_token_codim+var_enco_channels+self.static_channels_num
        
        if enable_cls_token:
            count = 1
        else:
            count = 0
        self.codim_size = hidden_token_codim * (var_num+count) # +1 is for the CLS token
        
        print("expected number of channels", self.codim_size)
        
        self.base = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0 and self.n_layers!=1:
                conv_op = integral_operator_top
            elif i == self.n_layers -1 and self.n_layers!=1:
                conv_op = integral_operator_bottom
            else:
                conv_op = self.integral_operator

            self.base.append(self.operator_block(
                                            n_modes=self.n_modes[i],
                                            n_head = self.n_heads[i],
                                            token_codim = hidden_token_codim,
                                            output_scaling_factor = [self.scalings[i]],
                                            SpectralConv = conv_op,
                                            codim_size=self.codim_size,
                                            per_channel_attention=per_channel_attention,
                                            **self.layer_kwargs))
        if self.projection:
            print("Using Projection Layer")
            self.projection = Projection(in_channels=hidden_token_codim, out_channels=out_token_codim,\
                                         hidden_channels=lifting_token_codim,non_linearity=non_linearity,\
                                         n_dim=self.n_dim, permutation_invariant=True) #permutation
        
        ### Code for varibale encoding
        
        self.enable_cls_token = enable_cls_token
        if enable_cls_token:
            print("intializing CLS token")
            self.cls_token = VaribaleEncoding2d(hidden_token_codim, var_enco_mode_x, var_enco_mode_y, basis=var_enco_basis)

            
                                        
    def get_output_scaling_factor(self, initial_scale, scalings_per_layer):
        for k in scalings_per_layer:
            initial_scale = np.multiply(initial_scale, k)
        initial_scale = initial_scale.tolist()
        if len(initial_scale) == 1:
            initial_scale = initial_scale[0]
        return initial_scale
        
    def get_device(self,):
        return self.cls_token.coefficients_r.device
    
    def forward(self, inp):
        if self.re_grid_input:
            inp = self.input_regrider(inp)
            
        if self.var_encoding:
            x = torch.zeros((inp.shape[0],len(self.variable_channels)+len(self.encoding_channels)+len(self.static_cahnnels),\
                             inp.shape[2], inp.shape[3]), device=inp.device, dtype=inp.dtype)
            var_encoding = self.var_encoding_funtions(x).to(x.device)
            x[:,self.variable_channels,:,:] = inp
            x[:,self.encoding_channels,:,:] = var_encoding[None,:,:,:].repeat(x.shape[0],1,1,1)
            if self.static_channels_num !=0:
                x[:,self.static_cahnnels,:,:] = \
                self.static_features[:,:,:,:].repeat(x.shape[0],len(self.static_cahnnels)//self.static_features.shape[1],1,1)
        else:
            x = inp
            

        if self.lifting:
            x = self.lifting(x)
        
        if self.enable_cls_token:
            cls_token = self.cls_token(x)
            x = torch.cat([cls_token[None,:,:,:].repeat(x.shape[0],1,1,1), x], dim=1)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        output_shape_en = [int(round(i*j)) for (i,j) in zip(x.shape[-self.n_dim:], self.end_to_end_scaling)]

        cur_output_shape = None
        for layer_idx in range(self.n_layers):                
            if layer_idx == self.n_layers -1:
                cur_output_shape = output_shape_en
            x = self.base[layer_idx](x, output_shape = cur_output_shape)
        if self.projection:
            x = self.projection(x)
        
        if self.re_grid_output:
            x = self.output_regrider(x)

        return x