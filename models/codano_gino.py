from .codano import CodANO
from layers.gno_layer import GnoPremEq
from layers.codano_block_2D import CodanoBlocks2d
from layers.fino_2D import SpectralConvKernel2d
from functools import partial
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from layers.regrider import Regird
from neuralop.layers.padding import DomainPadding
import numpy as np
import torch
from layers.variable_encoding import VariableEncoding2d

# TODO replace with nerualop.MLP module


class Projection(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        n_dim=2,
        non_linearity=F.gelu,
        permutation_invariant=False,
    ):
        """Permutation invariant projection layer.

        Performs linear projections on each channel separately.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (in_channels
                                if hidden_channels is None else
                                hidden_channels)
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')

        self.permutation_invariant = permutation_invariant

        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.norm = nn.InstanceNorm2d(hidden_channels, affine=True)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        batch = x.shape[0]
        if self.permutation_invariant:
            assert x.shape[1] % self.in_channels == 0, \
                "Total Number of Channels is not divisible by number of tokens"
            x = rearrange(x, 'b (g c) h w -> (b g) c h w', c=self.in_channels)

        x = self.fc1(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        if self.permutation_invariant:
            x = rearrange(x, '(b g) c h w -> b (g c) h w', b=batch)
        return x


class CondnoGino(nn.Module):
    def __init__(self,
                 in_token_codim,
                 input_grid,
                 output_grid=None,
                 grid_size=None,
                 radius=None,
                 n_neigbor=10,
                 fixed_neighbour=False,
                 out_token_codim=None,
                 hidden_token_codim=None,
                 lifting_token_codim=None,
                 kqv_non_linear=False,
                 n_layers=4,
                 n_modes=None,
                 scalings=None,
                 n_heads=1,
                 layer_kwargs={'incremental_n_modes': None,
                               'use_mlp': False,
                               'mlp_dropout': 0,
                               'mlp_expansion': 1.0,
                               'non_linearity': torch.sin,
                               'norm': None, 'preactivation': False,
                               'fno_skip': 'linear',
                               'horizontal_skip': 'linear',
                               'mlp_skip': 'linear',
                               'separable': False,
                               'factorization': None,
                               'rank': 1.0,
                               'fft_norm': 'forward',
                               'normalizer': 'instance_norm',
                               'joint_factorization': False,
                               'fixed_rank_modes': False,
                               'implementation': 'factorized',
                               'decomposition_kwargs': dict(),
                               'normalizer': False},
                 operator_block=CodanoBlocks2d,
                 per_channel_attention=False,
                 integral_operator=SpectralConvKernel2d,
                 integral_operator_top=None,
                 integral_operator_bottom=None,
                 re_grid_input=False,
                 re_grid_output=False,
                 projection=True,
                 gno_mlp_layers=None,
                 lifting=True,
                 domain_padding=None,
                 domain_padding_mode='one-sided',
                 var_encoding=False,
                 var_num=None,  # denotes the number of varibales
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
        assert len(
            n_modes) == n_layers, "number of modes for all layers are not given"
        assert len(
            n_heads) == n_layers, "number of Attention head for all layers are not given"
        if output_grid is None:
            output_grid = input_grid.clone()
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
            self.input_regrider = Regird("equiangular", "legendre-gauss")
        if self.re_grid_output:
            self.output_regrider = Regird("legendre-gauss", "equiangular")

        self.input_grid = input_grid
        self.output_grid = output_grid
        self.grid_size = grid_size

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

        self.radius = radius
        self.n_neigbor = n_neigbor
        self.fixed_neighbour = fixed_neighbour
        self.gno_mlp_layers = gno_mlp_layers
        self.per_channel_attention = per_channel_attention

        self.register_buffer("static_features", static_features)
        self.static_channels_num = static_channels_num
        # calculating scaling
        if self.scalings is not None:
            self.end_to_end_scaling = self.get_output_scaling_factor(
                np.ones_like(self.scalings[0]), self.scalings)
            print("End to End Scaling", self.end_to_end_scaling)
        else:
            self.end_to_end_scaling = 1
        if isinstance(self.end_to_end_scaling, (float, int)):
            self.end_to_end_scaling = [self.end_to_end_scaling] * self.n_dim

        # Setting up domain padding for encoder and reconstructor

        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=self.end_to_end_scaling)
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        # Code for varibale encoding

        # initializing Components
        if self.lifting:
            print('Using lifing Layer')

            # a varibale + it's varibale encoding + the static channen together
            # constitute a token

            self.lifting = GnoPremEq(
                var_num=var_num,
                in_dim=self.in_token_codim,
                out_dim=hidden_token_codim,
                input_grid=self.input_grid,
                output_grid=self.output_grid,
                projection_hidden_dim=lifting_token_codim,
                mlp_layers=self.gno_mlp_layers,
                radius=self.radius,
                n_neigbor=n_neigbor,
                fixed_neighbour=fixed_neighbour,
                var_encoding=var_encoding,
                var_encoding_channels=var_enco_channels)

        elif var_encoding:
            hidden_token_codim = self.in_token_codim + \
                var_enco_channels + self.static_channels_num

        if enable_cls_token:
            count = 1
        else:
            count = 0
        self.codim_size = hidden_token_codim * \
            (var_num + count)  # +1 is for the CLS token

        print("expected number of channels", self.codim_size)

        self.base = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0 and self.n_layers != 1:
                conv_op = integral_operator_top
            elif i == self.n_layers - 1 and self.n_layers != 1:
                conv_op = integral_operator_bottom
            else:
                conv_op = self.integral_operator

            self.base.append(self.operator_block(
                n_modes=self.n_modes[i],
                n_head=self.n_heads[i],
                token_codimension=hidden_token_codim,
                output_scaling_factor=[self.scalings[i]],
                SpectralConvolution=conv_op,
                codimension_size=self.codim_size,
                per_channel_attention=self.per_channel_attention,
                kqv_non_linear=kqv_non_linear,
                **self.layer_kwargs))
        if self.projection:
            print("Using Projection Layer")
            self.projection = GnoPremEq(
                var_num=var_num,
                in_dim=self.hidden_token_codim,
                out_dim=self.hidden_token_codim,
                input_grid=self.output_grid,
                output_grid=self.input_grid,
                mlp_layers=self.gno_mlp_layers,
                radius=self.radius,
                n_neigbor=n_neigbor,
                fixed_neighbour=fixed_neighbour,
                var_encoding=False,
                projection_hidden_dim=lifting_token_codim,
                var_encoding_channels=0,
                end_projection=True,
                end_projection_outdim=out_token_codim)

        # Code for varibale encoding

        self.enable_cls_token = enable_cls_token
        if enable_cls_token:
            print("intializing CLS token")
            self.cls_token = VariableEncoding2d(
                1, hidden_token_codim, (var_enco_mode_x, var_enco_mode_y), basis=var_enco_basis)

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
        '''
        inp = (batch_size, n_points, in_dims/Channels)
        currenly only batch_size = 1
        '''
        inp = inp[0, :, :]
        inp = inp[None, ...]
        if self.re_grid_input:
            inp = self.input_regrider(inp)
        if self.lifting:
            x = self.lifting(inp)
            x = rearrange(x, 'b (h w) c -> b c h w', h=self.grid_size[0])
        else:
            x = inp

        if self.enable_cls_token:
            cls_token = self.cls_token(x)
            x = torch.cat([cls_token[None, :, :, :].repeat(
                x.shape[0], 1, 1, 1), x], dim=1)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        output_shape_en = [int(round(i * j)) for (i,
                                                  j) in zip(x.shape[-self.n_dim:],
                                                            self.end_to_end_scaling)]

        cur_output_shape = None
        for layer_idx in range(self.n_layers):
            if layer_idx == self.n_layers - 1:
                cur_output_shape = output_shape_en
            x = self.base[layer_idx](x, output_shape=cur_output_shape)

        if self.re_grid_output:
            x = self.output_regrider(x)
        if self.projection:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.projection(x)

        return x
