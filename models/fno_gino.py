from layers.gno_layer import GNO
from layers.fino_2D import SpectralConvKernel2d
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from layers.regrider import Regird
from neuralop.layers.padding import DomainPadding
from neuralop.layers.fno_block import FNOBlocks
import numpy as np
import torch
from layers.variable_encoding import VariableEncoding2d


class FnoGno(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 input_grid,
                 output_grid=None,
                 grid_size=None,
                 radius=None,
                 fixed_neighbour=False,
                 n_neigbor=10,
                 hidden_dim=None,
                 lifting_dim=None,
                 n_layers=4,
                 max_n_modes=None,
                 n_modes=None,
                 scalings=None,
                 initial_mesh=None,
                 non_linearity=F.gelu,
                 layer_kwargs={'use_mlp': False,
                               'mlp_dropout': 0,
                               'mlp_expansion': 1.0,
                               'non_linearity': F.gelu,
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
                 operator_block=FNOBlocks,
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
                 ):
        super().__init__()
        self.n_layers = n_layers
        assert len(
            n_modes) == n_layers, "number of modes for all layers are not given"
        if output_grid is None:
            output_grid = input_grid.clone()
        if integral_operator_bottom is None:
            integral_operator_bottom = integral_operator
        if integral_operator_top is None:
            integral_operator_top = integral_operator
        self.n_dim = len(max_n_modes[0])
        self.in_dim = in_dim
        if hidden_dim is None:
            hidden_dim = in_dim
        if lifting_dim is None:
            lifting_dim = in_dim
        if out_dim is None:
            out_dim = in_dim
        self.re_grid_input = re_grid_input
        self.re_grid_output = re_grid_output

        if self.re_grid_input:
            self.input_regrider = Regird("equiangular", "legendre-gauss")
        if self.re_grid_output:
            self.output_regrider = Regird("legendre-gauss", "equiangular")

        self.input_grid = input_grid
        self.output_grid = output_grid
        self.grid_size = grid_size

        self.hidden_dim = hidden_dim
        self.n_modes = n_modes
        self.max_n_modes = max_n_modes
        self.scalings = scalings
        self.integral_operator = integral_operator
        self.layer_kwargs = layer_kwargs
        self.operator_block = operator_block
        self.lifting = lifting
        self.projection = projection
        self.radius = radius
        self.fixed_neighbour = fixed_neighbour
        self.n_neigbor = n_neigbor
        self.gno_mlp_layers = gno_mlp_layers

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
        self.initial_mesh = initial_mesh
        # Code for varibale encoding

        # initializing Components
        if self.lifting:
            print('Using lifing Layer')

            # a varibale + it's varibale encoding + the static channen together
            # constitute a token

            self.lifting = GNO(
                in_dim=self.in_dim,
                out_dim=hidden_dim,
                input_grid=self.input_grid,
                output_grid=self.output_grid,
                projection_hidden_dim=lifting_dim,
                mlp_layers=self.gno_mlp_layers,
                radius=self.radius,
                fixed_neighbour=self.fixed_neighbour,
                n_neigbor=self.n_neigbor)

        self.base = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0 and self.n_layers != 1:
                conv_op = integral_operator_top
            elif i == self.n_layers - 1 and self.n_layers != 1:
                conv_op = integral_operator_bottom
            else:
                conv_op = self.integral_operator

            self.base.append(self.operator_block(
                hidden_dim,
                hidden_dim,
                max_n_modes=self.max_n_modes[i],
                n_modes=self.n_modes[i],
                output_scaling_factor=[self.scalings[i]],
                SpectralConv=conv_op,
                **self.layer_kwargs))
        if self.projection:
            # input and output grid is swapped

            print("Using Projection Layer")
            self.projection = GNO(
                in_dim=self.hidden_dim,
                out_dim=out_dim,
                input_grid=self.output_grid,
                projection_hidden_dim=lifting_dim,
                output_grid=self.input_grid,
                mlp_layers=self.gno_mlp_layers,
                radius=self.radius,
                fixed_neighbour=self.fixed_neighbour,
                n_neigbor=self.n_neigbor)

    def get_output_scaling_factor(self, initial_scale, scalings_per_layer):
        for k in scalings_per_layer:
            initial_scale = np.multiply(initial_scale, k)
        initial_scale = initial_scale.tolist()
        if len(initial_scale) == 1:
            initial_scale = initial_scale[0]
        return initial_scale

    def get_device(self,):
        return self.cls_token.coefficients_r.device

    def forward(
            self,
            inp,
            out_grid_displacement=None,
            in_grid_displacement=None):
        '''
        inp = (batch_size, n_points, in_dims/Channels)
        currenly only batch_size = 1
        '''
        if out_grid_displacement is not None:
            with torch.no_grad():
                self.lifting.update_grid(
                    self.initial_mesh + in_grid_displacement, None)
                self.projection.update_grid(
                    None, self.initial_mesh + out_grid_displacement)

        if self.re_grid_input:
            inp = self.input_regrider(inp)
        if self.lifting:
            # print("In Lifting")
            x = self.lifting(inp)
            x = rearrange(x, 'b (h w) c -> b c h w', h=self.grid_size[0])
        else:
            x = inp

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
            # print("projection")
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.projection(x)
        return x
