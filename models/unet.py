from layers.gno_layer import GNO
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from layers.regrider import Regird
import numpy as np
import torch
from layers.unet_sublayer import UNet2d


class UnetGno(nn.Module):
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
                 pad_to_size=None,
                 initial_mesh=None,
                 non_linearity=F.gelu,
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
        if output_grid is None:
            output_grid = None  # input_grid.clone()

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
        self.lifting = lifting
        self.projection = projection
        self.radius = radius
        self.fixed_neighbour = fixed_neighbour
        self.n_neigbor = n_neigbor
        self.gno_mlp_layers = gno_mlp_layers
        self.pad_to_size = pad_to_size

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

        self.base = UNet2d(in_channels=hidden_dim,
                           out_channels=hidden_dim, init_features=hidden_dim)

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

        pad_size_1 = self.pad_to_size[-2] - x.shape[-2]
        pad_size_2 = self.pad_to_size[-1] - x.shape[-1]
        x = nn.functional.pad(
            x,
            (pad_size_2 // 2,
             pad_size_2 - pad_size_2 // 2,
             pad_size_1 // 2,
             pad_size_1 - pad_size_1 // 2))
        x = self.base(x)



        x = x[:, :, pad_size_1 // 2: -(pad_size_1 - pad_size_1 // 2),
              pad_size_2 // 2: -(pad_size_2 - pad_size_2 // 2)]

        if self.re_grid_output:
            x = self.output_regrider(x)
        if self.projection:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.projection(x)
        return x
