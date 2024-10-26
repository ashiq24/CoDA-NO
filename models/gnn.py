from layers.gnn_layer import GnnLayer
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.mlp import MLPLinear
import numpy as np
import torch

class GNN(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 input_grid,
                 output_grid=None,
                 n_neigbor=None,
                 hidden_dim=None,
                 lifting_dim=None,
                 n_layers=4,
                 initial_mesh=None,
                 non_linearity=F.gelu,
                 projection=True,
                 gno_mlp_layers=None,
                 lifting=True,
                 ):
        super().__init__()
        self.n_layers = n_layers

        if output_grid is None:
            output_grid = input_grid.clone()

        self.n_dim = input_grid.shape[-1]

        self.in_dim = in_dim

        if hidden_dim is None:
            hidden_dim = in_dim
        if lifting_dim is None:
            lifting_dim = in_dim
        if out_dim is None:
            out_dim = in_dim

        self.input_grid = input_grid
        self.output_grid = output_grid

        self.hidden_dim = hidden_dim

        self.lifting = lifting
        self.projection = projection
        self.n_neigbor = n_neigbor
        self.gno_mlp_layers = gno_mlp_layers

        self.initial_mesh = initial_mesh
        # Code for varibale encoding

        # initializing Components
        if self.lifting:
            print('Using lifing Layer')
            self.lifting = MLPLinear(
                layers=[self.in_dim, self.hidden_dim],
            )

        self.base = nn.ModuleList([])
        for i in range(self.n_layers):
            self.base.append(GnnLayer(
                hidden_dim,
                hidden_dim,
                self.initial_mesh,
                self.initial_mesh,
                gno_mlp_layers,
                lifting_dim,
                n_neigbor))

        if self.projection:
            print("Using Projection Layer")
            self.projection = MLPLinear(
                layers=[self.hidden_dim, out_dim]
            )

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
                for i in range(self.n_layers):
                    if i == self.n_layers - 1:
                        # Doing different mesh for last layer
                        in_grid = self.initial_mesh + in_grid_displacement

                        out_grid = self.initial_mesh + out_grid_displacement
                    else:
                        in_grid = self.initial_mesh + in_grid_displacement
                        out_grid = self.initial_mesh + in_grid_displacement
                    self.base[i].update_grid(
                        in_grid,
                        out_grid)

        if self.lifting:
            x = self.lifting(inp)
        else:
            x = inp
        x = x[0, ...]
        for layer_idx in range(self.n_layers):
            x = self.base[layer_idx](x)

        if self.projection:
            x = self.projection(x)
        return x[None, ...]
