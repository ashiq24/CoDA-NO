from layers.gnn_layer import GnnLayer
import torch.nn as nn
import numpy as np
import torch
from layers.gnn_layer import GnnLayer
from neuralop.layers.embeddings import PositionalEmbedding


class DeepONet(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 input_grid,
                 output_grid=None,
                 branch_layers=[128],
                 trunk_layers=[128],
                 initial_mesh=None,
                 positional_encoding_dim=8,
                 n_neigbor=10,
                 gno_mlp_layers=None,
                 ):
        super().__init__()
        if output_grid is None:
            output_grid = input_grid.clone()
        self.n_dim = input_grid.shape[-1]
        self.n_neigbor = n_neigbor
        self.gno_mlp_layers = gno_mlp_layers
        self.in_dim = in_dim
        print("in_dim", in_dim)
        if out_dim is None:
            out_dim = in_dim
        self.out_dim = out_dim
        self.positional_encoding_dim = positional_encoding_dim
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.initial_mesh = initial_mesh
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.gnn = None
        self.branch = self.get_branch()
        self.trunk = self.get_trunk()
        self.PE = PositionalEmbedding(positional_encoding_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Code for varibale encoding

    def get_branch(self, ):
        dim1 = self.in_dim + self.n_dim * self.positional_encoding_dim
        # self.branch_layers = [dim1] + self.branch_layers
        self.gnn = GnnLayer(
            dim1,
            self.branch_layers[0],
            self.initial_mesh,
            self.initial_mesh,
            self.gno_mlp_layers,
            self.branch_layers[0],
            self.n_neigbor)
        self.layer_norm = nn.LayerNorm(self.branch_layers[0])
        layers = []

        self.branch_layers[0] = self.branch_layers[0] * \
            self.input_grid.shape[0]
        for i in range(len(self.branch_layers) - 1):
            layers.append(
                nn.Linear(self.branch_layers[i], self.branch_layers[i + 1]))
            torch.nn.init.xavier_normal_(layers[-1].weight)
            if i != len(self.branch_layers) - 2:
                # layers.append(nn.LayerNorm(self.branch_layers[i+1]))
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def get_trunk(self, ):
        dim1 = self.n_dim + self.positional_encoding_dim * self.n_dim
        self.trunk_layers = [dim1] + self.trunk_layers
        self.trunk_layers[-1] = self.trunk_layers[-1] * self.out_dim
        layers = []
        for i in range(len(self.trunk_layers) - 1):
            layers.append(
                nn.Linear(self.trunk_layers[i], self.trunk_layers[i + 1]))
            torch.nn.init.xavier_normal_(layers[-1].weight)
            # if i != len(self.trunk_layers) - 2:
            # layers.append(nn.LayerNorm(self.trunk_layers[i+1]))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def get_pe(self, grid):
        pe = self.PE(grid.reshape(-1))
        pe = pe.reshape(grid.shape[0], -1)
        return pe

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
                in_grid = self.initial_mesh + in_grid_displacement
                out_grid = self.initial_mesh + out_grid_displacement
                self.gnn.update_grid(in_grid.clone(), in_grid.clone())

        in_pe = self.get_pe(in_grid)
        in_data = torch.cat([inp, in_pe.unsqueeze(0)], dim=-1)

        bout = self.gnn(in_data[0])[None, ...]  # (btach , dim)

        bout = self.layer_norm(bout)

        bout = self.branch(bout.reshape(inp.shape[0], -1))

        bout = bout / np.sqrt(self.branch_layers[-1])

        pe = self.get_pe(out_grid)  # self.PE(out_grid.reshape(-1))
        # pe = pe.reshape(out_grid.shape[0], -1)
        grid_pe = torch.cat([out_grid, pe], axis=1)

        tout = self.trunk(grid_pe)  # (ngrid, dim * out_dim)
        # (ngrid, out_dim, dim)
        tout = tout.reshape(out_grid.shape[0], self.out_dim, -1)

        out = torch.einsum('bd,ncd->bnc', bout, tout)

        return out + self.bias
