from .codano import CodANO
from layers.gnn_layer import GnnLayer
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from functools import partial
from einops import rearrange
import torch.nn as nn
import torch.nn.functional as F
from layers.regrider import Regird
from neuralop.layers.padding import DomainPadding
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.mlp import MLPLinear
import numpy as np
import torch
from layers.variable_encoding import VariableEncoding2d


class DeepONet(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 input_grid,
                 output_grid=None,
                 branch_layers= [128],
                 trunk_layers= [128],
                 initial_mesh=None,
                 ):
        super().__init__()
        if output_grid is None:
            output_grid = input_grid.clone()
        self.n_dim = input_grid.shape[-1]

        self.in_dim = in_dim
        print("in_dim", in_dim)
        if out_dim is None:
            out_dim = in_dim
        self.out_dim = out_dim

        self.input_grid = input_grid
        self.output_grid = output_grid
        self.register_buffer('initial_mesh', initial_mesh)
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.branch = self.get_branch()
        self.trunk = self.get_trunk()
        self.bias = nn.Parameter(torch.zeros(out_dim))

        # Code for varibale encoding

    def get_branch(self, ):
        dim1 = (self.in_dim) * self.input_grid.shape[0]
        self.branch_layers = [dim1] + self.branch_layers
        layers = []
        for i in range(len(self.branch_layers)-1):
            layers.append(nn.Linear(self.branch_layers[i], self.branch_layers[i + 1]))
            torch.nn.init.xavier_normal_(layers[-1].weight)
            if i != len(self.branch_layers) - 2:
                layers.append(nn.LayerNorm(self.branch_layers[i+1]))
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)
    
    def get_trunk(self, ):
        dim1 = self.n_dim
        self.trunk_layers = [dim1] + self.trunk_layers
        self.trunk_layers[-1] = self.trunk_layers[-1] * self.out_dim
        layers = []
        for i in range(len(self.trunk_layers)-1):
            layers.append(nn.Linear(self.trunk_layers[i], self.trunk_layers[i + 1]))
            torch.nn.init.xavier_normal_(layers[-1].weight)
            #if i != len(self.trunk_layers) - 2:
            layers.append(nn.LayerNorm(self.trunk_layers[i+1]))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

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
                # print("Doing different mesh for last layer")
                in_grid = self.initial_mesh+in_grid_displacement
                out_grid = self.initial_mesh+out_grid_displacement
                # print("in_grid", in_grid)
                # print("out_grid", out_grid)
                # print("=====================================")
        # data is concatenated with grid/positional information
        in_data = inp.reshape(inp.shape[0], -1) #torch.cat([inp, in_grid.unsqueeze(0)], dim=-1).reshape(inp.shape[0], -1)
        bout = self.branch(in_data) # (btach , dim)
        tout = self.trunk(out_grid) # (ngrid, dim * out_dim)
        tout = tout.reshape(out_grid.shape[0], self.out_dim, -1) # (ngrid, out_dim, dim)
        out = torch.einsum('bd,ncd->bnc', bout, tout)

        return out + self.bias
