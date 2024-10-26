from baseline_utlis import FixedNeighborSearch
from neuralop.layers.integral_transform import IntegralTransform
from neuralop.layers.mlp import MLPLinear
import torch.nn as nn
import torch.nn.functional as F


class GnnLayer(nn.Module):
    def __init__(self, in_dim, out_dim,
                 input_grid, output_grid, mlp_layers, projection_hidden_dim,
                 n_neigbor):
        '''
        var_num: number of variables
        in_dim: Input Condim/channel per variables
        out_dim: Output Condim/channel per variables
        input_grid: Input grid (points)
        output_grid: Output grid (points)
        mlp_layers: MLP layers (for integral operator)
        projection_hidden_dim: Before applying integral operator we have pointwise MLP. This parameter
                                determines the width of the multi-layered MLP
        n_neigbor: number of neighbours to consider
        '''
        super().__init__()

        n_dim = input_grid.shape[-1]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.mlp_layers = [2 * n_dim + out_dim] + mlp_layers + [out_dim]
        self.n_neigbor = n_neigbor
        # project to higher dim
        self.projection = MLPLinear([self.in_dim,
                                     projection_hidden_dim, out_dim])
        print([self.in_dim, projection_hidden_dim, out_dim])
        print(self.mlp_layers)
        # apply GNO to get  uniform grid
        NS = FixedNeighborSearch(use_open3d=False)

        self.neighbour = NS(
            input_grid.clone().cpu(),
            output_grid.clone().cpu(),
            n_neigbor=n_neigbor)

        for key, value in self.neighbour.items():
            self.neighbour[key] = self.neighbour[key].cuda()

        self.it = IntegralTransform(
            mlp_layers=self.mlp_layers, transform_type='nonlinear')

        self.normalize = nn.LayerNorm(out_dim)

    def update_grid(
        self,
        input_grid=None,
        output_grid=None
    ):

        if input_grid is None:
            input_grid = self.input_grid
        if output_grid is None:
            output_grid = self.output_grid

        input_grid = input_grid.clone()
        self.input_grid = self.input_grid[:input_grid.shape[0], :]
        self.output_grid = self.output_grid[:output_grid.shape[0], :]

        NS = FixedNeighborSearch(use_open3d=False)

        self.neighbour = NS(
            input_grid.clone(),
            output_grid.clone(),
            n_neigbor=self.n_neigbor)
        for key, value in self.neighbour.items():
            self.neighbour[key] = self.neighbour[key].cuda()

    def forward(self, inp):
        '''
        inp : (batch_size, n_points, in_dims/Channels)
        '''
        x = inp
        x = self.projection(x)
        out = self.it(self.input_grid, self.neighbour,
                      self.output_grid, x)


        if out.shape == x.shape:
            out = out + x
        out = self.normalize(out)
        return out
