from neuralop.layers.neighbor_search import NeighborSearch
from neuralop.layers.integral_transform import IntegralTransform
from neuralop.layers.mlp import MLPLinear
from baseline_utlis import FixedNeighborSearch
from einops import rearrange
from neuralop.layers.embeddings import PositionalEmbedding
import torch.nn as nn
import torch
import torch.nn.functional as F


class GnoPremEq(nn.Module):
    def __init__(
        self,
        var_num,
        in_dim, out_dim,
        input_grid,
        output_grid,
        mlp_layers,
        projection_hidden_dim,
        radius,
        var_encoding=False,
        n_neigbor=10,
        fixed_neighbour=False,
        var_encoding_channels=1,
        n_layers=2,
        postional_em_dim=4,  # always even
        end_projection=False,
        end_projection_outdim=None,
    ):
        '''
        var_num: number of variables
        in_dim: Input Condim/channel per variables
        out_dim: Output Condim/channel per variables
        input_grid: Input grid (points)
        output_grid: Output grid (points)
        mlp_layers: MLP layers (for integral operator)
        projection_hidden_dim: Before applying integral operator we have pointwise MLP. This parameter
                                determines the width of the multi-layered MLP
        radius: radius of the neighbourhood
        var_encoding: whether to use variable encoding
        var_encoding_channels: number of channels for variable encoding
        '''
        super().__init__()
        assert postional_em_dim % 2 == 0
        n_dim = input_grid.shape[-1]
        self.radius = radius
        self.fixed_neighbour = fixed_neighbour
        self.n_neigbor = n_neigbor
        self.var_num = var_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.mlp_layers = [2 * n_dim + self.out_dim] + mlp_layers + [out_dim]
        self.var_encoding = var_encoding
        self.postional_em_dim = postional_em_dim
        self.var_encoding_channels = var_encoding_channels
        self.n_layers = n_layers
        self.end_projection = end_projection
        self.end_projection_outdim = end_projection_outdim

        # get varibale encoding
        if self.var_encoding:
            self.var_encoder = MLPLinear(
                [n_dim + 2 * postional_em_dim, self.var_encoding_channels * var_num])
            self.PE = PositionalEmbedding(postional_em_dim)
            self.variable_channels = [
                i * (var_encoding_channels + self.in_dim) for i in range(var_num)]
            self.encoding_channels = list(set([i for i in range(
                (var_encoding_channels + 1) * var_num)]) - set(self.variable_channels))
        else:
            self.var_encoding_channels = 0

        # project to higher dim
        self.projection = MLPLinear([self.var_encoding_channels + self.in_dim,
                                    projection_hidden_dim, out_dim],
                                    non_linearity=F.gelu)

        # apply GNO to get  uniform grid

        self.neighbour = None
        self.neighbour_last = None
        self.update_grid()

        self.it = torch.nn.ModuleList()
        for i in range(n_layers):
            self.it.append(IntegralTransform(
                mlp_layers=self.mlp_layers,
                transform_type='nonlinear',
                mlp_non_linearity=F.gelu))

        if self.end_projection:
            self.end_projector = MLPLinear([self.out_dim,
                                            projection_hidden_dim, self.end_projection_outdim],
                                           non_linearity=F.gelu)

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
        if self.fixed_neighbour:
            NS = FixedNeighborSearch(use_open3d=False)
            self.neighbour = NS(
                input_grid.clone().cpu(),
                input_grid.clone().cpu(),
                n_neigbor=self.n_neigbor)
        else:
            NS = NeighborSearch(use_open3d=False)
            self.neighbour = NS(
                input_grid.clone().cpu(),
                input_grid.clone().cpu(),
                radius=self.radius)

        for key, value in self.neighbour.items():
            self.neighbour[key] = self.neighbour[key].cuda()

        NS_last = FixedNeighborSearch(use_open3d=False)
        self.neighbour_last = NS_last(
            input_grid.clone().cpu(),
            output_grid.clone().cpu(),
            n_neigbor=self.n_neigbor)

        for key, value in self.neighbour_last.items():
            self.neighbour_last[key] = self.neighbour_last[key].cuda()

    def _intergral_transform(self, x):
        for i in range(self.n_layers):
            if i == self.n_layers - 1:
                x = self.it[i](self.input_grid, self.neighbour_last,
                               self.output_grid, x)
                if self.end_projection:
                    x = self.end_projector(x)
            else:
                x = self.it[i](self.input_grid, self.neighbour,
                               self.input_grid, x) + x
        return x

    def forward(self, inp):
        '''
        inp : (batch_size, n_points, in_dims/Channels)
        '''

        if self.var_encoding:
            x = torch.zeros((inp.shape[0], inp.shape[1], len(
                self.variable_channels) + len(self.encoding_channels)), device=inp.device, dtype=inp.dtype)

            pe = self.PE(self.input_grid.reshape(-1))
            pe = pe.reshape(self.input_grid.shape[0], -1)
            grid_pe = torch.cat([self.input_grid, pe], axis=1)
            var_encoding = self.var_encoder(grid_pe).to(x.device)
            x[:, :, self.variable_channels] = inp
            x[:, :, self.encoding_channels] = var_encoding[None,
                                                           :, :].repeat(x.shape[0], 1, 1)
        else:
            x = inp


        x = rearrange(
            x,
            'b n (v c) -> (b n) v c',
            c=self.in_dim +
            self.var_encoding_channels)
        x = self.projection(x)

        out = None

        for i in range(x.shape[-2]):
            # print(i)

            temp = self._intergral_transform(x[:, i, :])
            if out is None:
                out = temp[None, ...]
            else:
                out = torch.cat([out, temp[None, ...]], dim=2)


        return out


class GNO(nn.Module):
    def __init__(self, in_dim, out_dim,
                 input_grid, output_grid, mlp_layers, projection_hidden_dim,
                 radius, fixed_neighbour=False, n_neigbor=10):
        '''
        var_num: number of variables
        in_dim: Input Condim/channel per variables
        out_dim: Output Condim/channel per variables
        input_grid: Input grid (points)
        output_grid: Output grid (points)
        mlp_layers: MLP layers (for integral operator)
        projection_hidden_dim: Before applying integral operator we have pointwise MLP. This parameter
                                determines the width of the multi-layered MLP
        radius: radius of the neighbourhood
        '''
        super().__init__()

        n_dim = input_grid.shape[-1]
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.mlp_layers = [2 * n_dim] + mlp_layers + [out_dim]
        self.fixed_neighbour = fixed_neighbour
        self.n_neigbor = n_neigbor
        self.radius = radius
        # project to higher dim
        self.projection = MLPLinear([self.in_dim,
                                     projection_hidden_dim, out_dim])

        self.neighbour = None
        self.update_grid()

        for key, value in self.neighbour.items():
            self.neighbour[key] = self.neighbour[key].cuda()

        self.it = IntegralTransform(mlp_layers=self.mlp_layers)

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

        if self.fixed_neighbour:
            NS = FixedNeighborSearch(use_open3d=False)
            self.neighbour = NS(
                input_grid.clone(),
                output_grid.clone(),
                n_neigbor=self.n_neigbor)
        else:
            NS = NeighborSearch(use_open3d=False)
            self.neighbour = NS(
                input_grid.clone(),
                output_grid.clone(),
                radius=self.radius)

    def forward(self, inp):
        '''
        inp : (batch_size, n_points, in_dims/Channels)
        '''

        x = inp
        x = self.projection(x)

        out = self.it(self.input_grid, self.neighbour,
                      self.output_grid, x[0, ...])

        return out[None, ...]
