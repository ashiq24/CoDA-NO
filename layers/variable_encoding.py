from functools import reduce
from typing import Tuple
from neuralop.layers.mlp import MLPLinear
import numpy as np
import torch
from torch import nn
import torch_harmonics as th
from neuralop.layers.embeddings import PositionalEmbedding


class VariableEncoding2d(nn.Module):
    def __init__(self,
                 n_variables: int,
                 variable_encoding_size: int,
                 modes: Tuple[int, ...],
                 basis='fft') -> None:
        super().__init__()
        self.modes = modes
        channel = n_variables * variable_encoding_size
        self.coefficients_r = nn.Parameter(
            torch.empty(channel, *modes))
        self.coefficients_i = nn.Parameter(
            torch.empty(channel, *modes))
        self.reset_parameters()
        self.basis = basis
        if basis == 'fft':
            self.transform = torch.fft.ifft2
        elif basis == 'sht':
            self.transform = th.InverseRealSHT(
                *modes,
                lmax=modes[-2],
                mmax=modes[-1],
                grid="legendre-gauss",
                norm="backward",
            )

    def reset_parameters(self):
        std = (1 / (self.modes[-1] * self.modes[-2]))**0.5
        torch.nn.init.normal_(self.coefficients_r, mean=0.0, std=std)
        torch.nn.init.normal_(self.coefficients_i, mean=0.0, std=std)

    def forward(self, x):
        """Take a resolution and outputs the positional encodings"""
        size_x, size_y = x.shape[-2], x.shape[-1]
        if self.basis == 'sht':
            if self.transform.nlat == size_x and self.transform.nlon == size_y:
                return self.transform(
                    self.coefficients_r + 1.0j * self.coefficients_i)

            self.transform = th.InverseRealSHT(
                size_x,
                size_y,
                lmax=self.modes[-2],
                mmax=self.modes[-1],
                grid="legendre-gauss",
                norm='backward'
            ).to(
                device=self.coefficients_i.device,
                dtype=self.coefficients_i.dtype
            )
            return self.transform(
                self.coefficients_r + 1.0j * self.coefficients_i,
                s=(size_x, size_y)
            ).real

        else:
            return self.transform(
                self.coefficients_r + 1.0j * self.coefficients_i,
                s=(size_x, size_y)
            ).real


# SHT doesn't make sense for 3 dimenstional data
class FourierVariableEncoding3D(nn.Module):
    def __init__(self, n_features: int, modes: Tuple[int, ...]) -> None:
        super().__init__()
        if len(modes) != 3:
            raise ValueError(
                f"Expected 3 frequency modes, but got {len(modes)} modes:\n{modes=}")

        self.modes = modes
        self.weights_re = nn.Parameter(torch.empty(n_features, *modes))
        self.weights_im = nn.Parameter(torch.empty(n_features, *modes))
        self.reset_parameters()
        # self.transform = torch.fft.ifftn

    def reset_parameters(self):
        std = 1 / np.sqrt(reduce(lambda a, b: a * b, self.modes))
        torch.nn.init.normal_(self.weights_re, mean=0.0, std=std)
        torch.nn.init.normal_(self.weights_im, mean=0.0, std=std)

    def forward(self, x_shape):
        """Take a resolution and outputs the positional encodings"""
        _, size_t, size_x, size_y = x_shape[0], x_shape[1], x_shape[2], x_shape[3]
        s = (size_t, size_x, size_y)
        # now check if s is a tuple of ints
        if not all(isinstance(i, int) for i in s):
            raise ValueError(
                f"Expected a tuple of integers, but got {s=}")
        else:
            s = tuple(s)
        return torch.fft.ifftn(
            self.weights_re + 1.0j * self.weights_im,
            s=s,
            norm="forward",  # don't multiply by any normalization factor
        ).real


class VariableEncodingIrregularMesh(nn.Module):
    def __init__(
        self,
        n_variables: int,
        variable_encoding_size: int,
        n_dim: int = 2,
        positional_encoding_dim: int = 8
    ) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.variable_encoding_size = variable_encoding_size
        self.n_dim = n_dim
        self.positional_encoding_dim = positional_encoding_dim
        self.var_encoder = MLPLinear(
            [n_dim + self.n_dim * positional_encoding_dim, self.variable_encoding_size * n_variables])
        self.PE = PositionalEmbedding(positional_encoding_dim)

    def forward(self, grid_poits):
        pe = self.PE(grid_poits.reshape(-1))
        pe = pe.reshape(grid_poits.shape[0], -1)
        grid_pe = torch.cat([grid_poits, pe], axis=1)
        var_encoding = self.var_encoder(grid_pe)
        return var_encoding


class VariableEncodingWrapper(nn.Module):
    def __init__(
            self,
            equation_dict: dict,
            variable_encoding_size: int,
            n_dim: int = 2,
            positional_encoding_dim: int = 8,
            varibale_encoding_modes: Tuple[int, ...] = (32, 32),
            basis='fft',
            uniform=False) -> None:
        '''
        For each equation in the equation_dict, we create a VariableEncodingIrregularMesh
        dic is of form {"Equation": n_variables, ...}
        '''
        super().__init__()
        self.n_dim = n_dim
        self.equation_dict = equation_dict
        self.variable_encoding_size = variable_encoding_size
        self.model_dict = nn.ModuleDict()
        self.uniform = uniform
        for i in equation_dict.keys():
            if not uniform:
                self.model_dict[i] = VariableEncodingIrregularMesh(
                    n_variables=equation_dict[i],
                    variable_encoding_size=self.variable_encoding_size,
                    n_dim=n_dim,
                    positional_encoding_dim=positional_encoding_dim
                )
            else:
                self.model_dict[i] = VariableEncoding2d(
                    n_variables=equation_dict[i],
                    variable_encoding_size=self.variable_encoding_size,
                    modes=varibale_encoding_modes,
                    basis=basis
                )

    def load_encoder(self, equation: str, path: str):
        self.model_dict[equation].load_state_dict(
            torch.load(path, map_location=torch.device('cpu')))

    def save_encoder(self, equation: str, path: str):
        torch.save(self.model_dict[equation].state_dict(), path)

    def save_all_encoder(self, path: str):
        for i in self.equation_dict.keys():
            torch.save(self.model_dict[i].state_dict(), path + f"_{i}" + ".pt")

    def freeze(self, equation: str):
        for param in self.model_dict[equation].parameters():
            param.requires_grad = False

    def forward(self, grid_poits, equation: str = None):
        '''
        grid_poits: (n_points, n_dim) or for uniform mesh input tensor of shape (D, channels, H, W)
        '''
        encoding_list = []
        if equation is None:
            equation = list(self.equation_dict.keys())
        for i in equation:
            encoding_list.append(self.model_dict[i](grid_poits))

        if self.uniform:
            return torch.cat(encoding_list, axis=0).unsqueeze(0)
        else:
            return torch.cat(encoding_list, axis=1).unsqueeze(0)


def get_variable_encoder(params):
    return VariableEncodingWrapper(
        params.equation_dict,
        variable_encoding_size=params.n_encoding_channels,
        n_dim=params.n_dim,
        positional_encoding_dim=params.positional_encoding_dim,
        uniform=params.grid_type == 'uniform',
        varibale_encoding_modes=(
            params.encoding_modes_x,
            params.encoding_modes_y) if hasattr(
            params,
            'encoding_modes_y') else None)



class TokenExpansion(nn.Module):
    def __init__(
            self,
            n_variables: int,
            n_encoding_channels,
            n_static_channels: int,
            uniform_grid=False) -> None:
        """
        stack the variables and the corresponsing encodings together

        Args:
            n_variables (int): number of variables
            n_encoding_channels (int): number of encoding channels
            n_static_channels (int): number of static channels
            unifor_grid (bool): if the grid is uniform
        
        """
        super().__init__()
        self.n_variables = n_variables
        self.n_encoding_channels = n_encoding_channels
        self.n_static_channels = n_static_channels
        self.uniform_grid = uniform_grid

        expansion_factor = 1 + self.n_static_channels + self.n_encoding_channels

        self.variable_channels = [
            i * expansion_factor for i in range(n_variables)]
        self.static_channels = []
        if self.n_static_channels != 0:
            for v in self.variable_channels:
                self.static_channels.extend(
                    range(v + 1, v + self.n_static_channels + 1))
        self.encoding_channels = []
        if self.n_encoding_channels != 0:
            self.encoding_channels = sorted(list(
                set(range(n_variables * expansion_factor))
                - set(self.variable_channels)
                - set(self.static_channels)
            ))

        print(self.variable_channels)
        print(self.static_channels)
        print(self.encoding_channels)

    def forward(
            self,
            inp: torch.Tensor,
            variable_encodings: torch.tensor,
            static_channels: torch.tensor) -> torch.Tensor:
        """
        x: (batch_size, points, n_variables)
        """
        if not self.uniform_grid:
            x = torch.zeros((inp.shape[0], inp.shape[1], len(self.variable_channels) + len(
                self.encoding_channels) + len(self.static_channels)), device=inp.device, dtype=inp.dtype)
            x[:, :, self.variable_channels] = inp
            if self.n_static_channels != 0:
                x[:, :, self.static_channels] = static_channels.repeat(
                    x.shape[0], 1, 1)
            if self.n_encoding_channels != 0:
                x[:, :, self.encoding_channels] = variable_encodings.repeat(

                    x.shape[0], 1, 1)

        else:
            # current support for only 2D

            x = torch.zeros((inp.shape[0], len(
                self.variable_channels) + len(self.encoding_channels) + len(self.static_channels),
                inp.shape[-2], inp.shape[-1]), device=inp.device, dtype=inp.dtype)
            x[:, self.variable_channels, :, :] = inp
            if self.n_static_channels != 0:
                x[:, self.static_channels, :, :] = static_channels.repeat(
                    1, self.n_variables, 1, 1)
            if self.n_encoding_channels != 0:
                x[:, self.encoding_channels, :, :] = variable_encodings.repeat(
                    x.shape[0], 1, 1, 1)

        return x
