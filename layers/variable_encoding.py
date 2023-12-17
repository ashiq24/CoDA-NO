from functools import reduce
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch_harmonics as th
from neuralop.layers.embeddings import PositionalEmbedding


class VariableEncoding2d(nn.Module):
    def __init__(self, channel, modes: Tuple[int, ...], basis='fft') -> None:
        super().__init__()
        self.modes = modes
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
                return self.transform(self.coefficients_r + 1.0j * self.coefficients_i)

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
            return self.transform

        else:
            return self.transform(
                self.coefficients_r + 1.0j * self.coefficients_i,
                s=(size_x, size_y)
            ).real


# SHT doesn't make sense for 3 variables
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
        *_, size_t, size_x, size_y = x_shape
        return torch.fft.ifftn(
            self.weights_re + 1.0j * self.weights_im,
            s=(size_t, size_x, size_y),
            norm="forward",  # don't multiply by any normalization factor
        ).real

class EncodingWrapper(nn.Module):
    def __init__(
        self,
        n_features: int,
        modes: Tuple[int, ...],
        basis: str = 'fft',
        mesh: str = 'uniform',
        n_dim: int = 2,
        positional_encoding_dim: int: 8) -> None:
        super().__init__()
        self.n_features = n_features
        self.modes = modes
        self.basis = basis
        self.mesh = mesh
        self.n_dim = n_dim
        self.positional_encoding_dim = positional_encoding_dim
        if mesh == 'uniform':
            if len(modes) == 2:
                self.encoding = VariableEncoding2d(n_features, modes, basis=basis)
            elif len(modes) == 3:
                self.encoding = FourierVariableEncoding3D(n_features, modes)
        else:
            self.var_encoder = MLPLinear(
                [n_dim + 2 * postional_em_dim, self.var_encoding_channels * var_num])
            self.PE = PositionalEmbedding(postional_em_dim)
            self.variable_channels = [
                i * (var_encoding_channels + self.in_dim) for i in range(var_num)]
            self.encoding_channels = list(set([i for i in range(
                (var_encoding_channels + 1) * var_num)]) - set(self.variable_channels)) 

    def forward(self, x):
        return self.encoding(x)