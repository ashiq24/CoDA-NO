from functools import reduce
from typing import Tuple

import numpy as np
import torch
from torch import nn
import torch_harmonics as th


class VariableEncoding2d(nn.Module):
    def __init__(self, channel, mode_x, mode_y, basis='fft') -> None:
        super().__init__()
        self.mode_x = mode_x
        self.mode_y = mode_y
        self.coefficients_r = nn.Parameter(torch.empty(channel, mode_x, mode_y))
        self.coefficients_i = nn.Parameter(torch.empty(channel, mode_x, mode_y))
        self.reset_parameters()
        self.basis = basis
        if basis == 'fft':
            self.transform = torch.fft.ifft2
        elif basis == 'sht':
            self.transform = th.InverseRealSHT(
                mode_x,
                mode_y,
                lmax=mode_x,
                mmax=mode_y,
                grid="legendre-gauss",
                norm="backward",
            )

    def reset_parameters(self):
        std = (1 / (self.mode_x * self.mode_y))**0.5
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
                lmax=self.mode_x,
                mmax=self.mode_y,
                grid="legendre-gauss",
                norm='backward'
            ).to(
                device=self.coefficients_i.device,
                dtype=self.coefficients_i.dtype
            )
            return self.transform

        else:
            return self.transform(
                self.coefficients_r + 1.0j *self.coefficients_i,
                s=(size_x, size_y)
            ).real


# SHT doesn't make sense for 3 variables
class FourierVariableEncoding3D(nn.Module):
    def __init__(self, channel_size: int, modes: Tuple[int, ...]) -> None:
        super().__init__()
        if len(modes) != 3:
            raise ValueError(
                f"Expected 3 frequency modes, but got {len(modes)}\n{modes=}")

        self.modes = modes
        self.weights_re = nn.Parameter(torch.empty(channel_size, *modes))
        self.weights_im = nn.Parameter(torch.empty(channel_size, *modes))
        self.reset_parameters()
        self.transform = torch.fft.ifftn

    def reset_parameters(self):
        std = 1 / np.sqrt(reduce(lambda a, b: a * b, self.modes))
        torch.nn.init.normal_(self.weights_re, mean=0.0, std=std)
        torch.nn.init.normal_(self.weights_im, mean=0.0, std=std)

    def forward(self, x):
        """Take a resolution and outputs the positional encodings"""
        size_t, size_x, size_y = x.shape[-3], x.shape[-2], x.shape[-1]
        return torch.fft.ifftn(
            self.weights_re + 1.0j * self.weights_im,
            s=(size_t, size_x, size_y),
            norm="backward",  # normalize by 1/n
        ).real
