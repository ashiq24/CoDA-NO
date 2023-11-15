import torch.nn as nn
import torch.nn.functional as F
import torch
import torch_harmonics as th


class VaribaleEncoding2d(nn.Module):
    def __init__(self, channel, mode_x, mode_y, basis='fft') -> None:
        super().__init__()
        self.mode_x = mode_x
        self.mode_y = mode_y
        self.coefficients_r = nn.Parameter(
            torch.empty(channel, mode_x, mode_y))
        self.coefficients_i = nn.Parameter(
            torch.empty(channel, mode_x, mode_y))
        self.reset_parameter()
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
                norm="backward")

    def reset_parameter(self):
        torch.nn.init.normal_(self.coefficients_r, mean=0.0, std=(
            1 / (self.mode_x * self.mode_y))**0.5)
        torch.nn.init.normal_(self.coefficients_i, mean=0.0, std=(
            1 / (self.mode_x * self.mode_y))**0.5)

    def forward(self, x):
        '''
        it take a resolution and outputs the positional encodings
        '''
        size_x, size_y = x.shape[-2], x.shape[-1]
        if self.basis == 'sht' and (
                self.transform.nlat != size_x or self.transform.nlon != size_y):
            self.transform = th.InverseRealSHT(
                size_x,
                size_y,
                lmax=self.mode_x,
                mmax=self.mode_y,
                grid="legendre-gauss",
                norm='backward').to(
                self.coefficients_i.device,
                dtype=self.coefficients_i.dtype)
        if self.basis == 'sht':
            return self.transform(
                self.coefficients_r + 1.0j * self.coefficients_i)
        else:
            return self.transform(
                self.coefficients_r + 1.0j * self.coefficients_i, s=(size_x, size_y)).real
