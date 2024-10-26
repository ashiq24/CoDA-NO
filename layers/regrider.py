import torch_harmonics as th
import torch.nn as nn


class Regird(nn.Module):
    def __init__(
            self,
            input_grid,
            output_grid,
            sht_nlat=128,
            sht_nlon=256,
            output_scaling_factor=None):
        super().__init__()
        self.input_transform = th.RealSHT(
            sht_nlat, sht_nlon, grid=input_grid, norm='backward').float()
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.output_transform = th.InverseRealSHT(
            sht_nlat, sht_nlon, grid=output_grid, norm='backward').float()

    def forward(self, x):
        if self.input_transform.nlat != x.shape[-2] or self.input_transform.nlon != x.shape[-1]:
            self.input_transform = th.RealSHT(x.shape[-2], x.shape[-1], grid=self.input_grid,
                                              norm='backward').to(x.device, dtype=x.dtype)
            self.output_transform = th.InverseRealSHT(x.shape[-2], x.shape[-1], grid=self.output_grid,
                                                      norm='backward').to(x.device, dtype=x.dtype)

        return self.output_transform(self.input_transform(x))
