from functools import reduce
import logging

import numpy as np
import torch
from torch import nn
import torch_harmonics as th
from neuralop.layers.spectral_convolution import SpectralConv


class SpectralConvKernel2d(SpectralConv):
    """
    Parameters
    ---
    transform_type : {'sht', 'fft'}
        * If "sht" it uses the Spherical Fourier Transform.
        * If "fft" it uses the Fast Fourier Transform.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        incremental_n_modes=None,
        bias=True,
        n_layers=1,
        separable=False,
        output_scaling_factor=None,
        rank=0.5,
        factorization='dense',
        implementation='reconstructed',
        fno_block_precision='full',
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs=None,
        init_std='auto',
        fft_norm='forward',
        transform_type="sht",
        sht_nlat=180,
        sht_nlon=360,
        sht_grid="legendre-gauss",
        isht_grid="legendre-gauss",
        sht_norm="backward",
        frequency_mixer=False,
        verbose=True,
        logger=None
    ):
        self.verbose = verbose

        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__(
            in_channels,
            out_channels,
            n_modes,
            incremental_n_modes,
            bias=bias,
            n_layers=n_layers,
            separable=separable,
            output_scaling_factor=output_scaling_factor,
            fno_block_precision=fno_block_precision,
            rank=rank,
            factorization=factorization,
            implementation=implementation,
            fixed_rank_modes=fixed_rank_modes,
            joint_factorization=joint_factorization,
            decomposition_kwargs=decomposition_kwargs,
            init_std=init_std,
            fft_norm=fft_norm,
        )

        # self.shared = shared

        # readjusting initialization
        if init_std == "auto":
            init_std = 1 / np.sqrt(in_channels * n_modes[-1] * n_modes[-2])
        else:
            init_std = init_std

        for w in self.weight:
            w.normal_(0, init_std)

        # weights for frequency mixers

        hm1, hm2 = self.half_n_modes[0], self.half_n_modes[1]
        if frequency_mixer:
            # if frequency mixer is true
            # then initializing weights for frequncy mixing
            # otherwise it is just a regular FNO or SFNO layer
            print("using Mixer")
            self.W1r = nn.Parameter(
                torch.empty(
                    hm1,
                    hm2,
                    hm1,
                    hm2,
                    dtype=torch.float))
            self.W2r = nn.Parameter(
                torch.empty(
                    hm1,
                    hm2,
                    hm1,
                    hm2,
                    dtype=torch.float))
            self.W1i = nn.Parameter(
                torch.empty(
                    hm1,
                    hm2,
                    hm1,
                    hm2,
                    dtype=torch.float))
            self.W2i = nn.Parameter(
                torch.empty(
                    hm1,
                    hm2,
                    hm1,
                    hm2,
                    dtype=torch.float))
            self.reset_parameter()

        self.sht_grid = sht_grid
        self.isht_grid = isht_grid
        self.sht_norm = sht_norm
        self.transform_type = transform_type
        self.frequency_mixer = frequency_mixer


        if self.output_scaling_factor is not None:
            out_nlat = round(sht_nlat * self.output_scaling_factor[0][0])
            out_nlon = round(sht_nlon * self.output_scaling_factor[0][1])
        else:
            out_nlat = sht_nlat
            out_nlon = sht_nlon

        if self.transform_type == "sht":
            self.forward_sht = th.RealSHT(
                sht_nlat,
                sht_nlon,
                grid=self.sht_grid,
                norm=self.sht_norm,
            )
            self.inverse_sht = th.InverseRealSHT(
                out_nlat,
                out_nlon,
                grid=self.isht_grid,
                norm=self.sht_norm,
            )

    def reset_parameter(self):
        # Initial model parameters.
        scaling_factor = ((1 / self.in_channels)**0.5) / \
            (self.half_n_modes[0] * self.half_n_modes[1])
        torch.nn.init.normal_(self.W1r, mean=0.0, std=scaling_factor)
        torch.nn.init.normal_(self.W2r, mean=0.0, std=scaling_factor)
        torch.nn.init.normal_(self.W1i, mean=0.0, std=scaling_factor)
        torch.nn.init.normal_(self.W2i, mean=0.0, std=scaling_factor)

    @staticmethod
    def mode_mixer(x, weights):
        return torch.einsum("bimn,mnop->biop", x, weights)

    def forward_transform(self, x):
        height, width = x.shape[-2:]
        if self.transform_type == "fft":
            return torch.fft.rfft2(x.float(), norm=self.fft_norm)

        if self.transform_type == "sht":
            # The SHT is expensive to initialize, and during training we expect
            # the data to all be of the same shape. If we have a correct SHT,
            # let's use it:
            if (
                self.forward_sht.nlat == height and
                self.forward_sht.nlon == width
            ):
                return self.forward_sht(x.double()).to(dtype=torch.cfloat)

            # Otherwise, initialize a new SHT:
            self.forward_sht = th.RealSHT(
                height,
                width,
                grid=self.sht_grid,
                norm=self.sht_norm,
            ).to(x.device)
            return self.forward_sht(x.double()).to(dtype=torch.cfloat)

        raise ValueError(
            'Expected `transform_type` to be one of "fft" or "sht"; '
            f'Got {self.transform_type=}'
        )

    # Although a previous implementation kept an initialized
    # ``th.InverseRealSHT`` in its state, it always checked if its lat/lon grid
    # size matched the input's
    # resolution. Thus, it never really mattered that an object was in state.
    def inverse_transform(
        self,
        x: torch.Tensor,
        target_height: int,
        target_width: int,
        device,
    ):
        source_height, source_width = x.shape[-2:]
        if self.transform_type == "fft":
            return torch.fft.irfft2(
                x,
                s=(target_height, target_width),
                dim=(-2, -1),
                norm=self.fft_norm,
            )

        if self.transform_type == "sht":
            # The SHT is expensive to initialize, and during training we expect
            # the data to all be of the same shape. If we have a correct SHT,
            # let's use it:
            if (
                self.inverse_sht.lmax == source_height and
                self.inverse_sht.mmax == source_width and
                self.inverse_sht.nlat == target_height and
                self.inverse_sht.nlon == target_width
            ):
                return self.inverse_sht(x.to(dtype=torch.cdouble)).float()

            # Otherwise, initialize a new SHT:
            self.inverse_sht = th.InverseRealSHT(
                target_height,
                target_width,
                lmax=source_height,
                mmax=source_width,
                grid=self.sht_grid,
                norm=self.sht_norm,
            ).to(device)
            return self.inverse_sht(x.to(dtype=torch.cdouble)).float()

        raise ValueError(
            'Expected `transform_type` to be one of "fft" or "sht"; '
            f'Got {self.transform_type=}'
        )

    def forward(self, x, indices=0, output_shape=None):
        batch_size, channels, height, width = x.shape

        x = self.forward_transform(x)

        upper_modes = [
            slice(None),
            slice(None),
            slice(None, self.half_n_modes[0]),
            slice(None, self.half_n_modes[1]),
        ]
        """Slice for upper frequency modes.

        Equivalent to: ``x[:, :, :self.half_n_modes[0], :self.half_n_modes[1]]``
        """

        lower_modes = [
            slice(None),
            slice(None),
            slice(-self.half_n_modes[0], None),
            slice(None, self.half_n_modes[1]),
        ]
        """Slice for lower frequency modes.

        Equivalent to: ``x[:, :, -self.half_n_modes[0]:, :self.half_n_modes[1]]``
        """

        # mode mixer
        # uses separate MLP to mix mode along each co-dim/channels
        if self.frequency_mixer:
            W1 = self.W1r + 1.0j * self.W1i
            W2 = self.W2r + 1.0j * self.W2i

            x[upper_modes] = self.mode_mixer(x[upper_modes].clone(), W1)
            x[lower_modes] = self.mode_mixer(x[lower_modes].clone(), W2)

        # spectral conv / channel mixer

        # The output will be of size:
        # (batch_size, self.out_channels, x.size(-2), x.size(-1)//2 + 1)
        out_fft = torch.zeros(
            [batch_size, self.out_channels, height, width // 2 + 1],
            dtype=x.dtype,
            device=x.device,
        )

        # Upper block (truncate high frequencies):
        out_fft[upper_modes] = self._contract(
            x[upper_modes],
            self._get_weight(2 * indices),
            separable=self.separable,
        )
        # Lower block (truncate low frequencies):
        out_fft[lower_modes] = self._contract(
            x[lower_modes],
            self._get_weight(2 * indices + 1),
            separable=self.separable,
        )

        if self.output_scaling_factor is not None and output_shape is None:
            height = round(height * self.output_scaling_factor[indices][0])
            width = round(width * self.output_scaling_factor[indices][1])

        if output_shape is not None:
            height = output_shape[0]
            width = output_shape[1]

        x = self.inverse_transform(out_fft, height, width, x.device)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x
