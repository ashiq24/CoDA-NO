from functools import reduce
import logging
import operator
from typing import Optional, Tuple

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
        max_n_modes=None,
        bias=True,
        n_layers=1,
        separable=False,
        output_scaling_factor=None,
        rank=0.5,
        factorization="dense",
        implementation="reconstructed",
        fno_block_precision="full",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs=None,
        init_std="auto",
        fft_norm="forward",
        transform_type="sht",
        sht_nlat=180,
        sht_nlon=360,
        sht_grid="legendre-gauss",
        isht_grid="legendre-gauss",
        sht_norm="backward",
        frequency_mixer=False,
        verbose=True,
        logger=None,
    ):
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.verbose = verbose

        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__(
            in_channels,
            out_channels,
            n_modes,
            max_n_modes,
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

        if frequency_mixer:
            # Initializing weights for frequency mixing:
            upper_modes = self.mk_slices()[0]
            modes = tuple([s.stop for s in upper_modes[-2:]])
            weights_shape = modes + modes  # concat
            self.W1 = nn.Parameter(torch.empty(weights_shape, dtype=torch.cfloat))
            self.W2 = nn.Parameter(torch.empty(weights_shape, dtype=torch.cfloat))
            self.reset_parameter()

        self.sht_grid = sht_grid
        self.isht_grid = isht_grid
        self.sht_norm = sht_norm
        self.transform_type = transform_type
        self.frequency_mixer = frequency_mixer

        ####
        # This following line might have a version dependent nature
        # only using for SHT
        ####
        if self.output_scaling_factor is not None:
            out_nlat = round(sht_nlat * self.output_scaling_factor[0][0])
            out_nlon = round(sht_nlon * self.output_scaling_factor[0][1])
        else:
            out_nlat = sht_nlat
            out_nlon = sht_nlon

        if self.transform_type == "sht":
            self.forward_sht = th.RealSHT(
                sht_nlat, sht_nlon, grid=self.sht_grid, norm=self.sht_norm,
            )
            self.inverse_sht = th.InverseRealSHT(
                out_nlat, out_nlon, grid=self.isht_grid, norm=self.sht_norm,
            )
        else:
            self.forward_sht = None
            self.inverse_sht = None

    def reset_parameter(self):
        """
            Reset the model parameters.

            This method initializes the model parameters `W1` and `W2` with random values
            drawn from a normal distribution with mean 0 and standard deviation calculated
            based on the scaling factor.

            Parameters:
            None

            Returns:
            None
            """
        scaling_factor = 2 / (
            np.sqrt(self.in_channels) * self.n_modes[0] * self.n_modes[1]
        )
        torch.nn.init.normal_(self.W1, mean=0.0, std=scaling_factor)
        torch.nn.init.normal_(self.W2, mean=0.0, std=scaling_factor)

    def mk_slices(self) -> Tuple[Tuple[slice, ...], ...]:
        """Returns slices for upper and lower frequency modes.

            Returns:
                A tuple containing two tuples of slices. The first tuple represents
                the slices for the upper frequency modes, and the second tuple represents
                the slices for the lower frequency modes.
            """

        m1, m2 = self.n_modes[:2]
        m1 = m1 // 2
        upper_modes = tuple(
            [slice(None), slice(None), slice(None, m1), slice(None, m2),]
        )
        """Slice for upper frequency modes.

            Equivalent to: ``x[:, :, :self.n_modes[0], :self.n_modes[1]//2]``
            """

        lower_modes = tuple(
            [slice(None), slice(None), slice(-m1, None), slice(None, m2),]
        )
        """Slice for lower frequency modes.

            Equivalent to: ``x[:, :, -self.n_modes[0]//2:, :self.n_modes[1]]``
            """

        return upper_modes, lower_modes

    @staticmethod
    def mode_mixer(x, weights):
        """
        Mixes the modes of the input tensor using the given weights.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_modes, input_size, input_size).
            weights (torch.Tensor): Weight tensor of shape (input_modes, output_modes, input_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_modes, input_size, input_size).
        """
        return torch.einsum("bimn,mnop->biop", x, weights)

    def forward_transform(self, x):
        """
        Applies the forward transform to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Transformed tensor after applying the forward transform.

        Raises:
            ValueError: If `transform_type` is not one of "fft" or "sht".
        """
        height, width = x.shape[-2:]
        if self.transform_type == "fft":
            return torch.fft.rfft2(x.float(), norm=self.fft_norm)

        if self.transform_type == "sht":
            # The SHT is expensive to initialize, and during training we expect
            # the data to all be of the same shape. If we have a correct SHT,
            # let's use it:
            if self.forward_sht.nlat == height and self.forward_sht.nlon == width:
                return self.forward_sht(x.double()).to(dtype=torch.cfloat)

            # Otherwise, initialize a new SHT:
            self.forward_sht = th.RealSHT(
                height, width, grid=self.sht_grid, norm=self.sht_norm,
            ).to(x.device)
            return self.forward_sht(x.double()).to(dtype=torch.cfloat)

        raise ValueError(
            'Expected `transform_type` to be one of "fft" or "sht"; '
            f"Got {self.transform_type=}"
        )

    def inverse_transform(
        self,
        x: torch.Tensor,
        target_height: int,
        target_width: int,
        device: Optional[torch.device] = None,
    ):
        """
        Applies the inverse transform to the input tensor `x` and returns the transformed tensor.

        Args:
            x (torch.Tensor): The input tensor to be transformed.
            target_height (int): The desired height of the transformed tensor.
            target_width (int): The desired width of the transformed tensor.
            device (Optional[torch.device]): The device on which the transformed tensor should be placed.
                If None, the device of the input tensor `x` will be used.

        Returns:
            torch.Tensor: The transformed tensor.

        Raises:
            ValueError: If `transform_type` is not one of "fft" or "sht".

        """
        source_height, source_width = x.shape[-2:]
        if device is None:
            device = x.device

        if self.transform_type == "fft":
            return torch.fft.irfft2(
                x, s=(target_height, target_width), dim=(-2, -1), norm=self.fft_norm,
            )

        if self.transform_type == "sht":
            # The SHT is expensive to initialize, and during training we expect
            # the data to all be of the same shape. If we have a correct SHT,
            # let's use it:
            if (
                self.inverse_sht.lmax == source_height
                and self.inverse_sht.mmax == source_width
                and self.inverse_sht.nlat == target_height
                and self.inverse_sht.nlon == target_width
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
            f"Got {self.transform_type=}"
        )

    def forward(self, x, indices=0, output_shape=None):
        """
        Forward pass of the FINO layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            indices (int): Index of the weight tensor to use.
            output_shape (tuple): Desired output shape (height, width) of the tensor.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channels, height, width).
        """
        batch_size, channels, height, width = x.shape

        x = self.forward_transform(x)

        upper_modes, lower_modes = self.mk_slices()

        # Frequency mixing uses separate MLPs to mix modes along each co-dim/channels:
        if self.frequency_mixer:
            x[upper_modes] = self.mode_mixer(x[upper_modes].clone(), self.W1)
            x[lower_modes] = self.mode_mixer(x[lower_modes].clone(), self.W2)

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
            self._get_weight(indices)[upper_modes],
            separable=self.separable,
        )
        # Lower block (truncate low frequencies):
        out_fft[lower_modes] = self._contract(
            x[lower_modes],
            self._get_weight(indices)[lower_modes],
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


class SpectralConvolutionKernel3D(SpectralConv):
    """
    Parameters
    ---
    transform_type : {"fft"}
        * If "fft" it uses the Fast Fourier Transform.
        * Type "sht" is not well-defined on a 3D domain.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        max_n_modes=None,
        bias=True,
        n_layers=1,
        separable=False,
        output_scaling_factor=None,
        rank=0.5,
        factorization="dense",
        implementation="reconstructed",
        fno_block_precision="full",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs=None,
        init_std="auto",
        fft_norm="forward",
        transform_type="fft",
        frequency_mixer=False,
        verbose=False,
        logger=None,
    ):
        if logger is None:
            logger = logging.getLogger()
        self.logger = logger
        self.verbose = verbose

        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        super().__init__(
            in_channels,
            out_channels,
            n_modes,
            max_n_modes,
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
        if self.verbose:
            self.logger.debug(f"{out_channels=}")

        # Note that superclass ``SpectralConv`` defines a Parameter attribute
        # ``SpectralConv.weight`` which is distinct from the Parameters
        # ``SpectralConvolutionKernel3D.mixing_weights`` below.
        if frequency_mixer:
            # Initializing weights for frequency mixing:
            first_slice = self.mk_slices()[0]
            modes = tuple([s.stop for s in first_slice[-3:]])
            if self.verbose:
                self.logger.debug(f"{frequency_mixer=}\n{modes=}")

            s = np.sqrt(self.in_channels) * reduce(operator.mul, modes)
            scaling_factor = 1 / s
            weights_shape = modes + modes  # concat
            self.mixing_weights = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.normal(
                            mean=0.0,
                            std=scaling_factor,
                            size=weights_shape,
                            dtype=torch.cfloat,
                        )
                    )
                    for _ in range(4)
                ]
            )

        self.transform_type = transform_type
        self.frequency_mixer = frequency_mixer

    def mk_slices(self) -> Tuple[Tuple[slice, ...], ...]:
        """
        Returns ``slice`` indexes to encompass each lo/hi frequency combination.

        For an N-dimensional Fourier transform (here N=3), we are interested in
        ``m`` modes the first N-1 transformed dimensions and only the real (i.e.
        first) ``m/2`` modes in the last dimension. Therefore, we generate
        the Cartesian product of slice indices:

        {`0:m1`, `-m1:`} x {`0:m2`, `-m2:`}

        Recall that only the last N dimensions of the input tensor have been
        transformed. We thus take all of the first `D-N` dimensions, as noted by
        `slice(None)`. In this case, we expect the first 2 dimensions to
        correspond to batches and augmented_channels, respectively.

        As an example, the last element of ``slices`` would be used equivalently:
        ```python
        x[((slice(None), slice(None), slice(-m1, None), slice(-m2, None), slice(m3))]
            ==
        x[:, :, -m1:, -m2:, m3:]
        ```
        """
        # Slice using half modes. Note that the last dimension (i.e. m3)
        # has already been truncated by the superclass.
        m1, m2, m3 = self.n_modes[:3]
        m1 = m1 // 2
        m2 = m2 // 2
        return tuple(
            [
                (slice(None), slice(None), s1, s2, slice(None, m3))
                for s1 in [slice(None, m1), slice(-m1, None)]
                for s2 in [slice(None, m2), slice(-m2, None)]
            ]
        )

    @staticmethod
    def mode_mixer(x, weights):
        return torch.einsum("bimno,mnopqr->bipqr", x, weights)

    def forward_transform(self, x):
        """
        Applies the forward transform to the input tensor.

        Args:
            x (torch.Tensor): The input tensor to be transformed.

        Returns:
            torch.Tensor: The transformed tensor.

        Raises:
            ValueError: If the `transform_type` is not "fft".
        """
        if self.transform_type == "fft":
            return torch.fft.rfftn(x.float(), norm=self.fft_norm, dim=(-3, -2, -1))

        raise ValueError(
            f'Expected `transform_type` to be "fft"; Got {self.transform_type=}'
        )

    def inverse_transform(
        self, x: torch.Tensor, duration: int, height: int, width: int,
    ):
        """
        Applies the inverse transform to the input tensor `x`.

        Args:
            x (torch.Tensor): The input tensor to be transformed.
            duration (int): The duration of the transformed tensor.
            height (int): The height of the transformed tensor.
            width (int): The width of the transformed tensor.

        Returns:
            torch.Tensor: The inverse transformed tensor.

        Raises:
            ValueError: If `transform_type` is not "fft".
        """
        if self.transform_type == "fft":
            return torch.fft.irfftn(
                x, s=(duration, height, width), dim=(-3, -2, -1), norm=self.fft_norm,
            )

        raise ValueError(
            f'Expected `transform_type` to be "fft"; Got {self.transform_type=}'
        )

    def forward(self, x, indices=0, output_shape=None):
        """
        Forward pass of the FINO layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, augmented_channels, duration, height, width).
            indices (int): Index of the weight to use for the forward pass. Default is 0.
            output_shape (tuple): Desired output shape of the tensor. If provided, the output tensor will be resized to this shape. Default is None.

        Returns:
            torch.Tensor: Output tensor of the forward pass.

        """
        # In `TNOBlock.compute_attention()` different variables in the same
        # instance as separate batches. That is:
        # batch_size = batch_size_in * n_variables_in
        batch_size, _augmented_channels, duration, height, width = x.shape

        x = self.forward_transform(x)

        # Slice using half modes:
        slices = self.mk_slices()

        # Frequency mixing uses separate MLPs to mix modes
        # along each co-dim/augmented_channel(s):
        if self.frequency_mixer:
            for weights, _slice in zip(self.mixing_weights, slices):
                if self.verbose:
                    self.logger.debug(f"{_slice=}")
                x[_slice] = self.mode_mixer(x[_slice].clone(), weights)
                if self.verbose:
                    self.logger.debug(f"{x[_slice].shape=}")

        # Spectral conv / channel mixer
        # The output will be of size:
        # (batch_size, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1)
        out_fft = torch.zeros(
            [batch_size, self.out_channels, duration, height, width // 2 + 1],
            dtype=x.dtype,
            device=x.device,
        )

        for i, _slice in enumerate(slices):
            out_fft[_slice] = self._contract(
                x[_slice], self._get_weight(indices)[_slice], separable=self.separable
            )

        if self.output_scaling_factor is not None and output_shape is None:
            duration = round(duration * self.output_scaling_factor[indices][0])
            height = round(height * self.output_scaling_factor[indices][1])
            width = round(width * self.output_scaling_factor[indices][2])

        if output_shape is not None:
            duration, height, width = output_shape[:3]

        x = self.inverse_transform(out_fft, duration, height, width)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x
