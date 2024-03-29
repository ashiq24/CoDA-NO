from functools import partial
import logging

import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from neuralop.layers.fno_block import FNOBlocks
from .fino import SpectralConvKernel2d, SpectralConvolutionKernel3D


def NO_OP(x, *_args, **_kwargs):
    return x


AffineNormalizer2D = partial(nn.InstanceNorm2d, affine=True)
AffineNormalizer3D = partial(nn.InstanceNorm3d, affine=True)


class TNOBlock(nn.Module):
    def __init__(
        self,
        n_modes,
        n_head=1,
        token_codimension=1,
        output_scaling_factor=None,
        incremental_n_modes=None,
        head_codimension=None,
        use_mlp=False,
        mlp=None,
        non_linearity=F.gelu,
        preactivation=False,
        fno_skip="linear",
        mlp_skip="soft-gating",
        mlp_expansion=1.0,
        separable=False,
        factorization="tucker",
        rank=1.0,
        SpectralConvolution=None,
        Normalizer=None,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=None,
        fft_norm="forward",
        codimension_size=None,
        per_channel_attention=True,
        permutation_eq=True,
        temperature=1.0,
        kqv_non_linear=False,
        logger=None,
        **_kwargs,
    ):
        super().__init__()

        # Co-dimension of each variable/token. The token embedding space is
        # identical to the variable space, so their dimensionalities are equal.
        self.variable_codimension = token_codimension
        self.token_codimension = token_codimension

        # Codim of attention from each head
        self.head_codimension = (
            head_codimension if head_codimension is not None else token_codimension
        )
        self.n_head = n_head  # number of heads
        self.output_scaling_factor = output_scaling_factor
        self.temperature = temperature

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

        # attention per channel not per variables
        # making last mixer permutation equivariant
        # K,Q,V operator with or without non_lin

        if kqv_non_linear:
            kqv_activation = non_linearity
        else:
            kqv_activation = NO_OP

        self.permutation_eq = permutation_eq

        if self.n_head is not None:
            # recalculating the value of `head_codim`
            self.head_codimension = max(token_codimension // self.n_head, 1)

        self.codimension_size = codimension_size
        self.mixer_token_codimension = token_codimension

        if per_channel_attention:
            # for per channel attention, forcing the values of token dims
            self.token_codimension = 1
            self.head_codimension = 1

        # this scale used for downsampling Q,K functions
        scale = 2 if per_channel_attention else 1
        scale = min(self.n_head, scale)

        kqv_modes = [i // scale for i in n_modes]

        self.logger.debug(
            f"\n {rank=}"
            f"\n {factorization=}"
            f"\n {self.head_codimension=}"
            f"\n {scale=}"
            f"\n {kqv_modes=}"
        )

        if not per_channel_attention:
            self.logger.debug(
                f"\n {self.token_codimension=}"
                f"\n {self.n_head=}"
                f"\n {self.head_codimension=}"
            )

        if decomposition_kwargs is None:
            decomposition_kwargs = {}
        common_args = dict(
            use_mlp=use_mlp,
            mlp=mlp,
            preactivation=preactivation,
            mlp_skip=mlp_skip,
            mlp_dropout=0,
            incremental_n_modes=incremental_n_modes,
            rank=rank,
            fft_norm=fft_norm,
            mlp_expansion=mlp_expansion,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
        )

        kqv_args = dict(
            in_channels=self.token_codimension,
            out_channels=self.n_head * self.head_codimension,
            n_modes=kqv_modes,
            # args below are shared with Projection block
            non_linearity=kqv_activation,
            fno_skip="linear",
            norm=None,
            apply_skip=True,
            n_layers=1,
        )
        self.K = FNOBlocks(
            output_scaling_factor=1 / scale,
            SpectralConv=partial(
                SpectralConvolution,
                rank=0.5,
                factorization=None,
                logger=self.logger.getChild("K.convolution"),
            ),
            **kqv_args,
            **common_args,
        )
        self.Q = FNOBlocks(
            output_scaling_factor=1 / scale,
            SpectralConv=partial(
                SpectralConvolution,
                rank=0.5,
                factorization=None,
                logger=self.logger.getChild("Q.convolution"),
            ),
            **kqv_args,
            **common_args,
        )
        self.V = FNOBlocks(
            output_scaling_factor=1,
            SpectralConv=partial(
                SpectralConvolution,
                rank=0.5,
                factorization=None,
                logger=self.logger.getChild("V.convolution"),
            ),
            **kqv_args,
            **common_args,
        )

        if self.n_head * self.head_codimension != self.token_codimension:
            self.proj = FNOBlocks(
                in_channels=self.n_head * self.head_codimension,
                out_channels=self.token_codimension,
                n_modes=n_modes,
                output_scaling_factor=1,
                # args below are shared with KQV blocks
                apply_skip=True,
                non_linearity=kqv_activation,
                fno_skip="linear",
                norm=None,
                SpectralConv=partial(
                    SpectralConvolution, rank=0.5, factorization=None,
                ),
                n_layers=1,
                **common_args,
            )
        else:
            self.proj = None

        self.attention_normalizer = Normalizer(self.token_codimension)
        """
        NOTE: by default, ``self.attention_normalizer`` has parameters of 
        ``dtype=float32`` and thus expects inputs to be of the same type.
        """

        mixer_args = dict(
            n_modes=n_modes,
            output_scaling_factor=1,
            non_linearity=non_linearity,
            norm="instance_norm",
            fno_skip=fno_skip,
            SpectralConv=partial(
                SpectralConvolution, rank=0.5, factorization=None, bias=True,
            ),
        )
        # We have an option to make the last operator (MLP in regular
        # Transformer block) permutation equivariant. i.e., applying the
        # operator per variable or applying the operator on the whole channel
        # (like regular FNO).
        if permutation_eq:
            logger.debug(f"\n {permutation_eq=}" f"\n {self.mixer_token_codimension=}")
            self.mixer = FNOBlocks(
                in_channels=self.mixer_token_codimension,
                out_channels=self.mixer_token_codimension,
                apply_skip=True,
                n_layers=2,
                **mixer_args,
                **common_args,
            )
            self.norm1 = Normalizer(self.token_codimension)
            """
            NOTE: by default, ``self.norm1`` has parameters of ``dtype=float32``
            and thus expects inputs to be of the same type.
            """
            self.norm2 = Normalizer(self.mixer_token_codimension)
            """
            NOTE: by default, ``self.norm2`` has parameters of ``dtype=float32``
            and thus expects inputs to be of the same type. 
            """
            self.mixer_out_normalizer = Normalizer(self.mixer_token_codimension)
            """
            NOTE: by default, ``self.mixer_token_codimension`` has parameters of 
            ``dtype=float32`` and thus expects inputs to be of the same type.
            """

        else:
            self.mixer = FNOBlocks(
                in_channels=codimension_size,
                out_channels=codimension_size,
                n_layers=2,
                **mixer_args,
                **common_args,
            )
            self.norm1 = Normalizer(codimension_size)
            self.norm2 = Normalizer(codimension_size)
            self.mixer_out_normalizer = Normalizer(codimension_size)

    def forward(self, *args):
        raise NotImplementedError(
            "Use a proper subclass of TNOBlock (i.e. TnoBlock2d or TNOBlock3D)."
        )


class TnoBlock2d(TNOBlock):
    def __init__(self, *args, **kwargs):
        Normalizer = kwargs.get("Normalizer")
        if Normalizer is None:
            Normalizer = AffineNormalizer2D
        kwargs["Normalizer"] = Normalizer

        Convolution = kwargs.get("SpectralConvolution")
        if Convolution is None:
            Convolution = SpectralConvKernel2d
        kwargs["SpectralConvolution"] = Convolution

        super().__init__(*args, **kwargs)

    def compute_attention(self, xa, batch_size):
        """Compute the key-query-value variant of the attention matrix.

        Args:
            xa (torch.Tensor): The input tensor, assumed to be normalized.
            batch_size (int): The batch size of the input tensor.

        Returns:
            torch.Tensor: The computed attention matrix.

        Notes:
            This method assumes that the input tensor `xa` has been normalized.

        """
        # `xa` was rearranged like:
        # >>> xa = rearrange(x, 'b (t d) h w -> (b t) d h w',
        # >>>                d=self.token_codimension)
        k = self.K(xa)
        q = self.Q(xa)
        v = self.V(xa)

        value_x, value_y = v.shape[-2], v.shape[-1]

        rearrangement = dict(
            pattern="(b t) (a d) h w -> b a t (d h w)", b=batch_size, a=self.n_head,
        )
        k = rearrange(k, **rearrangement)
        q = rearrange(q, **rearrangement)
        v = rearrange(v, **rearrangement)

        dprod = torch.matmul(q, k.transpose(-1, -2)) / (
            np.sqrt(k.shape[-1]) * self.temperature
        )
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)
        attention = rearrange(
            attention,
            "b a t (d h w) -> b t a d h w",
            d=self.head_codimension,
            h=value_x,
            w=value_y,
        )
        attention = rearrange(attention, "b t a d h w -> (b t) (a d) h w")
        return attention

    def forward(self, x, output_shape=None):
        if self.permutation_eq:
            return self._forward_equivariant(x)
        else:
            return self._forward_non_equivariant(x)

    def _forward_equivariant(self, x):
        """
        Forward pass for the equivariant attention layer.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, channels, height, width).

        Returns:
            Output tensor of shape (batch_size, sequence_length, channels, height, width).
        """
        batch_size = x.shape[0]
        output_shape = x.shape[-2:]

        assert x.shape[1] % self.token_codimension == 0

        xa = rearrange(x, "b (t d) h w -> (b t) d h w", d=self.token_codimension)
        xa_norm = self.norm1(xa)

        attention = self.compute_attention(xa_norm, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        attention = self.attention_normalizer(attention + xa)
        attention = rearrange(attention, "(b t) d h w -> b (t d) h w", b=batch_size)
        attention = rearrange(
            attention, "b (t d) h w -> (b t) d h w", d=self.mixer_token_codimension
        )

        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        output = self.mixer_out_normalizer(output) + attention
        output = rearrange(output, "(b t) d h w -> b (t d) h w", b=batch_size)

        return output

    def _forward_non_equivariant(self, x):
        """
        Forward pass for the non-equivariant attention layer.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            output: Output tensor after applying the attention mechanism and mixer.
                    Shape is (batch_size, channels, height, width).
        """
        batch_size = x.shape[0]
        output_shape = x.shape[-2:]

        assert x.shape[1] % self.token_codimension == 0

        x_norm = self.norm1(x)
        xa = rearrange(x_norm, "b (t d) h w -> (b t) d h w", d=self.token_codimension)

        attention = self.compute_attention(xa, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        attention = rearrange(attention, "(b t) d h w -> b (t d) h w", b=batch_size)
        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        return output


class TNOBlock3D(TNOBlock):
    """
    3D version of the TNOBlock class.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        Normalizer: The normalizer class used for normalization.
        Convolution: The convolution class used for spectral convolution.

    """

    def __init__(self, *args, **kwargs):
        Normalizer = kwargs.get("Normalizer")
        if Normalizer is None:
            Normalizer = AffineNormalizer3D
        kwargs["Normalizer"] = Normalizer

        Convolution = kwargs.get("SpectralConvolution")
        if Convolution is None:
            Convolution = SpectralConvolutionKernel3D
        kwargs["SpectralConvolution"] = Convolution

        super().__init__(*args, **kwargs)

    def compute_attention(self, xa, batch_size):
        """Compute the key-query-value variant of the attention matrix.

        Args:
            xa (torch.Tensor): The input tensor, assumed to be normalized.
            batch_size (int): The batch size.

        Returns:
            torch.Tensor: The computed attention matrix.

        Notes:
            This method assumes that the input tensor `xa` has been normalized.

        """
        # `xa` was rearranged like:
        # rearrange(x, 'b (k d) t h w -> (b k) d t h w', d=self.token_codimension)
        k = self.K(xa)
        q = self.Q(xa)
        v = self.V(xa)

        v_duration, v_height, v_width = v.shape[-3:]

        rearrangement = dict(
            # index `k` counts the number of variables.
            # index `d` becomes the variable embedding dimensionality per head.
            pattern="(b k) (a d) t h w -> b a k (d t h w)",
            b=batch_size,
            a=self.n_head,
        )
        k = rearrange(k, **rearrangement)
        q = rearrange(q, **rearrangement)
        v = rearrange(v, **rearrangement)

        dprod = torch.matmul(q, k.transpose(-1, -2)) / (
            self.temperature * np.sqrt(k.shape[-1])
        )
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)
        attention = rearrange(
            attention,
            "b a k (d t h w) -> b k a d t h w",
            d=self.head_codimension,
            t=v_duration,
            h=v_height,
            w=v_width,
        )
        attention = rearrange(attention, "b k a d t h w -> (b k) (a d) t h w")
        return attention

    def forward(self, x, output_shape=None):
        if self.permutation_eq:
            return self._forward_equivariant(x)
        else:
            return self._forward_non_equivariant(x)

    def _forward_equivariant(self, x):
        """
        Forward pass for the equivariant attention layer.

        Args:
            x: Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            Output tensor of shape (batch_size, channels, time, height, width).
        """
        batch_size = x.shape[0]
        output_shape = x.shape[-3:]

        assert x.shape[1] % self.token_codimension == 0

        xa = rearrange(x, "b (k d) t h w -> (b k) d t h w", d=self.token_codimension)
        xa_norm = self.norm1(xa)

        attention = self.compute_attention(xa_norm, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        attention = self.attention_normalizer(attention + xa)
        attention = rearrange(
            attention, "(b k) d t h w -> b (k d) t h w", b=batch_size,
        )
        attention = rearrange(
            attention, "b (k d) t h w -> (b k) d t h w", d=self.mixer_token_codimension
        )

        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        output = self.mixer_out_normalizer(output) + attention
        output = rearrange(output, "(b k) d t h w -> b (k d) t h w", b=batch_size)

        return output

    def _forward_non_equivariant(self, x):
        """
        Forward pass for the non-equivariant attention layer.

        Args:
            x: Input tensor of shape (batch_size, channels, time, height, width).

        Returns:
            output: Output tensor after applying the attention layer, with shape (batch_size, channels, time, height, width).
        """
        batch_size = x.shape[0]
        output_shape = x.shape[-3:]

        assert x.shape[1] % self.token_codimension == 0

        x_norm = self.norm1(x)
        xa = rearrange(
            x_norm, "b (k d) t h w -> (b k) d t h w", d=self.token_codimension
        )

        attention = self.compute_attention(xa, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        attention = rearrange(
            attention, "(b k) d t h w -> b (k d) t h w", b=batch_size,
        )
        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        return output
