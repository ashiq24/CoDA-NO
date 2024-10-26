from functools import partial
import logging
from typing import Optional, Callable, Union, Dict

import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from neuralop.layers.fno_block import FNOBlocks
from .fino_nd import SpectralConvKernel2d, SpectralConvKernel1d, SpectralConvKernel3d


# Implementation of generic N dimentional Codano Block
AffineNormalizer1D = partial(nn.InstanceNorm1d, affine=True)
AffineNormalizer2D = partial(nn.InstanceNorm2d, affine=True)
AffineNormalizer3D = partial(nn.InstanceNorm3d, affine=True)
# For higher demnsion (>=4), need to implement custom

def Identity(x, *args, **kwargs):
    return x


class CodanoBlock(nn.Module):
    def __init__(
        self,
        n_modes: Union[int, tuple],
        n_head: int = 1,
        token_codimension: int = 1,
        output_scaling_factor: Optional[float] = None,
        max_n_modes: Optional[Union[int, tuple]] = None,
        head_codimension: Optional[int] = None,
        use_mlp: bool = False,
        mlp: Optional[nn.Module] = None,
        non_linearity: Callable = F.gelu,
        preactivation: bool = False,
        fno_skip: str = 'linear',
        mlp_skip: str = 'linear',
        mlp_expansion: float = 1.0,
        separable: bool = False,
        factorization: str = None,
        rank: float = 1.0,
        SpectralConvolution: Optional[Callable] = None,
        Normalizer: Optional[Callable] = None,
        joint_factorization: bool = False,
        fixed_rank_modes: bool = False,
        implementation: str = 'reconstructed',
        decomposition_kwargs: Optional[Dict] = None,
        fft_norm: str = 'forward',
        codimension_size: Optional[int] = None,
        per_channel_attention: bool = True,
        permutation_eq: bool = True,
        temperature: float = 1.0,
        kqv_non_linear: bool = False,
        num_dims: int = 2,
        **_kwargs,
    ):
        super().__init__()

        self.variable_codimension = token_codimension
        self.token_codimension = token_codimension
        self.head_codimension = head_codimension or token_codimension
        self.n_head = n_head
        self.output_scaling_factor = output_scaling_factor
        self.temperature = temperature
        self.num_dims = num_dims

        if kqv_non_linear:
            kqv_activation = non_linearity
        else:
            kqv_activation = Identity

        self.permutation_eq = permutation_eq

        if self.n_head is not None:
            self.head_codimension = max(token_codimension // self.n_head, 1)

        self.codimension_size = codimension_size
        self.mixer_token_codimension = token_codimension

        if per_channel_attention:
            self.token_codimension = 1
            self.head_codimension = 1

        scale = min(self.n_head, 1 if per_channel_attention else 1)

        mixer_modes = [i // scale for i in n_modes]
        mixer_n_modes = [i // scale for i in max_n_modes]

        decomposition_kwargs = decomposition_kwargs or {}
        common_args = dict(
            use_mlp=use_mlp,
            mlp=mlp,
            preactivation=preactivation,
            mlp_skip=mlp_skip,
            mlp_dropout=0,
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
            n_modes=mixer_modes,
            max_n_modes=mixer_n_modes,
            non_linearity=kqv_activation,
            fno_skip='linear',
            norm=None,
            apply_skip=True,
            n_layers=1,
        )

        rank = 1.0
        conv_kwargs = dict(rank=rank, factorization=None)
        self.K = FNOBlocks(
            output_scaling_factor=1 / scale,
            SpectralConv=partial(
                SpectralConvolution,
                **conv_kwargs
            ),
            **kqv_args,
            **common_args,
        )
        self.Q = FNOBlocks(
            output_scaling_factor=1 / scale,
            SpectralConv=partial(
                SpectralConvolution,
                **conv_kwargs,
            ),
            **kqv_args,
            **common_args,
        )
        self.V = FNOBlocks(
            output_scaling_factor=1,
            SpectralConv=partial(
                SpectralConvolution,
                **conv_kwargs,
            ),
            **kqv_args,
            **common_args,
        )

        if self.n_head * self.head_codimension != self.token_codimension:
            self.proj = FNOBlocks(
                in_channels=self.n_head * self.head_codimension,
                out_channels=self.token_codimension,
                n_modes=n_modes,
                max_n_modes=max_n_modes,
                output_scaling_factor=1,
                apply_skip=True,
                non_linearity=Identity,
                fno_skip='linear',
                norm=None,
                SpectralConv=partial(
                    SpectralConvolution,
                    rank=1.0,
                    factorization=None,
                ),
                n_layers=1,
                **common_args,
            )
        else:
            self.proj = None

        self.attention_normalizer = Normalizer(self.token_codimension)

        mixer_args = dict(
            n_modes=n_modes,
            max_n_modes=max_n_modes,
            output_scaling_factor=1,
            non_linearity=non_linearity,
            norm='instance_norm',
            fno_skip=fno_skip,
            SpectralConv=partial(
                SpectralConvolution,
                rank=rank,
                factorization=None,
                bias=True,
            ),
        )

        if self.permutation_eq:
            self.mixer = FNOBlocks(
                in_channels=self.mixer_token_codimension,
                out_channels=self.mixer_token_codimension,
                apply_skip=True,
                n_layers=2,
                **mixer_args,
                **common_args,
            )
            self.norm1 = Normalizer(self.token_codimension)
            self.norm2 = Normalizer(self.mixer_token_codimension)
            self.mixer_out_normalizer = Normalizer(
                self.mixer_token_codimension)
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
            "Use a proper subclass of CodanoBlock (i.e. CodanoBlockND or CodanoBlock3D).")


class CodanoBlockND(CodanoBlock):
    def __init__(self, *args, **kwargs):
        if kwargs["num_dims"] == 1:
            Normalizer = kwargs.get("Normalizer", AffineNormalizer1D)
            kwargs["Normalizer"] = Normalizer

            Convolution = kwargs.get(
                "SpectralConvolution", SpectralConvKernel1d)
            kwargs["SpectralConvolution"] = Convolution
        elif kwargs["num_dims"] == 2:
            Normalizer = kwargs.get("Normalizer", AffineNormalizer2D)
            kwargs["Normalizer"] = Normalizer

            Convolution = kwargs.get(
                "SpectralConvolution", SpectralConvKernel2d)
            kwargs["SpectralConvolution"] = Convolution
        elif kwargs["num_dims"] == 3:
            Normalizer = kwargs.get("Normalizer", AffineNormalizer3D)
            kwargs["Normalizer"] = Normalizer

            Convolution = kwargs.get(
                "SpectralConvolution", SpectralConvKernel3d)
            kwargs["SpectralConvolution"] = Convolution

        super().__init__(*args, **kwargs)

    def compute_attention(self, xa, batch_size):
        k = self.K(xa)
        q = self.Q(xa)
        v = self.V(xa)

        v_shape = v.shape[-self.num_dims:]

        rearrangement = dict(
            pattern=f'(b k) (a d) {" ".join(f"d{i}" for i in range(self.num_dims))} -> b a k (d {" ".join(f"d{i}" for i in range(self.num_dims))})',
            b=batch_size,
            a=self.n_head,
        )
        k = rearrange(k, **rearrangement)
        q = rearrange(q, **rearrangement)
        v = rearrange(v, **rearrangement)

        dprod = torch.matmul(q, k.transpose(-1, -2))
        dprod = dprod / (self.temperature * np.sqrt(k.shape[-1]))
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)
        rearrange_args = dict(
            pattern=f'b a k (d {" ".join(f"d{i}" for i in range(self.num_dims))}) -> b k a d {" ".join(f"d{i}" for i in range(self.num_dims))}',
            d=self.head_codimension,
        )
        rearrange_args.update(
            {f'd{i}': v_shape[i] for i in range(self.num_dims)})
        attention = rearrange(attention, **rearrange_args)
        attention = rearrange(
            attention,
            f'b k a d {" ".join(f"d{i}" for i in range(self.num_dims))} -> (b k) (a d) {" ".join(f"d{i}" for i in range(self.num_dims))}')
        return attention

    def forward(self, x, output_shape=None):
        if self.permutation_eq:
            return self._forward_equivariant(x)
        else:
            return self._forward_non_equivariant(x)

    def _forward_equivariant(self, x):
        batch_size = x.shape[0]
        output_shape = x.shape[-self.num_dims:]

        assert x.shape[1] % self.token_codimension == 0

        xa = rearrange(
            x,
            f'b (k d) {" ".join(f"d{i}" for i in range(self.num_dims))} -> (b k) d {" ".join(f"d{i}" for i in range(self.num_dims))}',
            d=self.token_codimension)
        xa_norm = self.norm1(xa)

        attention = self.compute_attention(xa_norm, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        attention = self.attention_normalizer(attention + xa)
        attention = rearrange(
            attention,
            f'(b k) d {" ".join(f"d{i}" for i in range(self.num_dims))} -> b (k d) {" ".join(f"d{i}" for i in range(self.num_dims))}',
            b=batch_size)
        attention = rearrange(
            attention,
            f'b (k d) {" ".join(f"d{i}" for i in range(self.num_dims))} -> (b k) d {" ".join(f"d{i}" for i in range(self.num_dims))}',
            d=self.mixer_token_codimension)

        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        output = self.mixer_out_normalizer(output) + attention
        output = rearrange(
            output,
            f'(b k) d {" ".join(f"d{i}" for i in range(self.num_dims))} -> b (k d) {" ".join(f"d{i}" for i in range(self.num_dims))}',
            b=batch_size)

        return output

    def _forward_non_equivariant(self, x):
        batch_size = x.shape[0]
        output_shape = x.shape[-self.num_dims:]

        assert x.shape[1] % self.token_codimension == 0

        x_norm = self.norm1(x)
        xa = rearrange(
            x_norm,
            f'b (k d) {" ".join(f"d{i}" for i in range(self.num_dims))} -> (b k) d {" ".join(f"d{i}" for i in range(self.num_dims))}',
            d=self.token_codimension)

        attention = self.compute_attention(xa, batch_size)
        if self.proj is not None:
            attention = self.proj(attention)

        attention = rearrange(
            attention,
            f'(b k) d {" ".join(f"d{i}" for i in range(self.num_dims))} -> b (k d) {" ".join(f"d{i}" for i in range(self.num_dims))}',
            b=batch_size)
        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        return output
