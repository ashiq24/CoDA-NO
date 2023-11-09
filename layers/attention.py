from functools import partial

import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from neuralop.layers.fno_block import FNOBlocks
from .fino import SpectralConvKernel2d

AffineNormalizer2D = partial(nn.InstanceNorm2d, affine=True)
AffineNormalizer3D = partial(nn.InstanceNorm3d, affine=True)

class TNOBlock(nn.Module):
    def __init__(
        self,
        n_modes,
        n_head=1,
        token_codim=1,
        output_scaling_factor=None,
        incremental_n_modes=None,
        head_codim=None,
        use_mlp=False,
        mlp=None,
        non_linearity=F.gelu,
        preactivation=False,
        fno_skip='linear',
        mlp_skip='soft-gating',
        mlp_expansion=1.0,
        separable=False,
        factorization='tucker',
        rank=1.0,
        SpectralConvolution=None,
        Normalizer=None,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation='factorized',
        decomposition_kwargs=None,
        fft_norm='forward',
        codim_size=None,
        per_channel_attention=True,
        permutation_eq=True,
        **_kwargs,
    ):
        super().__init__()

        self.variable_codim = token_codim  # codim of each variable
        self.token_codim = token_codim  # codim of each token, they are equal

        # codim of attention from each head
        self.head_codim = head_codim if head_codim is not None else token_codim
        self.n_head = n_head  # number of heads
        self.output_scaling_factor = output_scaling_factor  # output scaling factor

        # attention per channel not per variables
        self.per_channel_attention = per_channel_attention

        # making last mixer permutation equivariant
        self.permutation_eq = permutation_eq

        if self.n_head is not None:
            # recalculating the value of `head_codim`
            self.head_codim = max(token_codim // self.n_head, 1)

        self.codim_size = codim_size
        self.mixer_token_codim = token_codim

        if per_channel_attention:
            # for per channel attention, forcing the values of token dims
            self.token_codim = 1
            self.head_codim = 1

        # this scale used for downsampling Q,K functions
        scale = min(self.n_head, 2)
        if self.per_channel_attention:
            scale = 4

        mixer_modes = [i // scale for i in n_modes]

        print(f"{rank=}, "
              f"{factorization=}, "
              f"{self.head_codim=}, "
              f"{scale=}, "
              f"{mixer_modes=}")

        if not per_channel_attention:
            print(f"Token dim={self.token_codim}\n"
                  f"number heads={self.n_head}\n"
                  f"Head co-dim={self.head_codim}")

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
            in_channels=self.token_codim,
            out_channels=self.n_head * self.head_codim,
            n_modes=mixer_modes,
            # args below are shared with Projection block
            non_linearity=lambda x: x,
            fno_skip='linear',
            norm=None,
            apply_skip=True,
            SpectralConv=partial(
                SpectralConvolution,
                rank=0.5,
                factorization=None,
            ),
            n_layers=1,
        )
        self.K = FNOBlocks(
            output_scaling_factor=1 / scale,
            **kqv_args,
            **common_args,
        )
        self.Q = FNOBlocks(
            output_scaling_factor=1 / scale,
            **kqv_args,
            **common_args,
        )
        self.V = FNOBlocks(
            output_scaling_factor=1,
            **kqv_args,
            **common_args,
        )

        if self.n_head * self.head_codim != self.token_codim:
            self.proj = FNOBlocks(
                in_channels=self.n_head * self.head_codim,
                out_channels=self.token_codim,
                n_modes=n_modes,
                output_scaling_factor=1,
                # args below are shared with KQV blocks
                apply_skip=True,
                non_linearity=lambda x: x,
                fno_skip='linear',
                norm=None,
                SpectralConv=partial(
                    SpectralConvolution,
                    rank=0.5,
                    factorization=None,
                ),
                n_layers=1,
                **common_args,
            )
        else:
            self.proj = None

        self.attention_normalizer = Normalizer(self.token_codim)

        mixer_args = dict(
            n_modes=n_modes,
            output_scaling_factor=1,
            non_linearity=non_linearity,
            norm='instance_norm',
            fno_skip=fno_skip,
            SpectralConv=partial(
                SpectralConvolution,
                rank=0.5,
                factorization=None,
                bias=True,
            ),
        )
        # We have an option to make the last operator (MLP in regular
        # Transformer block) permutation equivariant. i.e., applying the
        # operator per variable or applying the operator on the whole channel
        # (like regular FNO).
        if permutation_eq:
            print("Permutation Equivariant with ", self.mixer_token_codim)
            self.mixer = FNOBlocks(
                in_channels=self.mixer_token_codim,
                out_channels=self.mixer_token_codim,
                apply_skip=True,
                n_layers=1,
                **mixer_args,
                **common_args,
            )
            self.norm1 = Normalizer(self.token_codim)
            self.norm2 = Normalizer(self.mixer_token_codim)
            self.mixer_out_normalizer = Normalizer(self.mixer_token_codim)

        else:
            self.mixer = FNOBlocks(
                in_channels=codim_size,
                out_channels=codim_size,
                n_layers=2,
                **mixer_args,
                **common_args,
            )
            self.norm1 = Normalizer(codim_size)
            self.norm2 = Normalizer(codim_size)
            self.mixer_out_normalizer = Normalizer(codim_size)

    def forward(self, *args):
        raise NotImplementedError(
            "Use a proper subclass of TNOBlock (i.e. TnoBlock2d or TNOBlock3D).")


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

        Assumes input ``xa`` has been normalized.
        """
        k = self.K.convs(xa)
        q = self.Q.convs(xa)
        v = self.V.convs(xa)

        value_x, value_y = v.shape[-2], v.shape[-1]

        rearrangement = dict(
            pattern='(b t) (a d) h w -> b a t (d h w)',
            b=batch_size,
            a=self.n_head,
        )
        k = rearrange(k, **rearrangement)
        q = rearrange(q, **rearrangement)
        v = rearrange(v, **rearrangement)

        dprod = torch.matmul(q, k.transpose(-1, -2)) * np.sqrt(k.shape[-1])
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)
        attention = rearrange(
            attention,
            'b a t (d h w) -> b t a d h w',
            d=self.head_codim,
            h=value_x,
            w=value_y,
        )
        attention = rearrange(attention, 'b t a d h w -> (b t) (a d) h w')
        return attention

    def forward(self, x, output_shape=None):
        if self.permutation_eq:
            return self._forward_equivariant(x)
        else:
            return self._forward_non_equivariant(x)

    def _forward_equivariant(self, x):
        batch_size = x.shape[0]
        output_shape = x.shape[-2:]

        assert x.shape[1] % self.token_codim == 0

        xa = rearrange(x, 'b (t d) h w -> (b t) d h w', d=self.token_codim)
        xa_norm = self.norm1(xa)

        attention = self.compute_attention(xa_norm, batch_size)
        if self.proj is not None:
            attention = self.proj.convs(attention)

        attention = self.attention_normalizer(attention) + xa
        attention = rearrange(attention, '(b t) d h w -> b (t d) h w', b=batch_size)
        # print("{attention.shape=}")
        attention = rearrange(
            attention,
            'b (t d) h w -> (b t) d h w',
            d=self.mixer_token_codim)
        # print("{attention.shape=}")

        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        output = self.mixer_out_normalizer(output) + attention
        # print(f"{output.shape=}")
        output = rearrange(output, '(b t) d h w -> b (t d) h w', b=batch_size)

        return output

    def _forward_non_equivariant(self, x):
        batch_size = x.shape[0]
        output_shape = x.shape[-2:]

        assert x.shape[1] % self.token_codim == 0

        x_norm = self.norm1(x)
        xa = rearrange(x_norm, 'b (t d) h w -> (b t) d h w', d=self.token_codim)

        attention = self.compute_attention(xa, batch_size)
        if self.proj is not None:
            attention = self.proj.convs(attention)

        attention = rearrange(attention, '(b t) d h w -> b (t d) h w', b=batch_size)
        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        return output


class TNOBlock3D(TNOBlock):
    def __init__(self, *args, **kwargs):
        Normalizer = kwargs.get("Normalizer")
        if Normalizer is None:
            Normalizer = AffineNormalizer3D
        kwargs["Normalizer"] = Normalizer

        # TODO write and use 3D kernel
        Convolution = kwargs.get("SpectralConvolution")
        if Convolution is None:
            Convolution = SpectralConvKernel2d
        kwargs["SpectralConvolution"] = Convolution

        super().__init__(*args, **kwargs)

    def compute_attention(self, xa, batch_size):
        """Compute the key-query-value variant of the attention matrix.

        Assumes input ``xa`` has been normalized.
        """
        k = self.K.convs(xa)
        q = self.Q.convs(xa)
        v = self.V.convs(xa)

        v_duration, v_height, v_width = v.shape[-3:]

        rearrangement = dict(
            pattern='(b k) (a d) t h w -> b a k (d t h w)',
            b=batch_size,
            a=self.n_head,
        )
        k = rearrange(k, **rearrangement)
        print(f"{k.shape=}")
        q = rearrange(q, **rearrangement)
        print(f"{q.shape=}")
        v = rearrange(v, **rearrangement)
        print(f"{v.shape=}")

        dprod = torch.matmul(q, k.transpose(-1, -2)) * np.sqrt(k.shape[-1])
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)
        attention = rearrange(
            attention,
            'b a k (d t h w) -> b k a d t h w',
            d=self.head_codim,
            t=v_duration,
            h=v_height,
            w=v_width,
        )
        attention = rearrange(attention, 'b k a d t h w -> (b k) (a d) t h w')
        return attention

    def forward(self, x, output_shape=None):
        if self.permutation_eq:
            return self._forward_equivariant(x)
        else:
            return self._forward_non_equivariant(x)

    def _forward_equivariant(self, x):
        batch_size = x.shape[0]
        output_shape = x.shape[-3:]

        assert x.shape[1] % self.token_codim == 0

        xa = rearrange(x, 'b (k d) t h w -> (b k) d t h w', d=self.token_codim)
        xa_norm = self.norm1(xa)

        attention = self.compute_attention(xa_norm, batch_size)
        if self.proj is not None:
            attention = self.proj.convs(attention)

        attention = self.attention_normalizer(attention) + xa
        attention = rearrange(
            attention,
            '(b k) d t h w -> b (k d) t h w',
            b=batch_size,
        )
        # print("{attention.shape=}")
        attention = rearrange(
            attention,
            'b (k d) t h w -> (b k) d t h w',
            d=self.mixer_token_codim)
        # print("{attention.shape=}")

        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        output = self.mixer_out_normalizer(output) + attention
        # print(f"{output.shape=}")
        output = rearrange(output, '(b k) d t h w -> b (k d) t h w', b=batch_size)

        return output

    def _forward_non_equivariant(self, x):
        batch_size = x.shape[0]
        output_shape = x.shape[-3:]

        assert x.shape[1] % self.token_codim == 0

        x_norm = self.norm1(x)
        xa = rearrange(x_norm, 'b (k d) t h w -> (b k) d t h w', d=self.token_codim)

        attention = self.compute_attention(xa, batch_size)
        if self.proj is not None:
            attention = self.proj.convs(attention)

        attention = rearrange(
            attention,
            '(b k) d t h w -> b (k d) t h w',
            b=batch_size,
        )
        attention_normalized = self.norm2(attention)
        output = self.mixer(attention_normalized, output_shape=output_shape)

        return output
