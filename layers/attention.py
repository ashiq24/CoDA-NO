from functools import partial

import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from neuralop.layers.fno_block import FNOBlocks
from .fino import SpectralConvKernel2d


class TnoBlock2d(nn.Module):
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
        mlp_dropout=0,
        non_linearity=F.gelu,
        norm=None,
        preactivation=False,
        fno_skip='linear',
        mlp_skip='soft-gating',
        mlp_expansion=1.0,
        separable=False,
        factorization='tucker',
        rank=1.0,
        SpectralConv=SpectralConvKernel2d,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation='factorized',
        decomposition_kwargs=None,
        fft_norm='forward',
        codim_size=None,
        per_channel_attention=True,
        permutation_eq=True,
        **kwarg,
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
                SpectralConv,
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

        # = partial(SpectralConv,frequency_mixer = False)

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
                    SpectralConv,
                    rank=0.5,
                    factorization=None,
                ),
                n_layers=1,
                **common_args,
            )
        else:
            self.proj = None

        self.attention_normalizer = nn.InstanceNorm2d(self.token_codim, affine=True)

        mixer_args = dict(
            n_modes=n_modes,
            output_scaling_factor=1,
            non_linearity=non_linearity,
            norm='instance_norm',
            fno_skip=fno_skip,
            SpectralConv=partial(
                SpectralConv,
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
            self.norm1 = nn.InstanceNorm2d(self.token_codim, affine=True)
            self.norm2 = nn.InstanceNorm2d(self.mixer_token_codim, affine=True)
            self.mixer_out_normalizer = nn.InstanceNorm2d(
                self.mixer_token_codim,
                affine=True,
            )
        else:
            self.mixer = FNOBlocks(
                in_channels=codim_size,
                out_channels=codim_size,
                n_layers=2,
                **mixer_args,
                **common_args,
            )
            self.norm1 = nn.InstanceNorm2d(codim_size, affine=True)
            self.norm2 = nn.InstanceNorm2d(codim_size, affine=True)
            self.mixer_out_normalizer = nn.InstanceNorm2d(codim_size, affine=True)

    def forward(self, x, output_shape=None):
        batch = x.shape[0]
        n_token = x.shape[1] // self.token_codim,
        in_res_x = x.shape[-2]
        in_res_y = x.shape[-1]

        assert x.shape[1] % self.token_codim == 0

        if not self.permutation_eq:
            x_norm = self.norm1(x)
        else:
            x_norm = x
        xa = rearrange(x_norm, 'b (t d) h w -> (b t) d h w', d=self.token_codim)

        if self.permutation_eq:
            xa_norm = self.norm1(xa)
        else:
            xa_norm = xa

        k = self.K.convs(xa_norm)
        q = self.Q.convs(xa_norm)
        v = self.V.convs(xa_norm)

        res_x, res_y = k.shape[-2], k.shape[-1]
        value_res_x, value_res_y = v.shape[-2], v.shape[-1]

        k = rearrange(k, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head)
        q = rearrange(q, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head)
        v = rearrange(v, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head)

        dprod = torch.matmul(q, k.transpose(-1, -2)) * np.sqrt(k.shape[-1])
        dprod = F.softmax(dprod, dim=-1)

        attention = torch.matmul(dprod, v)
        attention = rearrange(
            attention,
            'b a t (d h w) -> b t a d h w',
            d=self.head_codim,
            h=value_res_x,
            w=value_res_y,
        )
        attention = rearrange(attention, 'b t a d h w -> (b t) (a d) h w')

        if self.proj is not None:
            attention = self.proj.convs(attention)

        if not self.permutation_eq:
            attention = rearrange(attention, '(b t) d h w -> b (t d) h w', b=batch)
            attention_normalized = self.norm2(attention)
            output = self.mixer(attention_normalized, output_shape=(in_res_x, in_res_y))
        else:
            attention = self.attention_normalizer(attention) + xa
            attention = rearrange(attention, '(b t) d h w -> b (t d) h w', b=batch)
            # print("Attention shape", attention.shape)
            attention = rearrange(
                attention,
                'b (t d) h w -> (b t) d h w',
                d=self.mixer_token_codim)
            # print("Attention shape", attention.shape)

            attention_normalized = self.norm2(attention)
            output = self.mixer(attention_normalized, output_shape=(in_res_x, in_res_y))

            output = self.mixer_out_normalizer(output) + attention
            # print("outshape", output.shape)
            output = rearrange(output, '(b t) d h w -> b (t d) h w', b=batch)
        return output
