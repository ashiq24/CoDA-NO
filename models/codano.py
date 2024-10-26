from functools import partial
import logging
from typing import Literal, NamedTuple, Optional
import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F
from neuralop.layers.padding import DomainPadding
from layers.codano_block_nd import CodanoBlockND
from layers.fino_nd import SpectralConvKernel2d
from layers.variable_encoding import VariableEncoding2d


# TODO replace with nerualop.MLP module
class PermEqProjection(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        n_dim=2,
        non_linearity=F.gelu,
        permutation_invariant=False,
    ):
        """Permutation invariant projection layer.

        Performs linear projections on each channel separately.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = (in_channels
                                if hidden_channels is None else
                                hidden_channels)
        self.non_linearity = non_linearity
        Conv = getattr(nn, f'Conv{n_dim}d')

        self.permutation_invariant = permutation_invariant

        self.fc1 = Conv(in_channels, hidden_channels, 1)
        self.norm = nn.InstanceNorm2d(hidden_channels, affine=True)
        self.fc2 = Conv(hidden_channels, out_channels, 1)

    def forward(self, x):
        batch = x.shape[0]
        if self.permutation_invariant:
            assert x.shape[1] % self.in_channels == 0, \
                "Total Number of Channels is not divisible by number of tokens"
            x = rearrange(x, 'b (g c) h w -> (b g) c h w', c=self.in_channels)

        x = self.fc1(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        if self.permutation_invariant:
            x = rearrange(x, '(b g) c h w -> b (g c) h w', b=batch)
        return x


class VariableEncodingArgs(NamedTuple):
    basis: Literal["sht", "fft"]
    n_channels: int
    """Number of extra encoding channels per variable."""
    modes_x: int
    modes_y: int
    modes_t: Optional[int] = None


class CodANO(nn.Module):
    """
    Parameters
    ---
    input_token_codimension : input token codim/number of channel per input token
    out_token_codim=None : output token codim/number of channel per output token
    hidden_token_codim=None :
    lifting_token_codim=None :
    var_encoding=False : boolean
        if true then it adds variable encoding with each channel
    var_num=None :  denotes the number of variables
    var_enco_basis='sht' :  specify the basis funtion for variable encodings
    var_enco_channels=1 : number of channels for each variable encoding
    var_enco_mode_x=50 : number of x modes for each variable encoding
    var_enco_mode_y=50 : number of y models for each variable encoding
    enable_cls_token=False : if true, learnable cls token will be added
    static_channels_num=0 :
        Number of static channels to be concatenated (xy grid, land/sea mask etc)
    static_features=None :
        The static feature (it will be taken from the Preprocessor while
        initializing the model)
    integral_operator_top :
        Required for the re-grid operation (for example: from equiangular to LG grid.)
    integral_operator_bottom :
        Required for the re-grid operation (for example: from LG grid to equiangular)
    """

    def __init__(
        self,
        input_token_codimension,
        output_token_codimension=None,
        hidden_token_codimension=None,
        lifting_token_codimension=None,
        n_layers=4,
        n_modes=None,
        max_n_modes=None,
        scalings=None,
        n_heads=1,
        non_linearity=F.gelu,
        layer_kwargs={'use_mlp': False,
                      'mlp_dropout': 0,
                      'mlp_expansion': 1.0,
                      'non_linearity': F.gelu,
                      'norm': None,
                      'preactivation': False,
                      'fno_skip': 'linear',
                      'horizontal_skip': 'linear',
                      'mlp_skip': 'linear',
                      'separable': False,
                      'factorization': None,
                      'rank': 1.0,
                      'fft_norm': 'forward',
                      'normalizer': 'instance_norm',
                      'joint_factorization': False,
                      'fixed_rank_modes': False,
                      'implementation': 'factorized',
                      'decomposition_kwargs': dict(),
                      'normalizer': False},
        per_channel_attention=True,
        operator_block=CodanoBlockND,
        integral_operator=SpectralConvKernel2d,
        integral_operator_top=partial(
            SpectralConvKernel2d, sht_grid="legendre-gauss"),
        integral_operator_bottom=partial(
            SpectralConvKernel2d, isht_grid="legendre-gauss"),
        projection=True,
        lifting=True,
        domain_padding=0.5,
        domain_padding_mode='one-sided',
        n_variables=None,
        variable_encoding_args: VariableEncodingArgs = None,
        enable_cls_token=False,
        logger=None,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert len(
            n_modes) == n_layers, "number of modes for all layers are not given"
        assert len(n_heads) == n_layers, \
            "number of Attention head for all layers are not given"
        if integral_operator_bottom is None:
            integral_operator_bottom = integral_operator
        if integral_operator_top is None:
            integral_operator_top = integral_operator
        self.n_dim = len(n_modes[0])
        self.input_token_codimension = input_token_codimension
        # self.n_variables = n_variables
        if hidden_token_codimension is None:
            hidden_token_codimension = input_token_codimension
        if lifting_token_codimension is None:
            lifting_token_codimension = input_token_codimension
        if output_token_codimension is None:
            output_token_codimension = input_token_codimension

        self.hidden_token_codimension = hidden_token_codimension
        self.n_modes = n_modes
        self.max_n_modes = max_n_modes
        self.scalings = scalings
        self.non_linearity = non_linearity
        self.n_heads = n_heads
        self.integral_operator = integral_operator
        self.lifting = lifting
        self.projection = projection
        self.num_dims = len(n_modes[0])
        self.enable_cls_token = enable_cls_token

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

        self.layer_kwargs = layer_kwargs
        if layer_kwargs is None:
            self.layer_kwargs = {
                'incremental_n_modes': None,
                'use_mlp': False,
                'mlp_dropout': 0,
                'mlp_expansion': 1.0,
                'non_linearity': F.gelu,
                'norm': None,
                'preactivation': False,
                'fno_skip': 'linear',
                'horizontal_skip': 'linear',
                'mlp_skip': 'linear',
                'separable': False,
                'factorization': None,
                'rank': 1.0,
                'fft_norm': 'forward',
                'normalizer': 'instance_norm',
                'joint_factorization': False,
                'fixed_rank_modes': False,
                'implementation': 'factorized',
                'decomposition_kwargs': None,
            }

        # self.n_static_channels = n_static_channels
        """The number of static channels for all variable channels."""

        # calculating scaling
        if self.scalings is not None:
            self.end_to_end_scaling = self.get_output_scaling_factor(
                np.ones_like(self.scalings[0]),
                self.scalings
            )
        else:
            self.end_to_end_scaling = 1
        self.logger.debug(f"{self.end_to_end_scaling=}")
        if isinstance(self.end_to_end_scaling, (float, int)):
            self.end_to_end_scaling = [self.end_to_end_scaling] * self.n_dim

        # Setting up domain padding for encoder and reconstructor
        if domain_padding is not None and domain_padding > 0:
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=self.end_to_end_scaling,
            )
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        # A variable + it's variable encoding + the static channel(s)
        # together constitute a token
        # n_lifted_channels = self.input_token_codimension + \
        #     variable_encoding_args.n_channels + \
        #     self.n_static_channels
        if self.lifting:
            self.lifting = PermEqProjection(
                in_channels=input_token_codimension,
                out_channels=hidden_token_codimension,
                hidden_channels=lifting_token_codimension,
                n_dim=self.n_dim,
                non_linearity=self.non_linearity,
                permutation_invariant=True,   # Permutation
            )
        # elif self.use_variable_encoding:
        #     hidden_token_codimension = n_lifted_channels

        cls_dimension = 1 if enable_cls_token else 0
        self.codimension_size = hidden_token_codimension * n_variables + cls_dimension

        self.logger.debug(
            f"Expected number of channels: {self.codimension_size=}")

        self.base = nn.ModuleList([])
        for i in range(self.n_layers):
            if i == 0 and self.n_layers != 1:
                conv_op = integral_operator_top
            elif i == self.n_layers - 1 and self.n_layers != 1:
                conv_op = integral_operator_bottom
            else:
                conv_op = self.integral_operator

            self.base.append(
                operator_block(
                    n_modes=self.n_modes[i],
                    max_n_modes=self.max_n_modes[i],
                    n_head=self.n_heads[i],
                    token_codim=hidden_token_codimension,
                    output_scaling_factor=[self.scalings[i]],
                    SpectralConvolution=conv_op,
                    codim_size=self.codimension_size,
                    per_channel_attention=per_channel_attention,
                    num_dims=self.num_dims,
                    logger=self.logger.getChild(f"base[{i}]"),
                    **self.layer_kwargs,
                )
            )

        if self.projection:
            self.projection = PermEqProjection(
                in_channels=hidden_token_codimension,
                out_channels=output_token_codimension,
                hidden_channels=lifting_token_codimension,
                n_dim=self.n_dim,
                non_linearity=self.non_linearity,
                permutation_invariant=True,   # Permutation
            )

        if enable_cls_token:
            self.cls_token = VariableEncoding2d(
                1,
                hidden_token_codimension,
                (variable_encoding_args.modes_x,
                 variable_encoding_args.modes_y),
                basis=variable_encoding_args.basis)

    def get_output_scaling_factor(self, initial_scale, scalings_per_layer):
        for k in scalings_per_layer:
            initial_scale = np.multiply(initial_scale, k)
        initial_scale = initial_scale.tolist()
        if len(initial_scale) == 1:
            initial_scale = initial_scale[0]
        return initial_scale

    def get_device(self,):
        return self.cls_token.coefficients_r.device

    def forward(self, x: torch.Tensor):
        if self.lifting:
            x = self.lifting(x)

        if self.enable_cls_token:
            cls_token = self.cls_token(x).unsqueeze(0)
            repeat_shape = [1 for _ in x.shape]
            repeat_shape[0] = x.shape[0]
            x = torch.cat(
                [
                    cls_token.repeat(*repeat_shape),
                    x,
                ],
                dim=1,
            )

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        output_shape_en = [round(i * j) for (i,
                                             j) in zip(x.shape[-self.n_dim:],
                                                       self.end_to_end_scaling)]

        cur_output_shape = None
        for layer_idx in range(self.n_layers):
            if layer_idx == self.n_layers - 1:
                cur_output_shape = output_shape_en
            x = self.base[layer_idx](x, output_shape=cur_output_shape)
            # self.logger.debug(f"{x.shape} (block[{layer_idx}])")

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        if self.projection:
            x = self.projection(x)
            # self.logger.debug(f"{x.shape} (projection)")

        return x


class CoDANOTemporal:
    def __call__(self, x):
        pass
