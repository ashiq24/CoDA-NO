from functools import partial
import logging
from typing import Literal, NamedTuple, Optional

import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F

from neuralop.layers.padding import DomainPadding
from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from layers.regrider import Regird
from layers.variable_encoding import VariableEncoding2d, FourierVariableEncoding3D


class Projection(nn.Module):
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
            assert x.shape[1] % self.in_channels == 0,\
                "Total Number of Channels is not divisible by number of tokens"
            x = rearrange(x, 'b (g c) h w -> (b g) c h w', c=self.in_channels)

        x = self.fc1(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        x = self.fc2(x)
        if self.permutation_invariant:
            x = rearrange(x, '(b g) c h w -> b (g c) h w', b=batch)
        return x


class ProjectionT(Projection):
    """Time-aware projection MLP layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.InstanceNorm3d(kwargs["hidden_channels"], affine=True)

    def forward(self, x):
        batch_size = x.shape[0]

        if self.permutation_invariant:
            assert x.shape[1] % self.in_channels == 0,\
                "Total Number of Channels is not divisible by number of tokens"
            x = rearrange(x, 'b (g c) t h w -> (b g) c t h w', c=self.in_channels)

        x = self.fc1(x)
        x = self.norm(x)
        x = self.non_linearity(x)
        x = self.fc2(x)

        if self.permutation_invariant:
            x = rearrange(x, '(b g) c t h w -> b (g c) t h w', b=batch_size)

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
        scalings=None,
        n_heads=1,
        non_linearity=F.gelu,
        layer_kwargs=None,
        per_channel_attention=False,
        operator_block=TnoBlock2d,
        integral_operator=SpectralConvKernel2d,
        integral_operator_top=partial(SpectralConvKernel2d, sht_grid="legendre-gauss"),
        integral_operator_bottom=\
            partial(SpectralConvKernel2d, isht_grid="legendre-gauss"),
        re_grid_input=False,
        re_grid_output=False,
        projection=True,
        lifting=True,
        domain_padding=None,
        domain_padding_mode='one-sided',
        use_variable_encodings=False,
        n_variables=None,
        variable_encoding_args: VariableEncodingArgs = None,
        enable_cls_token=False,
        n_static_channels=0,
        static_features=None,
        logger=None,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert len(n_modes) == n_layers, "number of modes for all layers are not given"
        assert len(n_heads) == n_layers, \
            "number of Attention head for all layers are not given"
        if integral_operator_bottom is None:
            integral_operator_bottom = integral_operator
        if integral_operator_top is None:
            integral_operator_top = integral_operator
        self.n_dim = len(n_modes[0])
        self.input_token_codimension = input_token_codimension
        self.n_variables = n_variables
        if hidden_token_codimension is None:
            hidden_token_codimension = input_token_codimension
        if lifting_token_codimension is None:
            lifting_token_codimension = input_token_codimension
        if output_token_codimension is None:
            output_token_codimension = input_token_codimension
        self.re_grid_input = re_grid_input
        self.re_grid_output = re_grid_output

        if self.re_grid_input:
            self.input_regrider = Regird("equiangular", "legendre-gauss")
        if self.re_grid_output:
            self.output_regrider = Regird("legendre-gauss", "equiangular")

        self.hidden_token_codimension = hidden_token_codimension
        self.n_modes = n_modes
        self.scalings = scalings
        self.n_encoding_channels = variable_encoding_args.n_channels
        self.n_heads = n_heads
        self.integral_operator = integral_operator
        self.lifting = lifting
        self.projection = projection

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

        self.register_buffer("static_features", static_features)
        self.n_static_channels = n_static_channels
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

        # Code for variable encoding
        self.use_variable_encoding = use_variable_encodings
        if self.use_variable_encoding:
            if variable_encoding_args is None:
                raise ValueError(
                    "Must provide a value for `variable_encoding_args`\n"
                    f"Got {variable_encoding_args=}"
                )
            self._initialize_variable_encoding_channels(
                n_variables,
                variable_encoding_args,
            )

        # A variable + it's variable encoding + the static channel(s)
        # together constitute a token
        n_lifted_channels = self.input_token_codimension + \
                            variable_encoding_args.n_channels + \
                            self.n_static_channels
        if self.lifting:
            self.logger.debug(
                'using lifting with:\n'
                f'\t{n_lifted_channels=}\n'
                f'\t{hidden_token_codimension=}\n'
                f'\t{lifting_token_codimension=}\n'
            )
            self.lifting = self._mk_lifting_operator(
                n_lifted_channels,
                hidden_token_codimension,
                lifting_token_codimension,
            )
        elif self.use_variable_encoding:
            hidden_token_codimension = n_lifted_channels

        count = 1 if enable_cls_token else 0
        self.codimension_size = hidden_token_codimension * (n_variables + count)

        self.logger.debug(f"Expected number of channels: {self.codimension_size=}")

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
                    n_head=self.n_heads[i],
                    token_codim=hidden_token_codimension,
                    output_scaling_factor=[self.scalings[i]],
                    SpectralConvolution=conv_op,
                    codim_size=self.codimension_size,
                    per_channel_attention=per_channel_attention,
                    logger=self.logger.getChild(f"base[{i}]"),
                    **self.layer_kwargs,
                )
            )

        if self.projection:
            self.logger.debug(
                'using projection with:\n'
                f'\t{hidden_token_codimension=}\n'
                f'\t{output_token_codimension=}\n'
                f'\t{lifting_token_codimension=}\n'
                f'\t{non_linearity=}\n'
            )
            self.projection = self._mk_projection_operator(
                hidden_token_codimension,
                output_token_codimension,
                lifting_token_codimension,
                non_linearity,
            )

        # Variable encoding
        self.enable_cls_token = enable_cls_token
        if enable_cls_token:
            self.logger.debug("initializing CLS token")
            cls_token_args = variable_encoding_args._replace(
                n_channels=hidden_token_codimension)
            self.cls_token = self._mk_variable_encoder(cls_token_args)

    def _initialize_variable_encoding_channels(
        self,
        n_variables,
        variable_encoding_args,
    ):
        """
        Each variable along with its variable encoding should remain
        consecutive to be considered a single token
        for variable_encoding with codim = 2
        the channels can be:
        ```python
        [
            variable1, variable_encoding1, variable_encoding1, static_channel,
            variable2, variable_encoding2, variable_encoding2, static_channel,
            ...
        ]
        ```
        Each token is extracted accordingly in the attention module
        """
        assert n_variables is not None
        self.logger.debug(
            "using variable encoding with:\n"
            f"{n_variables=}\n"
            f"{variable_encoding_args=}\n"
        )

        args = variable_encoding_args._replace(
            n_channels=n_variables * self.n_encoding_channels
        )
        self.variable_encoder = self._mk_variable_encoder(args)

        expansion_factor = 1 + self.n_static_channels + self.n_encoding_channels
        # Allocate every Nth channel for the untransformed variable
        # where N=``expansion_factor``
        self.variable_channels = [i * expansion_factor for i in range(n_variables)]

        # Allocate N static channels for each known variable,
        # where N=``n_static_channels``
        self.static_channels = []
        if self.n_static_channels != 0:
            for v in self.variable_channels:
                # append elements from an iterable:
                self.static_channels.extend(
                    range(v + 1, v + self.n_static_channels + 1))

        # Allocate all remaining channels as encoding channels
        # for the preceding variable:
        self.encoding_channels = sorted(list(
            set(range(n_variables * expansion_factor))
            - set(self.variable_channels)
            - set(self.static_channels)
        ))

    def _mk_variable_encoder(self, ve_args):
        return VariableEncoding2d(
            channel=ve_args.n_channels,
            mode_x=ve_args.modes_x,
            mode_y=ve_args.modes_y,
            basis=ve_args.basis,
        )

    def _mk_lifting_operator(
        self,
        n_lifted_channels,
        hidden_token_codimension,
        lifting_token_codimension,
    ):
        return Projection(
            in_channels=n_lifted_channels,
            out_channels=hidden_token_codimension,
            hidden_channels=lifting_token_codimension,
            n_dim=self.n_dim,
            permutation_invariant=True,   # Permutation
        )

    def _mk_projection_operator(
        self,
        hidden_token_codimension,
        out_token_codimension,
        lifting_token_codimension,
        non_linearity,
    ):
        return Projection(
            in_channels=hidden_token_codimension,
            out_channels=out_token_codimension,
            hidden_channels=lifting_token_codimension,
            non_linearity=non_linearity,
            n_dim=self.n_dim,
            permutation_invariant=True,  # permutation
        )

    @staticmethod
    def get_output_scaling_factor(initial_scale, scalings_per_layer):
        for k in scalings_per_layer:
            initial_scale = np.multiply(initial_scale, k)
        initial_scale = initial_scale.tolist()
        if len(initial_scale) == 1:
            initial_scale = initial_scale[0]
        return initial_scale

    def get_device(self,):
        return self.cls_token.coefficients_r.device

    def forward(self, inp):
        if self.re_grid_input:
            inp = self.input_regrider(inp)

        if self.use_variable_encoding:
            x = self.encode_variables(inp)
        else:
            x = inp

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

        output_shape_en = [round(i * j) for (i, j)
                           in zip(x.shape[-self.n_dim:], self.end_to_end_scaling)]

        cur_output_shape = None
        for layer_idx in range(self.n_layers):
            if layer_idx == self.n_layers - 1:
                cur_output_shape = output_shape_en
            x = self.base[layer_idx](x, output_shape=cur_output_shape)

        if self.projection:
            x = self.projection(x)

        if self.re_grid_output:
            x = self.output_regrider(x)

        return x

    def encode_variables(self, inp):
        """Applies variable encodings to given input tensor.

        Transform the low-dimensional input tensor to a higher-dimensional
        representation where each variable channel has been augmented with both
        learned encodings and static channels (where the latter may be
        positional encoding, etc.)
        """
        # Token dimensionality supersedes channel dimensionality:
        batch_size, _c, width, height = inp.shape
        token_size = (len(self.variable_channels) +
                      len(self.encoding_channels) +
                      len(self.static_channels))
        x = torch.zeros(
            (batch_size, token_size, width, height),
            device=inp.device,
            dtype=inp.dtype,
        )

        x[:, self.variable_channels, :, :] = inp

        variable_encoding = self.variable_encoder(x).to(x.device)
        # Unsqueeze variable encoding in 0th dimension and repeat it
        # until it matches the dimension of ``batch_size``
        x[:, self.encoding_channels, :, :] = \
            variable_encoding[None, :, :, :].repeat(batch_size, 1, 1, 1)

        # if self.n_static_channels != 0:
        if len(self.static_channels) > 0:
            # Repeat `static_features` as many times as we can
            # within the space of `static_channels`
            n_static_copies = \
                len(self.static_channels) // self.static_features.shape[1]
            x[:, self.static_channels, :, :] = \
                self.static_features[:, :, :, :] \
                    .repeat(batch_size, n_static_copies, 1, 1)

        return x

class CoDANOTemporal(CodANO):
    """Time-aware Co-domain Attention Operator that acts on 2+1 dim states"""

    def _mk_variable_encoder(self, ve_args):
        modes = (ve_args.modes_t, ve_args.modes_x, ve_args.modes_y,)
        return FourierVariableEncoding3D(
            channel_size=ve_args.n_channels,
            modes=modes,
        )

    def _mk_lifting_operator(
        self,
        n_lifted_channels,
        hidden_token_codimension,
        lifting_token_codimension,
    ):
        return ProjectionT(
            in_channels=n_lifted_channels,
            out_channels=hidden_token_codimension,
            hidden_channels=lifting_token_codimension,
            n_dim=self.n_dim,
            permutation_invariant=True,   # Permutation
        )

    def _mk_projection_operator(
        self,
        hidden_token_codimension,
        out_token_codimension,
        lifting_token_codimension,
        non_linearity,
    ):
        return ProjectionT(
            in_channels=hidden_token_codimension,
            out_channels=out_token_codimension,
            hidden_channels=lifting_token_codimension,
            non_linearity=non_linearity,
            n_dim=self.n_dim,
            permutation_invariant=True,  # permutation
        )

    def encode_variables(self, inp):
        """Applies variable encodings to given input tensor.

        Transform the low-dimensional input tensor to a higher-dimensional
        representation where each variable channel has been augmented with both
        learned encodings and static channels (where the latter may be
        positional encoding, etc.)
        """
        # Token dimensionality supersedes channel dimensionality:
        batch_size, _c, duration, width, height = inp.shape
        token_size = (len(self.variable_channels) +
                      len(self.encoding_channels) +
                      len(self.static_channels))
        x = torch.zeros(
            (batch_size, token_size, duration, width, height),
            device=inp.device,
            dtype=inp.dtype,
        )

        x[:, self.variable_channels, :, :, :] = inp

        variable_encoding = self.variable_encoder(x).to(x.device)
        # Unsqueeze variable encoding in 0th dimension (indexed by `None` below)
        # and repeat it until it matches the dimension of ``batch_size``
        x[:, self.encoding_channels, :, :, :] = \
            variable_encoding[None, :, :, :, :].repeat(batch_size, 1, 1, 1, 1)

        # if self.n_static_channels != 0:
        if len(self.static_channels) > 0:
            # Repeat `static_features` as many times as we can
            # within the space of `static_channels`
            n_static_copies = \
                len(self.static_channels) // self.static_features.shape[1]
            x[:, self.static_channels, :, :, :] = \
                self.static_features[:, :, :, :, :] \
                    .repeat(batch_size, n_static_copies, 1, 1, 1)

        return x
