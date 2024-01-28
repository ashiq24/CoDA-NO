import enum
from functools import partial
import logging
from typing import Optional, Type, Union, Tuple, Dict, List
import numpy as np
import torch
from torch import nn
from utils import *
from neuralop.layers.fno_block import FNOBlocks
from neuralop.models import FNO

from data_utils.data_utils import (
    MaskerNonuniformMesh,
    MaskerUniform,
    MaskerUniformIndependent,
    batched_masker,
)
from data_utils.hdf5_datasets import Equation
from layers.attention import TnoBlock2d, TNOBlock
from layers.fino import SpectralConvKernel2d, SpectralConvolutionKernel3D
from layers.variable_encoding import FourierVariableEncoding3D
from models.codano import CodANO, VariableEncodingArgs
from models.codano_gino import CondnoGino
from models.fno_gino import FnoGno
from models.gnn import GNN
from models.deeponet import DeepONet
from models.vit import VitGno

# TODO merge methods get_ssl_models_coda*()
def get_ssl_models_codaNo(
    params,
    module: Type[CodANO],
    block: Type[TNOBlock],
    convolution: Type[Union[SpectralConvKernel2d, SpectralConvolutionKernel3D]],
    logger: Optional[logging.Logger] = None,
    verbose=True,
):
    # We use TNO inside SSL transformer model. That has a encoder and
    # prediction/decoder part (the decoder is optional).
    # Encoder part - encodes the input function
    # Decoder part - does the prediction (e.g. fluid flow of next time step)
    # For SLL it has a reconstruction head and a dense contrastive head

    if params.tno_integral_operator == 'fno':
        integral_operator = partial(
            convolution,
            transform_type=params.transform_type,
            frequency_mixer=False,
            verbose=verbose,
        )
    elif params.tno_integral_operator == 'fino':
        integral_operator = partial(
            convolution,
            transform_type=params.transform_type,
            frequency_mixer=True,
            verbose=verbose,
        )
    else:
        raise ValueError(
            f'Invalid config: {params.tno_integral_operator=}. '
            f'Expected either "fno" or "fino"')

    if logger is None:
        logger = logging.getLogger()
    logger.info("Generating Encoder")

    static_features = None
    n_static_channels = 0

    # if params.add_static_feature:
    #     # Taking the static features from  which will be passed to the encoder
    #     # to concatenated with each of the input variables.
    #     static_features = None
    #     n_static_channels = 0

    logger.debug(
        f"per variable token dimension="
        f"{1 + params.n_encoding_channels + n_static_channels}")
    # logger.debug(
    #     # f"\n {params.n_variables=}"
    #     f"\n {n_static_channels=}"
    # )

    # TODO(mogab): Is this variable count still necessary? Probably?
    # TODO(mogab): Assumes no two equations share any variables.
    n_variables = sum(params.variables_per_equation.values())
    common_args = dict(
        operator_block=block,
        n_variables=n_variables,
        integral_operator=integral_operator,
        integral_operator_top=integral_operator,
        integral_operator_bottom=integral_operator,
        per_channel_attention=params.per_channel_attention,
        variable_encoding_args=VariableEncodingArgs(
            basis="fft",
            n_channels=params.n_encoding_channels,
            modes_x=params.encoding_modes_x,
            modes_y=params.encoding_modes_y,
            modes_t=params.encoding_modes_t,
        ),
    )

    encoder = module(
        **params.encoder,
        lifting=True,
        projection=False,
        # use_variable_encodings=params.use_variable_encodings,
        enable_cls_token=params.enable_cls_token,
        n_static_channels=n_static_channels,
        static_features=static_features,
        logger=logger.getChild("encoder"),
        **common_args,
    )
    logger.info("*" * 40)

    if params.reconstruction:
        logger.info("Generating Decoder")
        decoder = module(
            **params.decoder,
            lifting=False,
            projection=True,
            enable_cls_token=params.enable_cls_token,
            logger=logger.getChild("decoder"),
            **common_args,
        )
    else:
        decoder = None
    logger.info("*" * 40)

    contrastive = None

    logger.info('generating Predictor')
    predictor = module(
        **params.predictor,
        lifting=False,
        projection=False,
        logger=logger.getChild("predictor"),
        **common_args,
    )
    logger.info("*" * 40)

    return encoder, decoder, contrastive, predictor


def get_ssl_models_codano_gino(params):
    # We use tno inside SSLtransformer model. That has a encoder and prediction/(decoder) part.
    # Encoder part - encodes the input function
    # Decoder part - does the prediction (eg. fuild flow of next time step)
    # For SLL it has a reconstruction head and a dense contarstive head

    # read input grid and prepare uniform grid
    mesh = get_mesh(params.input_mesh_location)
    input_mesh = torch.from_numpy(mesh).type(torch.float).cuda()

    minx, maxx = np.min(mesh[:, 0]), np.max(mesh[:, 0])
    miny, maxy = np.min(mesh[:, 1]), np.max(mesh[:, 1])

    size_x, size_y = params.grid_size
    idx_x = torch.arange(
        start=minx,
        end=maxx + (maxx - minx) / size_x - 1e-5,
        step=(maxx - minx) / (size_x - 1),
    )
    idx_y = torch.arange(
        start=miny,
        end=maxy + (maxy - miny) / size_y - 1e-5,
        step=(maxy - miny) / (size_y - 1),
    )
    x, y = torch.meshgrid(idx_x, idx_y, indexing='ij')
    output_mesh = torch.transpose(
        torch.stack([x.flatten(), y.flatten()]),
        dim0=0,
        dim1=1,
    ).type(torch.float).cuda()

    assert x.shape[0] == size_x
    assert x.shape[1] == size_y

    # block = None
    block = TnoBlock2d

    if params.tno_integral_op == 'fno':
        int_op = partial(SpectralConvKernel2d,
                         transform_type=params.transform_type,
                         frequency_mixer=False)
        int_op_top = int_op
        int_op_bottom = int_op
    elif params.tno_integral_op == 'fino':
        int_op = partial(SpectralConvKernel2d,
                         transform_type=params.transform_type)
        int_op_top = int_op
        int_op_bottom = int_op
    else:
        raise (Exception('Int. Op. config Error'))
    print("Generating Encoder")

    static_features = None
    static_channels_num = params.n_static_channels

    print("Token Dim-->", 1 + params.n_encoding_channels + static_channels_num)
    expanded_token_dim = 1 + params.n_encoding_channels + static_channels_num

    encoder = CondnoGino(
        expanded_token_dim,
        input_grid=input_mesh,
        output_grid=output_mesh,
        radius=params.radius,
        n_neigbor=params.n_neigbor,
        fixed_neighbour=params.fixed_neighbour,
        gno_mlp_layers=params.gno_mlp_layers,
        grid_size=params.grid_size,
        hidden_token_codim=params.hidden_token_codim_en,
        lifting_token_codim=params.lifting_token_codim_en,
        n_layers=params.n_layers_en,
        n_heads=params.n_heads_en,
        n_modes=params.n_modes_en,
        scalings=params.scalings_en,
        lifting=True,
        projection=False,
        operator_block=block,
        re_grid_input=False,
        integral_operator=int_op,
        integral_operator_top=int_op_top,
        integral_operator_bottom=int_op_bottom,
        var_encoding=False,
        var_enco_channels=0,
        var_num=params.n_variables,
        enable_cls_token=params.enable_cls_token,
        static_channels_num=static_channels_num,
        static_features=static_features,
        per_channel_attention=params.per_channel_attention,
    )
    print("*********************")

    if params.reconstruction:
        print("Generating Decoder")
        decoder = CondnoGino(
            params.hidden_token_codim_en,
            input_grid=input_mesh,
            output_grid=output_mesh,
            radius=params.radius,
            n_neigbor=params.n_neigbor,
            fixed_neighbour=params.fixed_neighbour,
            grid_size=params.grid_size,
            gno_mlp_layers=params.gno_mlp_layers,
            hidden_token_codim=params.hidden_token_codim_en,
            lifting_token_codim=params.lifting_token_codim_en,
            out_token_codim=params.in_token_codim_en,
            n_layers=params.n_layers_dec,
            n_heads=params.n_heads_dec,
            n_modes=params.n_modes_dec,
            scalings=params.scalings_dec,
            lifting=False,
            re_grid_output=False,
            projection=True,
            operator_block=block,
            integral_operator=int_op,
            var_num=params.n_variables,
            integral_operator_top=int_op_top,
            integral_operator_bottom=int_op_bottom,
            per_channel_attention=params.per_channel_attention,
            enable_cls_token=False,  # should not add cls again in the decoder
        )
    else:
        decoder = None
    print("*********************")

    contrastive = None

    print('generating Predictor')
    predictor = CondnoGino(
        params.hidden_token_codim_en,
        input_grid=input_mesh,
        output_grid=output_mesh,
        radius=params.radius,
        n_neigbor=params.n_neigbor,
        fixed_neighbour=params.fixed_neighbour,
        grid_size=params.grid_size,
        gno_mlp_layers=params.gno_mlp_layers,
        hidden_token_codim=params.hidden_token_codim_en,
        lifting_token_codim=params.lifting_token_codim_pred,
        out_token_codim=params.out_token_codim_pred,
        n_layers=params.n_layers_pred,
        n_heads=params.n_heads_pred,
        n_modes=params.n_modes_pred,
        scalings=params.scalings_pred,
        lifting=False,
        projection=True,
        re_grid_output=False,
        operator_block=block,
        integral_operator=int_op,
        var_num=params.n_variables,
        integral_operator_top=int_op_top,
        integral_operator_bottom=int_op_bottom,
        per_channel_attention=params.per_channel_attention,
    )
    print("*********************")

    return encoder, decoder, contrastive, predictor


def get_model_fno(params):
    mesh = get_mesh(params.input_mesh_location)
    input_mesh = torch.from_numpy(mesh).type(torch.float).cuda()

    minx, maxx = np.min(mesh[:, 0]), np.max(mesh[:, 0])
    miny, maxy = np.min(mesh[:, 1]), np.max(mesh[:, 1])

    size_x, size_y = params.grid_size
    idx_x = torch.arange(start=minx,
                         end=maxx + (maxx - minx) / size_x - 1e-5,
                         step=(maxx - minx) / (size_x - 1))
    idx_y = torch.arange(start=miny,
                         end=maxy + (maxy - miny) / size_y - 1e-5,
                         step=(maxy - miny) / (size_y - 1))
    x, y = torch.meshgrid(idx_x, idx_y, indexing='ij')
    output_mesh = torch.transpose(
        torch.stack([x.flatten(), y.flatten()]),
        dim0=0,
        dim1=1,
    ).type(torch.float).cuda()

    assert x.shape[0] == size_x
    assert x.shape[1] == size_y

    # block = None
    block = FNOBlocks

    if params.tno_integral_op == 'fno':
        int_op = partial(SpectralConvKernel2d,
                         transform_type=params.transform_type,
                         frequency_mixer=False)
        int_op_top = int_op
        int_op_bottom = int_op
    elif params.tno_integral_op == 'fino':
        int_op = partial(SpectralConvKernel2d,
                         transform_type=params.transform_type)
        int_op_top = int_op
        int_op_bottom = int_op
    else:
        raise (Exception('Int. Op. config Error'))
    print("Generating Encoder")
    in_dim = params.in_dim + params.n_static_channels
    if params.grid_type == 'uniform':
        model = FNO(
            params.n_modes,
            params.hidden_dim,
            in_channels=in_dim,
            out_channels=params.out_dim,
            lifting_channels=params.lifting_dim,
            projection_channels=params.projection_dim,
            n_layers=params.n_layers,
            output_scaling_factor=params.scalings,
            SpectralConv=int_op,
        )
    else:
        if params.nettype == 'gnn':
            model = GNN(
                in_dim,
                params.out_dim,
                input_grid=input_mesh,
                output_grid=output_mesh,
                n_neigbor=params.n_neigbor,
                gno_mlp_layers=params.gno_mlp_layers,
                hidden_dim=params.hidden_dim,
                lifting_dim=params.lifting_dim,
                initial_mesh=input_mesh,
                n_layers=params.n_layers,
                lifting=True,
                projection=True,
            )
        elif params.nettype == 'deeponet':
            model = DeepONet(
                in_dim,
                params.out_dim,
                input_grid=input_mesh,
                output_grid=output_mesh,
                branch_layers=params.branch_layers,
                trunk_layers=params.trunk_layers,
                initial_mesh=input_mesh,
                n_neigbor=params.n_neigbor,
                gno_mlp_layers=params.gno_mlp_layers,
            )
        elif params.nettype == 'vit':
            model = VitGno(
                in_dim,
                params.out_dim,
                input_grid=input_mesh,
                output_grid=output_mesh,
                radius=params.radius,
                gno_mlp_layers=params.gno_mlp_layers,
                hidden_dim=params.hidden_dim,
                lifting_dim=params.lifting_dim,
                n_layers=params.n_layers,
                grid_size = tuple(params.grid_size),
                patch_size = tuple(params.patch_size),
                heads = params.heads,
                initial_mesh=input_mesh,
                lifting=True,
                projection=True,
                re_grid_input=False,
            )
        else:
            model = FnoGno(
                in_dim,
                params.out_dim,
                input_grid=input_mesh,
                output_grid=output_mesh,
                radius=params.radius,
                gno_mlp_layers=params.gno_mlp_layers,
                grid_size=params.grid_size,
                hidden_dim=params.hidden_dim,
                lifting_dim=params.lifting_dim,
                n_layers=params.n_layers,
                n_modes=params.n_modes,
                scalings=params.scalings,
                initial_mesh=input_mesh,
                lifting=True,
                projection=True,
                operator_block=block,
                re_grid_input=False,
                integral_operator=int_op,
                integral_operator_top=int_op_top,
                integral_operator_bottom=int_op_bottom,
            )

    return model

    return model


class StageEnum(enum.Enum):
    RECONSTRUCTIVE = "RECONSTRUCTIVE"
    PREDICTIVE = "PREDICTIVE"


class SSLWrapper(nn.Module):
    """Unlike the other wrapper, this takes an initialized model."""

    def __init__(
        self,
        params,
        encoder,
        decoder,
        contrastive,
        predictor,
        # variables_per_equations: Dict[Equation, int] = None,
        # n_encoding_channels: int = 1,
        n_static_channels: int = 0,
        stage=StageEnum.PREDICTIVE,
        logger=Optional[logging.Logger],
    ):
        super(SSLWrapper, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.contrastive = contrastive
        self.predictor = predictor

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

        self.reconstruction = params.reconstruction
        self.next_channels: Optional[Tuple[Tuple[int]]] = None
        self.last_masks: Optional[torch.Tensor] = None

        self.enable_cls_token = params.enable_cls_token

        self.stage = stage
        self.freeze_encoder = params.freeze_encoder
        self.masking = params.masking
        self.grid_type = params.grid_type
        # grid type is either uniform or non uniform
        # uniform == PDE bench dataset
        # non uniform == NS Elastc dataset or archtectures with GNO layers

        if self.grid_type == 'uniform':  # TODO add a different option for this
            variables_per_equation = params.variables_per_equation
            if params.variables_per_equation is None:
                variables_per_equation = {}
            equation_to_encoders = {eq: [] for eq in variables_per_equation}

            v = 0
            for eq, size in variables_per_equation.items():
                for _ in range(size):
                    equation_to_encoders[eq] = equation_to_encoders[eq] + [v]
                    v += 1
            self.equation_to_encoders: Dict[Equation, Tuple[int, ...]] = \
                {k: tuple(v) for k, v in equation_to_encoders.items()}

            # TODO support multiple encoders (?)
            self.n_encoding_channels = params.n_encoding_channels
            self.variable_encoders = [
                FourierVariableEncoding3D(
                    self.n_encoding_channels,
                    (
                        params.encoding_modes_x,
                        params.encoding_modes_y,
                        params.encoding_modes_t,
                    ),
                ) for _ in range(v)
            ]
            self.n_static_channels = n_static_channels

        if params.grid_type == 'uniform':
            Masker = MaskerUniformIndependent if params.time_axis else MaskerUniform
            self.augmenter_masker = Masker(
                drop_type=params.drop_type,
                max_block=params.max_block,
                drop_pix=params.drop_pix,
                channel_per=params.channel_per,
                channel_drop_per=params.channel_drop_per,
            )

            self.validation_augmenter = Masker(
                drop_type=params.drop_type,
                max_block=params.max_block_val,
                drop_pix=params.drop_pix_val,
                channel_per=params.channel_per_val,
                channel_drop_per=params.channel_drop_per_val,
            )
        else:
            self.augmenter_masker = MaskerNonuniformMesh(
                grid_non_uni=encoder.input_grid.clone().detach(),
                gird_uni=encoder.output_grid.clone().detach(),
                radius=params.masking_radius,
                drop_type=params.drop_type,
                drop_pix=params.drop_pix,
                channel_aug_rate=params.channel_per,
                channel_drop_rate=params.channel_drop_per,
            )

            # XXX unused in testing
            # If following augmenter is used by external method during testing
            self.validation_augmenter = MaskerNonuniformMesh(
                grid_non_uni=encoder.input_grid.clone().detach(),
                gird_uni=encoder.output_grid.clone().detach(),
                radius=params.masking_radius,
                drop_type=params.drop_type,
                max_block=params.max_block_val,
                drop_pix=params.drop_pix_val,
                channel_aug_rate=params.channel_per_val,
                channel_drop_rate=params.channel_drop_per_val,
            )

    def reset_channels(self):
        # TODO add a setting where `next_channels` have some persistence.
        self.next_channels = None

    def set_initial_mesh(self, mesh):
        # self.register_buffer('initial_mesh', mesh)
        self.initial_mesh = mesh

    # NOTE: this should support both 3D and 2D models
    # only used for uniform grids
    def _encode_variable(self, x: torch.Tensor, encoder):
        """Applies variable encodings to given input variable.

        Exactly 1 variable channel is expected in a squeezed "axis" (i.e.
        the variable should not occupy a singleton axis).

        Transform the low-dimensional input tensor to a higher-dimensional
        representation where the variable has been augmented with both
        learned encodings and static channels (where the latter may be
        positional encoding, etc.)

        Returns a Tensor like:
        ``(batch_size, token_size, *domain_size)`` for:
        ``batch_size, *domain_size = x.shape`` and:
        ``token_size = 1 + self.n_encoding_channels + self.n_static_channels``
        """
        # Token dimensionality supersedes channel dimensionality:
        batch_size, *domain_size = x.shape
        token_size = 1 + self.n_encoding_channels + self.n_static_channels
        encoding_channels = [n + 1 for n in range(self.n_encoding_channels)]
        static_channels = [n + len(encoding_channels)
                           for n in range(self.n_static_channels)]

        y = torch.zeros(
            (batch_size, token_size, *domain_size),
            device=x.device,
            dtype=x.dtype,
        )
        # y[:, 0, ...] = x.unsqueeze(1)
        y[:, 0, ...] = x

        variable_encoding = encoder(x.shape).to(x.device)
        # self.logger.debug(f"{variable_encoding.shape=}")
        # Unsqueeze variable encoding in 0th dimension (indexed by `None` below)
        # and repeat it until it matches the dimension of ``batch_size``
        r_size = [batch_size, 1] + [1 for _ in domain_size]
        y[:, encoding_channels, ...] = variable_encoding[None, ...].repeat(
            *r_size)

        if self.n_static_channels > 0:
            # Repeat `static_features` as many times as we can
            # within the space of `static_channels`
            n_static_copies = self.n_static_channels // self.static_features.shape[1]
            r_size = [batch_size, n_static_copies] + [1 for _ in domain_size]
            y[:, static_channels, ...] = self.static_features.repeat(*r_size)

        return y

    # only used for uniform grids
    def encode_variables(self, x: torch.Tensor, encoders):
        """Encodes each (physical) variable channel with an encoder.

        Assumes the input ``x`` is shaped like:
        ``(batch_size, n_variables, *domain_size)``

        Then the 0th variable ``x[:, 0]`` will be encoded with the 0th encoder
        ``encoders[0]``, the 1st variable will be encoded with the 1st encoder,
        and so forth.

        Returns:
            Input x "augmented" with the encoding of each variable. The returned
            Tensor is shaped like:
            ``(batch_size, n_variables * token_size, *domain_size)`` for:
            ``batch_size, *domain_size = x.shape`` and:
            ``token_size = 1 + self.n_encoding_channels + self.n_static_channels``
        """
        xs = [self._encode_variable(x[:, i], encoder)
              for i, encoder in enumerate(encoders)]
        return torch.concatenate(xs, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        static_random_tensor=None,
        # ``Tensor`` contains wrapped ``int`` corresponding to ``Equation`` enum:
        equations: Optional[torch.Tensor] = None,
        out_grid_displacement=None,
        in_grid_displacement=None
    ):
        if self.grid_type == 'uniform':
            # used for PDE bench datasets
            if equations is None or len(equations) == 0:
                raise ValueError(
                    "The equation(s) must be defined for a variable encoding.")

            _equations: Set[int] = {eq.item() for eq in equations}
            if len(_equations) > 1:
                raise ValueError(
                    "All equations in a batch must be of the same kind. "
                    f"Received multiple different kinds in one batch: {_equations=}")

            equation: Equation = Equation(_equations.pop())
            # TODO add support for arbitrary sets of variables.
            # This would be especially useful for mixed-physics scenarios.
            encoders_idxs = self.equation_to_encoders[equation.value]
            encoders = [self.variable_encoders[idx] for idx in encoders_idxs]
            x_embedded = self.encode_variables(x, encoders)
        else:
            x_embedded = x

        if self.stage == StageEnum.RECONSTRUCTIVE:
            return self.forward_reconstructive(
                x_embedded,
                in_grid_displacement,
                out_grid_displacement,
            )

        if self.stage == StageEnum.PREDICTIVE:
            #print(in_grid_displacement, out_grid_displacement)
            return self.forward_predictive(
                x_embedded,
                in_grid_displacement,
                out_grid_displacement,
            )


        raise ValueError(f'Expected stage to be one of {list(StageEnum)};\n'
                         f'Got {self.stage=}')

    def do_mask(self, x):
        if not self.masking:
            return x

        with torch.no_grad():
            x_masked, masks = batched_masker(
                x,
                self.augmenter_masker,
                batched_channels=self.next_channels,
            )
        self.last_masks = masks.type(torch.int8)  # for reconstructing on loss
        # Enforce that `next_channels` must be set before every forward call:
        self.reset_channels()

        return x_masked

    # Assumes `x` is already embedded in its higher-dimensional repr:
    def forward_reconstructive(
        self,
        x,
        in_grid_displacement=None,
        out_grid_displacement=None,
    ):
        # adjusting for chnage of mesh
        if self.grid_type != "uniform":
            with torch.no_grad():
                # updating neigbors of GNO layers for the new mesh at each time step
                self.encoder.lifting.update_grid(
                    self.initial_mesh + in_grid_displacement, None)
                self.decoder.projection.update_grid(
                    None, self.initial_mesh + out_grid_displacement)

        x_masked = self.do_mask(x)
        x_encoded = self.encoder(x_masked)
        # print("Feature Shape", x_encoded.shape)

        cls_offset = 1 if self.enable_cls_token else 0
        if self.reconstruction:
            reconstructed = self.decoder(x_encoded)

            # Removing the CLS token and also discarding if some additional
            # channels if in the end
            if self.grid_type == 'uniform':
                _slice = [
                    slice(None),  # :
                    slice(cls_offset, cls_offset + x.shape[1]),
                    slice(None),  # :
                    slice(None),  # :
                ]
            else:
                _slice = [
                    slice(None),  # :
                    slice(None),  # :
                    slice(cls_offset, None),
                ]
            reconstructed = reconstructed[_slice]
        else:
            reconstructed = None

        # reconstructed, aug_contra  = self.model(x_masked, 'ssl')
        # print("Model Forward done")

        # Placeholders for contrastive losses. Not used for now.
        clean_contra = None
        neg_contra = None
        aug_contra = None
        # print(reconstructed.shape, _slice)

        return reconstructed, clean_contra, aug_contra, neg_contra

    def encode_input(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(x)
        else:
            return self.encoder(x)

    def decode_output(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                return self.decoder(x)
        else:
            return self.decoder(x)

    # Assumes `x` is already embedded in its higher-dimensional repr:
    def forward_predictive(
        self,
        x: torch.Tensor,
        in_grid_displacement=None,
        out_grid_displacement=None,
    ):
        if self.grid_type != 'uniform':
            with torch.no_grad():
                # updating neigbors of GNO layers for the new mesh at each time step
                self.encoder.lifting.update_grid(
                    self.initial_mesh + in_grid_displacement, None)
                self.predictor.projection.update_grid(
                    None, self.initial_mesh + out_grid_displacement)

        x_encoded = self.encode_input(x)
        out = self.predictor(x_encoded)

        if self.reconstruction and self.grid_type == 'uniform':
            out = self.decode_output(out)

        cls_offset = 1 if self.enable_cls_token else 0
        # discarding CLS token and additional static channels if added.
        # channel dimention is different for uniform and non uniform grids
        # i.e. channel first/last data format
        if self.grid_type == 'uniform':
            _slice = [
                slice(None),  # :
                slice(cls_offset, cls_offset + x.shape[1]),
                slice(None),  # :
                slice(None),  # :
            ]
        else:
            _slice = [
                slice(None),  # :
                slice(None),  # :
                slice(cls_offset, None),
            ]
        out = out[_slice]

        return out, None, None, None
