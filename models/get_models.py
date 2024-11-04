import enum
from functools import partial
import logging
from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from layers.unet_sublayer import UNet2d
from utils import *
from neuralop.layers.fno_block import FNOBlocks
from neuralop.models import FNO
from layers.codano_block_nd import CodanoBlockND
from data_utils.data_utils import MaskerNonuniformMesh, MaskerUniform, batched_masker
from layers.codano_block_2D import *
from layers.fino_2D import SpectralConvKernel2d
from layers.fino_nd import SpectralConvKernel2d as SPCONV
from models.codano import CodANO, VariableEncodingArgs
from models.codano_gino import CondnoGino
from models.fno_gino import FnoGno
from models.gnn import GNN
from models.deeponet import DeepONet
from models.vit import VitGno
from models.unet import UnetGno

# Uniform Codano


def get_ssl_models_codano(
    params,
    verbose=True,
):
    # block = None
    block = CodanoBlocks2d

    if params.tno_integral_op == 'fno':
        int_op = partial(SpectralConvKernel2d,
                         transform_type=params.transform_type,
                         frequency_mixer=False)

    elif params.tno_integral_op == 'fino':
        int_op = partial(SpectralConvKernel2d,
                         transform_type=params.transform_type)
    else:
        raise (Exception('Int. Op. config Error'))
    print("Generating Encoder")

    static_features = None
    static_channels_num = params.n_static_channels

    print("Token Dim-->", 1 + params.n_encoding_channels + static_channels_num)
    expanded_token_dim = 1 + params.n_encoding_channels + static_channels_num

    common_args = dict(
        operator_block=block,
        n_variables=params.n_variables,
        integral_operator=int_op,
        integral_operator_top=int_op,
        integral_operator_bottom=int_op,
        per_channel_attention=params.per_channel_attention,
        variable_encoding_args=VariableEncodingArgs(
            basis="fft",
            n_channels=params.n_encoding_channels,
            modes_x=params.encoding_modes_x,
            modes_y=params.encoding_modes_y,
            modes_t=params.encoding_modes_t,
        ),
    )

    encoder = CodANO(
        expanded_token_dim,
        hidden_token_codimension=params.hidden_token_codim_en,
        lifting_token_codimension=params.lifting_token_codim_en,
        n_layers=params.n_layers_en,
        n_heads=params.n_heads_en,
        n_modes=params.n_modes_en,
        max_n_modes=params.max_n_modes_en,
        scalings=params.scalings_en,
        lifting=True,
        projection=False,
        enable_cls_token=params.enable_cls_token,
        **common_args,
    )
    print("*********************")

    if params.reconstruction:
        print("Generating Decoder")
        decoder = CodANO(
            params.hidden_token_codim_en,
            hidden_token_codimension=params.hidden_token_codim_en,
            lifting_token_codimension=params.lifting_token_codim_en,
            output_token_codimension=params.in_token_codim_en,
            n_layers=params.n_layers_dec,
            n_heads=params.n_heads_dec,
            n_modes=params.n_modes_dec,
            max_n_modes=params.max_n_modes_dec,
            scalings=params.scalings_dec,
            lifting=False,
            projection=True,
            enable_cls_token=False,  # should not add cls again in the decoder
            **common_args,
        )
    else:
        decoder = None
    print("*********************")

    contrastive = None

    print('generating Predictor')
    predictor = CodANO(
        params.hidden_token_codim_en,
        hidden_token_codimension=params.hidden_token_codim_en,
        lifting_token_codimension=params.lifting_token_codim_pred,
        output_token_codimension=params.out_token_codim_pred,
        n_layers=params.n_layers_pred,
        n_heads=params.n_heads_pred,
        n_modes=params.n_modes_pred,
        max_n_modes=params.max_n_modes_pred,
        scalings=params.scalings_pred,
        lifting=False,
        projection=True,
        enable_cls_token=False,
        **common_args,
    )
    print("*********************")

    return encoder, decoder, contrastive, predictor


'''
non-uniform codano.
'''


def get_ssl_models_codano_gino(params):
    # We use tno inside SSLtransformer model. That has a encoder and prediction/(decoder) part.
    # Encoder part - encodes the input function
    # Decoder part - does the prediction (eg. fuild flow of next time step)
    # For SLL it has a reconstruction head and a dense contarstive head

    # read input grid and prepare uniform grid
    mesh = get_mesh(params)
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
    block = CodanoBlocks2d

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


'''
Following is for getting all the baselines
'''
def get_baseline_model(params):
    if params.grid_type == 'uniform':
        '''
        Models for uniform grid
        '''
        if params.nettype == 'fno':
            model = FNO(
                n_modes=params.n_modes,
                max_n_modes=params.max_n_modes,
                hidden_channels=params.hidden_dim,
                in_channels=params.in_dim,
                out_channels=params.out_dim,
                lifting_channels=params.lifting_dim,
                projection_channels=params.projection_dim,
                n_layers=params.n_layers,
            )
        elif params.nettype == 'unet':
            model = UNet2d(in_channels=params.in_dim,
                           out_channels=params.out_dim,
                           init_features=params.init_features,
                           )
        elif params.nettype == 'deeponet':
            raise (Exception('Not Implemented'))

        return model

    else:
        if params.input_mesh_location is not None:
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

            # General modules of FNO blocks and
            # integral operators
            block = FNOBlocks
            if params.tno_integral_op == 'fno':
                int_op = partial(SpectralConvKernel2d,
                                 transform_type=params.transform_type,
                                 frequency_mixer=False)
            elif params.tno_integral_op == 'fino':
                int_op = partial(SpectralConvKernel2d,
                                 transform_type=params.transform_type)
            else:
                raise (Exception('Int. Op. Not found'))
            
            in_dim = params.in_dim + params.n_static_channels

            if params.nettype == 'gnn':
                print("Generating GNN")
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
                    grid_size=tuple(params.grid_size),
                    patch_size=tuple(params.patch_size),
                    heads=params.heads,
                    initial_mesh=input_mesh,
                    lifting=True,
                    projection=True,
                    re_grid_input=False,
                )
            elif params.nettype == 'unet':
                model = UnetGno(
                    in_dim=params.in_dim,
                    out_dim=params.out_dim,
                    input_grid=params.input_mesh,
                    output_grid=params.output_mesh,
                    grid_size=tuple(params.grid_size),
                    radius=params.radius,
                    gno_mlp_layers=params.gno_mlp_layers,
                    hidden_dim=params.hidden_dim,
                    lifting_dim=params.lifting_dim,
                    n_layers=params.n_layers,
                    pad_to_size=params.pad_to_size,
                    initial_mesh=None,
                    lifting=False,
                    projection=False,
                    re_grid_input=False,
                )
            else:
                model = FnoGno(
                    in_dim=params.in_dim,
                    out_dim=params.out_dim,
                    input_grid=input_mesh,
                    output_grid=output_mesh,
                    radius=params.radius,
                    gno_mlp_layers=params.gno_mlp_layers,
                    grid_size=params.grid_size,
                    hidden_dim=params.hidden_dim,
                    lifting_dim=params.lifting_dim,
                    n_layers=params.n_layers,
                    max_n_modes=params.max_n_modes,
                    n_modes=params.n_modes,
                    scalings=params.scalings,
                    initial_mesh=input_mesh,
                    lifting=True,
                    projection=True,
                    operator_block=block,
                    re_grid_input=False,
                    integral_operator=int_op,
                    integral_operator_top=int_op,
                    integral_operator_bottom=int_op,
                )
            return model
        else:
            raise (Exception('Input mesh location not provided'))


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
        stage=StageEnum.PREDICTIVE,
        logger=Optional[logging.Logger],
    ):
        super(SSLWrapper, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.contrastive = contrastive
        self.predictor = predictor
        self.params = params

        if logger is None:
            logger = logging.getLogger()
        self.logger = logger

        self.next_channels: Optional[Tuple[Tuple[int]]] = None
        self.last_masks: Optional[torch.Tensor] = None

        self.enable_cls_token = params.enable_cls_token

        self.stage = stage
        self.freeze_encoder = params.freeze_encoder
        self.masking = params.masking
        self.grid_type = params.grid_type

        if params.grid_type == 'uniform':
            self.augmenter_masker = MaskerUniform(
                drop_type=params.drop_type,
                max_block=params.max_block,
                drop_pix=params.drop_pix,
                channel_per=params.channel_per,
                channel_drop_per=params.channel_drop_per,
            )
            # sperately creat validation augmenter
            # to test model's performnace on
            # missing and partially observed variables.

            self.validation_augmenter = MaskerUniform(
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

    def set_initial_mesh(self, mesh):
        self.initial_mesh = mesh

    def forward(
        self,
        x: torch.Tensor,
        out_grid_displacement=None,
        in_grid_displacement=None
    ):
        """
        x: torch.tensor. 
         For non-uniform mesh the tensor is of shape (batch, number of points on the mesh, number of channels/variables )
         For uniform mesh, x is of shape (batch, channels, H, W)
        """
        x_embedded = x
        if self.stage == StageEnum.RECONSTRUCTIVE:
            # Self supervised learning scheme
            return self.forward_reconstructive(
                x_embedded,
                in_grid_displacement,
                out_grid_displacement,
            )

        if self.stage == StageEnum.PREDICTIVE:
            # supervised learning scheme
            return self.forward_predictive(
                x_embedded,
                in_grid_displacement,
                out_grid_displacement,
            )

        raise ValueError(f'Expected stage to be one of {list(StageEnum)};\n'
                         f'Got {self.stage=}')

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
                # updating neigbors of GNO layers for the new mesh at each time
                # step
                self.encoder.lifting.update_grid(
                    self.initial_mesh + in_grid_displacement, None)
                self.decoder.projection.update_grid(
                    None, self.initial_mesh + out_grid_displacement)

        x_encoded = self.encoder(x)
        cls_offset = 1 if self.enable_cls_token else 0

        reconstructed = self.decoder(x_encoded)

        # Removing the CLS token
        if self.grid_type == 'uniform':
            _slice = [
                slice(None),  # :
                slice(cls_offset, cls_offset + x.shape[1]),
                slice(None),  # :
                slice(None),  # :
            ]
        else:
            # tokens are at the last dimensioms
            # for non-uniform mesh
            _slice = [
                slice(None),  # :
                slice(None),  # :
                slice(cls_offset, None),
            ]
        reconstructed = reconstructed[_slice]

        return reconstructed
    def do_mask(self, x):
        if not self.masking:
            return x
        with torch.no_grad():
            x_masked, masks = batched_masker(
                x,
                self.augmenter_masker,
                batched_channels=self.next_channels,
            )

        return x_masked

    def encode_input(self, x):
        if self.freeze_encoder:
            with torch.no_grad():
                return self.encoder(x)
        else:
            return self.encoder(x)

    # Assumes `x` is already embedded in its higher-dimensional repr:
    def forward_predictive(
        self,
        x: torch.Tensor,
        in_grid_displacement=None,
        out_grid_displacement=None,
    ):
        if self.grid_type != 'uniform':
            with torch.no_grad():
                # updating neigbors of GNO layers for the new mesh at each time
                # step
                self.encoder.lifting.update_grid(
                    self.initial_mesh + in_grid_displacement, None)
                self.predictor.projection.update_grid(
                    None, self.initial_mesh + out_grid_displacement)

        x_encoded = self.encode_input(x.clone())
        out = self.predictor(x_encoded)

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

        return out
