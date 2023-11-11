from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from data_utils.data_utils import MakserNonuniformMest, batched_masker, MaskerUniform
import torch.nn as nn
from models.codano_gino import CondnoGino
from models.fno_gino import FnoGno
from functools import partial
from neuralop.layers.fno_block import FNOBlocks
from models.codano import CodANO
import torch
import numpy as np
from neuralop.models import FNO

def get_ssl_models_codaNo(params):
    ## We use tno inside SSLtransformer model. That has a encoder and prediction/(decoder) part. 
    ## Encoder part - encodes the input function
    ## Decoder part - does the prediction (eg. fuild flow of next time step)
    ## For SLL it has a reconstruction head and a dense contarstive head

    block = None
    block = TnoBlock2d

    if params.tno_integral_op == 'fno':
        int_op = partial(SpectralConvKernel2d, fft_type = params.fft_type, frequency_mixer = False)
        int_op_top = int_op
        int_op_bottom = int_op
    elif params.tno_integral_op == 'fino':
        int_op = partial(SpectralConvKernel2d, fft_type = params.fft_type)
        int_op_top = int_op
        int_op_bottom = int_op
    else:
        raise( Exception('Int. Op. config Error'))

    print("Generating Encoder")
    
    static_features = None
    static_channels_num = 0
    
    if params.add_static_feature:
        # taking the static features from  which will be passed to the encoder
        # to concated with each of the input varibales
        static_features = None
        static_channels_num = 0
        
    print("Token Dim-->", 1+params.var_enco_channels+static_channels_num)
    print("var num", params.var_num, "static channels", static_channels_num)

    encoder = CodANO(params.in_token_codim_en,
                    hidden_token_codim=params.hidden_token_codim_en,
                    lifting_token_codim=params.lifting_token_codim_en,
                    n_layers=params.n_layers_en,
                    n_heads = params.n_heads_en,
                    n_modes = params.n_modes_en,
                    scalings=params.scalings_en,
                    lifting=True,
                    projection=False,
                    operator_block = block,
                    re_grid_input=False,
                    integral_operator=int_op,
                    integral_operator_top=int_op_top,
                    integral_operator_bottom=int_op_bottom,
                    var_encoding=params.var_encoding,
                    var_enco_channels = params.var_enco_channels,
                    var_num=params.var_num,
                    enable_cls_token=params.enable_cls_token,
                    static_channels_num=static_channels_num,
                    static_features=static_features,
                    per_channel_attention=params.per_channel_attention
            )
    print("*********************")

    if params.enable_cls_token:
        count = 1
    else:
        count = 0
        
    if params.reconstruction:
        print("Generating Decoder")
        decoder = CodANO(params.hidden_token_codim_en,
                        hidden_token_codim=params.hidden_token_codim_en,
                        lifting_token_codim=params.lifting_token_codim_en,
                        out_token_codim=params.in_token_codim_en,
                        n_layers=params.n_layers_dec,
                        n_heads = params.n_heads_dec,
                        n_modes = params.n_modes_dec,
                        scalings=params.scalings_dec,
                        lifting=False,
                        re_grid_output=False,
                        projection=True,
                        operator_block = block,
                        integral_operator=int_op,
                        var_num=params.var_num,
                        integral_operator_top=int_op_top,
                        integral_operator_bottom=int_op_bottom,
                        per_channel_attention=params.per_channel_attention
        )
    else:
        decoder = None
    print("*********************")
 
    contrastive = None

    print('generating Predictor')
    predictor = CodANO(params.hidden_token_codim_en,
                    hidden_token_codim=params.hidden_token_codim_en,
                    lifting_token_codim=params.lifting_token_codim_pred,
                    out_token_codim=params.out_token_codim_pred,
                    n_layers=params.n_layers_pred,
                    n_heads = params.n_heads_pred,
                    n_modes = params.n_modes_pred,
                    scalings=params.scalings_pred,
                    lifting=False,
                    projection=True,
                    re_grid_output=False,
                    operator_block = block,
                    integral_operator=int_op,
                    var_num=params.var_num,
                    integral_operator_top=int_op_top,
                    integral_operator_bottom=int_op_bottom,
                    per_channel_attention=params.per_channel_attention
    )
    print("*********************")
    
    return encoder, decoder, contrastive, predictor

def get_ssl_models_codano_gino(params):
    ## We use tno inside SSLtransformer model. That has a encoder and prediction/(decoder) part. 
    ## Encoder part - encodes the input function
    ## Decoder part - does the prediction (eg. fuild flow of next time step)
    ## For SLL it has a reconstruction head and a dense contarstive head

    # read input grid and prepare uniform grid
    mesh = np.loadtxt(params.input_mesh_location, delimiter=',')
    input_mesh = torch.transpose(torch.stack([torch.tensor(mesh[0,:]),\
                                        torch.tensor(mesh[1,:])]), 0, 1).type(torch.float).cuda()
    
    minx, maxx = np.min(mesh[0,:]), np.max(mesh[0,:])
    miny, maxy = np.min(mesh[1,:]), np.max(mesh[1,:])

    size_x, size_y = params.grid_size
    idx_x = torch.arange(start=minx,end=maxx + (maxx - minx)/size_x - 1e-5, step=(maxx - minx)/(size_x-1))
    idx_y = torch.arange(start=miny,end=maxy + (maxy - miny)/size_y - 1e-5, step=(maxy - miny)/(size_y-1))
    x, y  = torch.meshgrid(idx_x, idx_y, indexing='ij')
    output_mesh = torch.transpose(torch.stack([x.flatten(), y.flatten()]), 0, 1).type(torch.float).cuda()

    assert x.shape[0] == size_x
    assert x.shape[1] == size_y

    block = None
    block = TnoBlock2d

    if params.tno_integral_op == 'fno':
        int_op = partial(SpectralConvKernel2d, fft_type = params.fft_type, frequency_mixer = False)
        int_op_top = int_op
        int_op_bottom = int_op
    elif params.tno_integral_op == 'fino':
        int_op = partial(SpectralConvKernel2d, fft_type = params.fft_type)
        int_op_top = int_op
        int_op_bottom = int_op
    else:
        raise( Exception('Int. Op. config Error'))
    print("Generating Encoder")
    
    static_features = None
    static_channels_num = 0
    
    if params.add_static_feature:
        # taking the static features from  which will be passed to the encoder
        # to concated with each of the input varibales
        static_features = None
        static_channels_num = 0
        
    print("Token Dim-->", 1+params.var_enco_channels+static_channels_num)
    print("var num", params.var_num, "static channels", static_channels_num)

    encoder = CondnoGino(params.in_token_codim_en,
                    input_grid= input_mesh,
                    output_grid= output_mesh,
                    radius=params.radius,
                    gno_mlp_layers=params.gno_mlp_layers,
                    grid_size=params.grid_size,
                    hidden_token_codim=params.hidden_token_codim_en,
                    lifting_token_codim=params.lifting_token_codim_en,
                    n_layers=params.n_layers_en,
                    n_heads = params.n_heads_en,
                    n_modes = params.n_modes_en,
                    scalings=params.scalings_en,
                    lifting=True,
                    projection=False,
                    operator_block = block,
                    re_grid_input=False,
                    integral_operator=int_op,
                    integral_operator_top=int_op_top,
                    integral_operator_bottom=int_op_bottom,
                    var_encoding=params.var_encoding,
                    var_enco_channels = params.var_enco_channels,
                    var_num=params.var_num,
                    enable_cls_token=params.enable_cls_token,
                    static_channels_num=static_channels_num,
                    static_features=static_features,
                    per_channel_attention=params.per_channel_attention
            )
    print("*********************")

    if params.enable_cls_token:
        count = 1
    else:
        count = 0
        
    if params.reconstruction:
        print("Generating Decoder")
        decoder = CondnoGino(params.hidden_token_codim_en,
                        input_grid= input_mesh,
                        output_grid= output_mesh,
                        radius=params.radius,
                        grid_size=params.grid_size,
                        gno_mlp_layers=params.gno_mlp_layers,
                        hidden_token_codim=params.hidden_token_codim_en,
                        lifting_token_codim=params.lifting_token_codim_en,
                        out_token_codim=params.in_token_codim_en,
                        n_layers=params.n_layers_dec,
                        n_heads = params.n_heads_dec,
                        n_modes = params.n_modes_dec,
                        scalings=params.scalings_dec,
                        lifting=False,
                        re_grid_output=False,
                        projection=True,
                        operator_block = block,
                        integral_operator=int_op,
                        var_num=params.var_num,
                        integral_operator_top=int_op_top,
                        integral_operator_bottom=int_op_bottom,
                        per_channel_attention=params.per_channel_attention
        )
    else:
        decoder = None
    print("*********************")
 
    contrastive = None

    print('generating Predictor')
    predictor = CondnoGino(params.hidden_token_codim_en,
                    input_grid= input_mesh,
                    output_grid= output_mesh,
                    radius=params.radius,
                    grid_size=params.grid_size,
                    gno_mlp_layers=params.gno_mlp_layers,
                    hidden_token_codim=params.hidden_token_codim_en,
                    lifting_token_codim=params.lifting_token_codim_pred,
                    out_token_codim=params.out_token_codim_pred,
                    n_layers=params.n_layers_pred,
                    n_heads = params.n_heads_pred,
                    n_modes = params.n_modes_pred,
                    scalings=params.scalings_pred,
                    lifting=False,
                    projection=True,
                    re_grid_output=False,
                    operator_block = block,
                    integral_operator=int_op,
                    var_num=params.var_num,
                    integral_operator_top=int_op_top,
                    integral_operator_bottom=int_op_bottom,
                    per_channel_attention=params.per_channel_attention
    )
    print("*********************")
    
    return encoder, decoder, contrastive, predictor

def get_model_fno(params):
    mesh = np.loadtxt(params.input_mesh_location, delimiter=',')
    input_mesh = torch.transpose(torch.stack([torch.tensor(mesh[0,:]),\
                                        torch.tensor(mesh[1,:])]), 0, 1).type(torch.float).cuda()
    
    minx, maxx = np.min(mesh[0,:]), np.max(mesh[0,:])
    miny, maxy = np.min(mesh[1,:]), np.max(mesh[1,:])

    size_x, size_y = params.grid_size
    idx_x = torch.arange(start=minx,end=maxx + (maxx - minx)/size_x - 1e-5, step=(maxx - minx)/(size_x-1))
    idx_y = torch.arange(start=miny,end=maxy + (maxy - miny)/size_y - 1e-5, step=(maxy - miny)/(size_y-1))
    x, y  = torch.meshgrid(idx_x, idx_y, indexing='ij')
    output_mesh = torch.transpose(torch.stack([x.flatten(), y.flatten()]), 0, 1).type(torch.float).cuda()

    assert x.shape[0] == size_x
    assert x.shape[1] == size_y

    block = None
    block = FNOBlocks

    if params.tno_integral_op == 'fno':
        int_op = partial(SpectralConvKernel2d, fft_type = params.fft_type, frequency_mixer = False)
        int_op_top = int_op
        int_op_bottom = int_op
    elif params.tno_integral_op == 'fino':
        int_op = partial(SpectralConvKernel2d, fft_type = params.fft_type)
        int_op_top = int_op
        int_op_bottom = int_op
    else:
        raise( Exception('Int. Op. config Error'))
    print("Generating Encoder")
    if params.grid_type == 'uniform':

        model = FNO(params.n_modes,
                    params.hidden_dim,
                    in_channels=params.in_dim,
                    out_channels=params.out_dim,
                    lifting_channels=params.lifting_dim,
                    projection_channels=params.projection_dim,
                    n_layers=params.n_layers,
                    output_scaling_factor=params.scalings,
                    SpectralConv=int_op)
    else:
        model = FnoGno(params.in_dim,
                        params.out_dim,
                        input_grid=input_mesh,
                        output_grid=output_mesh,
                        initial_mesh=input_mesh,
                        radius=params.radius,
                        gno_mlp_layers=params.gno_mlp_layers,
                        grid_size=params.grid_size,
                        hidden_dim=params.hidden_dim,
                        lifting_dim=params.lifting_dim,
                        n_layers=params.n_layers,
                        n_modes = params.n_modes,
                        scalings=params.scalings,
                        lifting=True,
                        projection=True,
                        operator_block = block,
                        re_grid_input=False,
                        integral_operator=int_op,
                        integral_operator_top=int_op_top,
                        integral_operator_bottom=int_op_bottom,
                )
    
    return model



class SslWrapper(nn.Module):
    '''
    unlike other wrapper, this take initialized model
    '''
    def __init__(self, params, encoder, decoder, contrastive, predictor, stage):
        super(SslWrapper, self).__init__()
    
        self.encoder = encoder
        self.decoder = decoder
        self.contrastive = contrastive
        self.predictor = predictor

        self.reconstruction = params.reconstruction
        self.enable_cls_token = params.enable_cls_token
        self.stage = stage
        self.freeze_encoder = params.freeze_encoder
        self.grid_type = params.grid_type

        print("Doing Wrapper for", self.stage)
        if params.grid_type == 'uniform':
            self.agumenter_masker = MaskerUniform(drop_type=params.drop_type, max_block=params.max_block,\
                                                drop_pix=params.drop_pix,channel_per=params.channel_per,\
                                                channel_drop_per=params.channel_drop_per)
            
            # If following augmenter is used by external method during testing 
            self.validation_agumenter = MaskerUniform(drop_type= params.drop_type, max_block=params.max_block_val,\
                                                    drop_pix=params.drop_pix_val,channel_per = params.channel_per_val,\
                                                    channel_drop_per = params.channel_drop_per_val)
        else:
            self.agumenter_masker = MakserNonuniformMest(grid_non_uni=encoder.input_grid.clone().detach(),\
                                                        gird_uni=encoder.output_grid.clone().detach(),\
                                                        radius=params.masking_radius,\
                                                        drop_type=params.drop_type,drop_pix=params.drop_pix,\
                                                        channel_aug_rate=params.channel_per,\
                                                        channel_drop_rate=params.channel_drop_per)
            
            # If following augmenter is used by external method during testing

            self.validation_agumenter = MakserNonuniformMest(grid_non_uni=encoder.input_grid.clone().detach(),\
                                                            gird_uni=encoder.output_grid.clone().detach(),\
                                                            radius=params.masking_radius,\
                                                            drop_type= params.drop_type, max_block=params.max_block_val,\
                                                            drop_pix=params.drop_pix_val,\
                                                            channel_aug_rate=params.channel_per_val,\
                                                            channel_drop_rate=params.channel_drop_per_val)
            


            
        
        self.params = params
        
        

    def forward(self, x, static_random_tensor=None):
        inp = x.clone()
        if self.stage == 'ssl':
            with torch.no_grad():
                inp_masked, mask = batched_masker(inp, self.agumenter_masker)
            augmented_inp_features = self.encoder(inp_masked) 
            if self.enable_cls_token:
                cls_offset = 1
            else:
                cls_offset = 0
            if self.reconstruction:    
                reconstruced = self.decoder(augmented_inp_features)
                #Removing the CLS token and also discarding if some additional channels if
                # in the end
                if self.grid_type == 'uniform':
                    reconstruced =  reconstruced[:,cls_offset:cls_offset+x.shape[1],:,:]
                else:
                    reconstruced =  reconstruced[:,:, cls_offset:]
            else:
                reconstruced = None
            #place holders for contrastive losses.Not used now.
            clean_contra = None
            neg_contra = None
            aug_contra = None
            return reconstruced, clean_contra, aug_contra, neg_contra
        else:
            if self.enable_cls_token:
                cls_offset = 1
            else:
                cls_offset = 0
            if self.freeze_encoder:
                with torch.no_grad():
                    feature = self.encoder(inp)
            else:
                feature = self.encoder(inp)
            out = self.predictor(feature)
            # discarding CLS token and addtion static channels if added.
            if self.grid_type == 'uniform':
                out =  out[:,cls_offset:cls_offset+x.shape[1],:,:]
            else:
                out =  out[:,:, cls_offset:]
            
            return out, None, None, None

class SslWrapChangingMesh(nn.Module):
    '''
    Wrapper of complex non_uniform changing mesh.
    '''
    def __init__(self, params, encoder, decoder, contrastive, predictor, stage):
        super(SslWrapChangingMesh, self).__init__()
    
        self.encoder = encoder
        self.decoder = decoder
        self.contrastive = contrastive
        self.predictor = predictor

        self.reconstruction = params.reconstruction
        self.enable_cls_token = params.enable_cls_token
        self.stage = stage
        self.freeze_encoder = params.freeze_encoder
        self.grid_type = params.grid_type

        print("Doing Wrapper for", self.stage)
        '''
        Assuming grid will be non_uniform
        '''
        self.agumenter_masker = MakserNonuniformMest(grid_non_uni=encoder.input_grid.clone().detach(),\
                                                    gird_uni=encoder.output_grid.clone().detach(),\
                                                    radius=params.masking_radius,\
                                                    drop_type=params.drop_type,drop_pix=params.drop_pix,\
                                                    channel_aug_rate=params.channel_per,\
                                                    channel_drop_rate=params.channel_drop_per)
        self.validation_agumenter = MakserNonuniformMest(grid_non_uni=encoder.input_grid.clone().detach(),\
                                                        gird_uni=encoder.output_grid.clone().detach(),\
                                                        radius=params.masking_radius,\
                                                        drop_type= params.drop_type, max_block=params.max_block_val,\
                                                        drop_pix=params.drop_pix_val,\
                                                        channel_aug_rate=params.channel_per_val,\
                                                        channel_drop_rate=params.channel_drop_per_val)
        self.params = params

    def set_initial_mesh(self, mesh):
        self.register_buffer('initial_mesh', mesh)

    def forward(self, x, out_grid_displacement=None):
        inp = x.clone()
        if self.stage == 'ssl':
            with torch.no_grad():
                # last 3 channel is displacement, taking (x,y), z is 0
                displacement = x[0,:,-3:-1].clone().detach()
                self.encoder.lifting.update_grid(self.initial_mesh + displacement, None)
                self.decoder.projection.update_grid(None, self.initial_mesh + displacement)

            with torch.no_grad():
                inp_masked, mask = batched_masker(inp, self.agumenter_masker)
            augmented_inp_features = self.encoder(inp_masked) 
            if self.enable_cls_token:
                cls_offset = 1
            else:
                cls_offset = 0
            if self.reconstruction:    
                reconstruced = self.decoder(augmented_inp_features)
                #Removing the CLS token and also discarding if some additional channels if
                # in the end
                if self.grid_type == 'uniform':
                    reconstruced =  reconstruced[:,cls_offset:cls_offset+x.shape[1],:,:]
                else:
                    reconstruced =  reconstruced[:,:, cls_offset:]
            else:
                reconstruced = None
            #place holders for contrastive losses.Not used now.
            clean_contra = None
            neg_contra = None
            aug_contra = None
            return reconstruced, clean_contra, aug_contra, neg_contra
        else:

            with torch.no_grad():
                # last 3 channel is displacement, taking (x,y), z is 0
                displacement_inp = x[0,:,-3:-1].clone().detach()
                self.encoder.lifting.update_grid(self.initial_mesh + displacement_inp, None)
                self.predictor.projection.update_grid(None, self.initial_mesh + out_grid_displacement)
                
            if self.enable_cls_token:
                cls_offset = 1
            else:
                cls_offset = 0
            if self.freeze_encoder:
                with torch.no_grad():
                    feature = self.encoder(inp)
            else:
                feature = self.encoder(inp)
            out = self.predictor(feature)
            # discarding CLS token and addtion static channels if added.
            if self.grid_type == 'uniform':
                out =  out[:,cls_offset:cls_offset+x.shape[1],:,:]
            else:
                out =  out[:,:, cls_offset:]
            
            return out, None, None, None