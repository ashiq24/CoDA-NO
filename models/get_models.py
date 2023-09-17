from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from data_utils.data_utils import mask_patches, batched_masker
import torch.nn as nn
from functools import partial


def get_ssl_models(params):
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
                    static_features=static_features
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
    )
    print("*********************")
    
    return encoder, decoder, contrastive, predictor

class SslWrapper(nn.Module):
    '''
    unlike other wrapper, this take initialized model
    '''
    def __init__(self, params, encoder, decoder, contrastive, predictor, stage):
        super(SslWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)

        self.encoder = encoder
        self.decoder = decoder
        self.contrastive = contrastive
        self.predictor = predictor

        self.reconstruction = params.reconstruction

        self.enable_cls_token = params.enable_cls_token
        
        self.stage = stage
        self.freeze_encoder = params.freeze_encoder
        summary(self.encoder, (self.encoder.var_num,params.size_x,params.size_y))
                
        print("Doing Wrapper for", self.stage)
        self.agumenter_masker = partial(mask_patches, drop_type=params.drop_type, max_block=params.max_block, drop_pix=params.drop_pix,\
                                                  channel_per=params.channel_per, channel_drop_per=params.channel_drop_per)
        
        self.validation_agumenter = partial(mask_patches, drop_type= params.drop_type, max_block=params.max_block_val, drop_pix=params.drop_pix_val,\
                                                  channel_per = params.channel_per_val, channel_drop_per = params.channel_drop_per_val)
        self.params = params
        
        

    def forward(self, x, static_random_tensor=None):
         # first append unpredicted features
        inp = self.preprocessor.append_unpredicted_features(x)

        # now normalize
        self.preprocessor.history_compute_stats(inp)
        inp = self.preprocessor.history_normalize(inp, target=False)

        # now add static features if requested
        #inp = self.preprocessor.add_static_features(inp)
        
        #print("printing x and added feature shape", x.shape, inp.shape)
        
        if self.stage == 'ssl':
            with torch.no_grad():
                inp_masked, mask = batched_masker(inp, self.agumenter_masker)

            augmented_inp_features = self.encoder(inp_masked)
            
            
            #print("Feature Shape", augmented_inp_features.shape)
            
            if self.enable_cls_token:
                cls_offset = 1
            else:
                cls_offset = 0
            if self.reconstruction:    
                reconstruced = self.decoder(augmented_inp_features)
                #Removing the CLS token and also discarding if some additional channels if
                # in the end
                reconstruced =  reconstruced[:,cls_offset:cls_offset+x.shape[1],:,:]
            else:
                reconstruced = None

            # reconstruced, aug_contra  = self.model(inp_masked, 'ssl')
            # print("Model Forward done")

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
            out =  out[:,cls_offset:cls_offset+x.shape[1],:,:]
            return out, None, None, None