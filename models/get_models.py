from layers.attention import TnoBlock2d
from layers.fino import SpectralConvKernel2d
from data_utils.data_utils import mask_patches, batched_masker

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
        self.enable_cls_token = params.enable_cls_token
        self.predictor = predictor
        self.stage = stage
        self.freeze_encoder = params.freeze_encoder
        summary(self.encoder, (self.encoder.var_num,params.size_x,params.size_y))
                
        print("Doing Wrapper for", self.stage)
        self.agumenter_masker = partial(mask_patches, drop_type=params.drop_type, max_block=params.max_block, drop_pix=params.drop_pix,\
                                                  channel_per=params.channel_per, channel_drop_per=params.channel_drop_per)
        
        self.validation_agumenter = partial(mask_patches, drop_type= params.drop_type, max_block=params.max_block_val, drop_pix=params.drop_pix_val,\
                                                  channel_per = params.channel_per_val, channel_drop_per = params.channel_drop_per_val)
        self.params = params
        self.reconstruction_loss = params.reconstruction_loss
        

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
                inp_shuffled = None
                if self.params.apply_contrastive_loss and self.params.apply_shuffle:
                    inp_shuffled, _ = self.augmenter_channel_shuffler(inp, shuffle_all = True, static_random_int=static_random_tensor)

            augmented_inp_features = self.encoder(inp_masked)
            
            
            #print("Feature Shape", augmented_inp_features.shape)
            
            if self.enable_cls_token:
                cls_offset = 1
            else:
                cls_offset = 0
            if self.reconstruction_loss:    
                reconstraced = self.decoder(augmented_inp_features)
                #Removing the CLS token and also discarding if some additional channels if
                # in the end
                reconstraced =  reconstraced[:,cls_offset:cls_offset+x.shape[1],:,:]
            else:
                reconstraced = None
            



            # reconstraced, aug_contra  = self.model(inp_masked, 'ssl')
            # print("Model Forward done")

            clean_contra = None
            neg_contra = None
            aug_contra = None
            # at this momemnt this constrastive loss is not adding much
            # benifit and not used. We can skip it for now.
            if self.params.apply_contrastive_loss:
                if self.enable_cls_token:
                    aug_contra = augmented_inp_features[:,:self.encoder.hidden_token_codim,:,:]
                    #print("Output CLS token Shape", aug_contra.shape)
                    clean_inp_features = self.encoder(inp)
                    clean_contra = clean_inp_features[:,:self.encoder.hidden_token_codim,:,:]
                else:
                    aug_contra = self.contrastive(augmented_inp_features)
                    clean_inp_features = self.encoder(inp)
                    clean_contra = self.contrastive(clean_inp_features)
                
                if self.params.apply_shuffle:
                    neg_input_feature = self.encoder(inp_shuffled)
                    if self.enable_cls_token:
                        neg_contra = neg_input_feature[:,:self.encoder.hidden_token_codim,:,:]
                    else:
                        neg_contra = self.contrastive(neg_input_feature)

            return reconstraced, clean_contra, aug_contra, neg_contra
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