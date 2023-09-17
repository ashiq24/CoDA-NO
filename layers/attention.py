from neuralop.layers.fno_block import FNOBlocks
import torch.nn as nn
from neuralop.layers.spectral_convolution import SpectralConv
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from einops.layers.torch import Rearrange
from .fino import SpectralConvKernel2d
import torch

class TnoBlock2d(nn.Module):
    def __init__(self, n_modes,
                 n_head=1,
                 token_codim=1,
                 output_scaling_factor=None,
                 incremental_n_modes=None,
                 head_codim=None,
                 use_mlp=False, mlp=None, mlp_dropout=0,
                 non_linearity=F.gelu,
                 norm=None, preactivation=False,
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
                 decomposition_kwargs=dict(),
                 fft_norm='forward',
                 codim_size=None,
                 per_var_attention=True,
                 permutation_eq=True,
                 **kwarg):
        
        super().__init__()
        self.variable_codim = token_codim
        self.token_codim = token_codim
        self.head_codim = token_codim if head_codim is None else head_codim
        self.n_head = n_head
        self.output_scaling_factor = output_scaling_factor
        self.per_var_attention = per_var_attention
        self.permutation_eq = permutation_eq
        if self.n_head is not None:
            self.head_codim = max(token_codim//self.n_head, 1)
        
        self.codim_size = codim_size
        self.mixer_token_codim = token_codim
        
        if per_var_attention:
            self.token_codim = 1
            self.head_codim = 1

            
        scale = min(self.n_head,2)
        if self.per_var_attention:
            scale = 4
            
        mixer_modes = [i//scale for i in n_modes]
        
        print(rank, factorization,self.head_codim, scale, mixer_modes)
        if not per_var_attention:
            print("Token dim", self.token_codim, "number heads", self.n_head, 'Head co-dim', self.head_codim)
            #assert self.token_codim == self.n_head * self.head_codim
        
        self.K = FNOBlocks(in_channels=self.token_codim, out_channels= self.n_head * self.head_codim, n_modes= mixer_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor=1/scale,non_linearity=lambda x: x, apply_skip=True,\
                                            norm=None, preactivation=preactivation, fno_skip='linear',mlp_skip=mlp_skip,mlp_dropout=0,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm, mlp_expansion=mlp_expansion,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv=partial(SpectralConv,rank=0.5, factorization=None),n_layers=1)

        self.Q = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.head_codim, n_modes= mixer_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor=1/scale, non_linearity=lambda x: x, apply_skip=True,
                                            norm=None, preactivation=preactivation, fno_skip='linear',mlp_skip=mlp_skip, mlp_dropout=0,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,mlp_expansion=mlp_expansion,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv=partial(SpectralConv,rank=0.5, factorization=None), n_layers=1)

        self.V = FNOBlocks(in_channels= self.token_codim, out_channels= self.n_head * self.head_codim, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor=1,non_linearity=lambda x: x, apply_skip=True,
                                            norm=None, preactivation=preactivation, fno_skip='linear',mlp_skip=mlp_skip, mlp_dropout=0,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,mlp_expansion=mlp_expansion,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv=partial(SpectralConv,rank=0.5, factorization=None),n_layers=1)
        #= partial(SpectralConv,frequency_mixer = False)
        
        if  self.n_head * self.head_codim != self.token_codim:
            self.proj = FNOBlocks(in_channels= self.n_head * self.head_codim, out_channels=self.token_codim, n_modes= n_modes,\
                                            use_mlp=use_mlp, mlp=mlp, output_scaling_factor=1,non_linearity=lambda x: x, apply_skip=True,
                                            norm=None, preactivation=preactivation, fno_skip='linear',mlp_skip=mlp_skip, mlp_dropout=0,\
                                            incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,mlp_expansion=mlp_expansion,\
                                            fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                            factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                            SpectralConv=partial(SpectralConv,rank=0.5, factorization=None),n_layers=1)
        else:
            self.proj = None

        self.attention_normalizer = nn.InstanceNorm2d(self.token_codim, affine=True)
        
        if permutation_eq:
            print("Permutation Equivariant with ", self.mixer_token_codim)
            self.mixer = FNOBlocks(in_channels=self.mixer_token_codim, out_channels=self.mixer_token_codim, n_modes= n_modes,\
                                                use_mlp=use_mlp, mlp=mlp, output_scaling_factor=1,non_linearity=non_linearity,\
                                                norm='instance_norm', preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip,mlp_expansion=mlp_expansion,\
                                                mlp_dropout=0,incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                                fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                                factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                                SpectralConv=partial(SpectralConv,rank=0.5, factorization=None, bias=True),n_layers=1)
            self.norm1 = nn.InstanceNorm2d(self.token_codim, affine=True)
            self.norm2 = nn.InstanceNorm2d(self.mixer_token_codim, affine=True)
            self.mixer_out_normalizer = nn.InstanceNorm2d(self.mixer_token_codim, affine=True)
        else:
            self.mixer = FNOBlocks(in_channels=codim_size, out_channels=codim_size, n_modes= n_modes,\
                                                use_mlp=use_mlp, mlp=mlp, output_scaling_factor=1,non_linearity=non_linearity,\
                                                norm='instance_norm', preactivation=preactivation, fno_skip=fno_skip,mlp_skip=mlp_skip,mlp_expansion=mlp_expansion,\
                                                mlp_dropout=0,incremental_n_modes=incremental_n_modes, rank=rank, fft_norm=fft_norm,\
                                                fixed_rank_modes=fixed_rank_modes, implementation=implementation, separable=separable,\
                                                factorization=factorization,decomposition_kwargs=decomposition_kwargs,joint_factorization=joint_factorization,\
                                                SpectralConv=partial(SpectralConv,rank=0.5, factorization=None, bias=True),n_layers=2)
            self.norm1 = nn.InstanceNorm2d(codim_size, affine=True)
            self.norm2 = nn.InstanceNorm2d(codim_size, affine=True)
            self.mixer_out_normalizer = nn.InstanceNorm2d(codim_size, affine=True)
        

        
    def forward(self, x, output_shape=None):
        batch, n_token , in_res_x, in_res_y=x.shape[0], x.shape[1]//self.token_codim, x.shape[-2], x.shape[-1]
        
        
        assert x.shape[1]%self.token_codim == 0
        
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
        
        k = rearrange(k, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head )
        q = rearrange(q, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head )
        v = rearrange(v, '(b t) (a d) h w -> b a t (d h w)', b=batch, a=self.n_head )


        dprod = torch.matmul(q, k.transpose(-1, -2))/(k.shape[-1]**0.5)

        dprod = F.softmax(dprod, dim=-1)
                
        atten =  torch.matmul(dprod, v)
        
        atten = rearrange(atten, 'b a t (d h w) -> b t a d h w', d=self.head_codim, h=value_res_x, w=value_res_y)
        
        atten = rearrange(atten, 'b t a d h w -> (b t) (a d) h w')

        if self.proj is not None:
            atten = self.proj.convs(atten)
        
        if not self.permutation_eq:
            atten = rearrange(atten, '(b t) d h w -> b (t d) h w', b = batch)
            atten_normalized = self.norm2(atten)
            #atten_normalized = rearrange(atten_normalized, 'b (t d) h w -> (b t) d h w', d=self.variable_codim)
            output = self.mixer(atten_normalized, output_shape=(in_res_x, in_res_y))
            #output = rearrange(output, '(b t) d h w -> b (t d) h w', b = batch)
        else:
            atten = self.attention_normalizer(atten) + xa
            atten = rearrange(atten, '(b t) d h w -> b (t d) h w', b = batch)
            #print("Attention shape", atten.shape)
            atten = rearrange(atten, 'b (t d) h w -> (b t) d h w', d = self.mixer_token_codim)
            #print("Attention shape", atten.shape)
            atten_normalized = self.norm2(atten)
            output = self.mixer(atten_normalized, output_shape=(in_res_x, in_res_y)) 
            output = self.mixer_out_normalizer(output) + atten
            #print("outshape", output.shape)
            output = rearrange(output, '(b t) d h w -> b (t d) h w', b = batch)
        return output
