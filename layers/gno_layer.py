from neuralop.layers.neighbor_search import NeighborSearch
from neuralop.layers.integral_transform import IntegralTransform
from neuralop.layers.mlp import MLPLinear
from einops import rearrange
import torch.nn as nn
import torch

class gno_layer(nn.Module):
    def __init__(self, var_num, in_dim, out_dim, \
                input_grid, output_grid, mlp_layers, projection_hidden_dim, \
                radius, var_encoding=False, var_encoding_channels=1,):
        super().__init__()

        n_dim = input_grid.shape[-1]
        self.var_num = var_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.mlp_layers = [2*n_dim] +mlp_layers + [out_dim]
        self.var_encoding = var_encoding
        self.var_encoding_channels = var_encoding_channels
        
        ### get varibale encoding
        if self.var_encoding:
            self.var_encoder = MLPLinear([n_dim, self.var_encoding_channels*var_num])
            self.variable_channels = [i*(var_encoding_channels+self.in_dim) for i in range(var_num)]
            self.encoding_channels = list(set([i for i in range((var_encoding_channels+1)*var_num)]) -set(self.variable_channels))
        else:
            self.var_encoding_channels = 0

        ### project to higher dim
        self.projection = MLPLinear([self.var_encoding_channels+self.in_dim,\
                                        projection_hidden_dim ,out_dim])

        ### apply GNO to get  uniform grid

        NS = NeighborSearch(use_open3d=False)

        self.neighbour = NS(input_grid.clone().cpu(), output_grid.clone().cpu(), radius=radius)

        for key, value in self.neighbour.items():
            self.neighbour[key] = self.neighbour[key].cuda()
        
        self.it = IntegralTransform(mlp_layers=self.mlp_layers)
    
    def forward(self, inp):
        '''
        inp : (batch_size, n_points, in_dims/Channels)
        '''
        print("Input Shape", inp.shape)
        if self.var_encoding:
            x = torch.zeros((inp.shape[0], inp.shape[1],len(self.variable_channels)+len(self.encoding_channels)), device=inp.device, dtype=inp.dtype)
            var_encoding = self.var_encoder(self.input_grid).to(x.device)
            x[:,:,self.variable_channels] = inp
            x[:,:,self.encoding_channels] = var_encoding[None,:,:].repeat(x.shape[0],1,1)
        else:
            x = inp
        print("Input Shape after Var Encdoing", x.shape)
        ## Currently GNO only works for batch_size = 1

        x  = rearrange(x, 'b n (v c) -> (b n) v c', c = self.in_dim+self.var_encoding_channels)
        print("Input Shape after Rearrange", x.shape)
        x = self.projection(x)
        
        out = None

        for i in range(x.shape[-2]):
            print(i)
            print(x[:,i,:].shape)

            temp = self.it(self.input_grid, self.neighbour,self.output_grid, x[:,i,:])
            if out is None:
                out = temp[None,...]
            else:
                out = torch.cat([out, temp[None,...]], dim=2)
        print("Output Shape after Rearrange", out.shape)
        return out