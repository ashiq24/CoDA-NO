import pickle
import numpy as np
import torch

# Load the NumPy array from a pickle file
with open('../Data/MP_data/displacements0-5000.pkl', 'rb') as file:
    dis_array = pickle.load(file)

from neuralop.layers.neighbor_search import NeighborSearch
from neuralop.layers.integral_transform import IntegralTransform

mesh = np.loadtxt('../Data/dataset/mesh.csv', delimiter=',')

idx_x = torch.arange(0,2.5, 0.01)
idx_y = torch.arange(0,.41, 0.01)
x, y  = torch.meshgrid(idx_x, idx_y, indexing='ij') 

simple_mesh = torch.stack([x.flatten(), y.flatten()]).type(torch.float)

com_mesh = torch.stack( [torch.tensor(mesh[0,:]),torch.tensor(mesh[1,:])]).type(torch.float)

from layers.gno_layer import gno_layer

GN = gno_layer(2,1, torch.transpose(com_mesh, 0, 1).cuda(), torch.transpose(simple_mesh,0, 1).cuda(),\
               [4,1], radius=0.08, var_encoding=False, var_encoding_channels=1).cuda()

out = GN(torch.randn(1,1317,2).cuda())