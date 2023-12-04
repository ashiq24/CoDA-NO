from layers.gno_layer import GnoPremEq
import pickle
import numpy as np
import torch


from neuralop.layers.neighbor_search import NeighborSearch
from neuralop.layers.integral_transform import IntegralTransform

mesh = np.loadtxt('../Data/test_data/mesh.csv', delimiter=',')

print(mesh.shape)
idx_x = torch.arange(0, 2.5, 0.01)
idx_y = torch.arange(0, .41, 0.01)
x, y = torch.meshgrid(idx_x, idx_y, indexing='ij')

simple_mesh = torch.stack([x.flatten(), y.flatten()]).type(torch.float)

com_mesh = torch.stack(
    [torch.tensor(mesh[0, :]), torch.tensor(mesh[1, :])]).type(torch.float)


GN = GnoPremEq(2, 1, 3, torch.transpose(com_mesh, 0, 1).cuda(), torch.transpose(simple_mesh, 0, 1).cuda(),
               mlp_layers=[10], projection_hidden_dim=20, radius=0.08, var_encoding=True, var_encoding_channels=1).cuda()

out = GN(torch.randn(1, 1317, 2).cuda())

print(out.shape)
