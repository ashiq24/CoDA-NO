from rigid_neighbor import *
import torch

in_ = torch.randn((10, 3))
out = torch.randn((10, 3))

NS = NeighborSearch(use_open3d=False)

neighbour = NS(in_, in_, 3)
print(neighbour)