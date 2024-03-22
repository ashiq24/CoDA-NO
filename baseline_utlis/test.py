from rigid_neighbor import *
import torch


def find_neighbours(in_: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """
    Finds the nearest neighbors for each point in the input tensor.

    Args:
        in_: Input tensor of shape (N, 3) representing the points.
        out: Output tensor of shape (N, 3) representing the points.

    Returns:
        neighbour: Tensor of shape (N, 3) representing the nearest neighbors for each point.
    """
    NS = FixedNeighborSearch(use_open3d=False)
    neighbour = NS(in_, out, 3)
    return neighbour


in_ = torch.randn((10, 3))
out = torch.randn((10, 3))

neighbour = find_neighbours(in_, out)
print(neighbour)
