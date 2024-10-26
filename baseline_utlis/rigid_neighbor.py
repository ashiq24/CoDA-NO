import torch
from torch import nn


def simple_neighbor_search(
        data: torch.Tensor,
        queries: torch.Tensor,
        n_neigbor: float):
    """

    Parameters
    ----------
    Density-Based Spatial Clustering of Applications with Noise
    data : torch.Tensor
        vector of data points from which to find neighbors
    queries : torch.Tensor
        centers of neighborhoods

    """

    # shaped num query points x num data points
    dists = torch.cdist(queries, data).to(queries.device)
    sorted_dist, _ = torch.sort(dists, dim=1)
    k = sorted_dist[:, n_neigbor]
    dists = dists - k[:, None]
    in_nbr = torch.where(dists < 0, 1., 0.)  # i,j is one if j is i's neighbor
    # only keep the column indices
    nbr_indices = in_nbr.nonzero()[:, 1:].reshape(-1,)
    # num points in each neighborhood, summed cumulatively
    nbrhd_sizes = torch.cumsum(torch.sum(in_nbr, dim=1), dim=0)
    splits = torch.cat((torch.tensor([0.]).to(queries.device), nbrhd_sizes))
    nbr_dict = {}
    nbr_dict['neighbors_index'] = nbr_indices.long().to(queries.device)
    nbr_dict['neighbors_row_splits'] = splits.long()
    return nbr_dict


class FixedNeighborSearch(nn.Module):
    """Neighbor search within a ball of a given radius

    Parameters
    ----------
    use_open3d : bool
        Whether to use open3d or torch_cluster
        NOTE: open3d implementation requires 3d data
    """

    def __init__(self, use_open3d=True, use_torch_cluster=False):
        super().__init__()
        self.search_fn = simple_neighbor_search
        self.use_open3d = False

    def forward(self, data, queries, n_neigbor):
        """Find the neighbors, in data, of each point in queries
        within a ball of radius. Returns in CRS format.

        Parameters
        ----------
        data : torch.Tensor of shape [n, d]
            Search space of possible neighbors
            NOTE: open3d requires d=3
        queries : torch.Tensor of shape [m, d]
            Point for which to find neighbors
            NOTE: open3d requires d=3
        radius : float
            Radius of each ball: B(queries[j], radius)

        Output
        ----------
        return_dict : dict
            Dictionary with keys: neighbors_index, neighbors_row_splits
                neighbors_index: torch.Tensor with dtype=torch.int64
                    Index of each neighbor in data for every point
                    in queries. Neighbors are ordered in the same orderings
                    as the points in queries. Open3d and torch_cluster
                    implementations can differ by a permutation of the
                    neighbors for every point.
                neighbors_row_splits: torch.Tensor of shape [m+1] with dtype=torch.int64
                    The value at index j is the sum of the number of
                    neighbors up to query point j-1. First element is 0
                    and last element is the total number of neighbors.
        """
        return_dict = {}

        return_dict = self.search_fn(data, queries, n_neigbor)

        return return_dict
