from layers.gnn_layer import GnnLayer
import torch.nn as nn
import numpy as np
import torch
from layers.gnn_layer import GnnLayer
from neuralop.layers.embeddings import PositionalEmbedding


class DeepONet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        input_grid,
        output_grid=None,
        branch_layers=[128],
        trunk_layers=[128],
        initial_mesh=None,
        positional_encoding_dim=8,
        n_neigbor=10,
        gno_mlp_layers=None,
    ):
        """
        DeepONet is a neural network model used for deep operator learning.

        Args:
            in_dim (int): The number of input dimensions/channels.
            out_dim (int): The number of output dimensions/channels.
            input_grid (torch.Tensor): The input grid.
            output_grid (torch.Tensor, optional): The output grid. If not provided, it is set to the same as the input grid.
            branch_layers (list[int], optional): The number of units in each branch layer. Defaults to [128].
            trunk_layers (list[int], optional): The number of units in each trunk layer. Defaults to [128].
            initial_mesh (torch.Tensor, optional): The initial mesh. Defaults to None.
            positional_encoding_dim (int, optional): The dimension of positional encoding. Defaults to 8.
            n_neigbor (int, optional): The number of neighbors. Defaults to 10.
            gno_mlp_layers (int, optional): The number of MLP layers in GNO. Defaults to None.

        Attributes:
            n_dim (int): The number of dimensions in the input grid.
            n_neigbor (int): The number of neighbors.
            gno_mlp_layers (int): The number of MLP layers in GNO.
            in_dim (int): The number of input dimensions/channels.
            out_dim (int): The number of output dimensions/channels.
            positional_encoding_dim (int): The dimension of positional encoding.
            input_grid (torch.Tensor): The input grid.
            output_grid (torch.Tensor): The output grid.
            initial_mesh (torch.Tensor): The initial mesh.
            branch_layers (list[int]): The number of units in each branch layer.
            trunk_layers (list[int]): The number of units in each trunk layer.
            gnn (GnnLayer): The GNN layer.
            branch (nn.Sequential): The branch layers.
            trunk (nn.Sequential): The trunk layers.
            PE (PositionalEmbedding): The positional embedding.
            bias (nn.Parameter): The bias parameter.

        Methods:
            get_branch(): Returns the branch layers.
            get_trunk(): Returns the trunk layers.
            get_pe(grid): Returns the positional encoding for the given grid.
            forward(inp, out_grid_displacement, in_grid_displacement): Performs forward pass of the model.

        """
        super().__init__()
        if output_grid is None:
            output_grid = input_grid.clone()
        self.n_dim = input_grid.shape[-1]
        self.n_neigbor = n_neigbor
        self.gno_mlp_layers = gno_mlp_layers
        self.in_dim = in_dim
        print("in_dim", in_dim)
        if out_dim is None:
            out_dim = in_dim
        self.out_dim = out_dim
        self.positional_encoding_dim = positional_encoding_dim
        self.input_grid = input_grid
        self.output_grid = output_grid
        self.register_buffer("initial_mesh", initial_mesh)
        self.branch_layers = branch_layers
        self.trunk_layers = trunk_layers
        self.gnn = None
        self.branch = self.get_branch()
        self.trunk = self.get_trunk()
        self.PE = PositionalEmbedding(positional_encoding_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    # Code for varibale encoding
    def get_branch(self):
        """
        Returns a sequential neural network model for the branch layer.

        Returns:
            nn.Sequential: The sequential model for the branch layer.
        """
        dim1 = self.in_dim + self.n_dim * self.positional_encoding_dim
        self.gnn = GnnLayer(
            dim1,
            self.branch_layers[0],
            self.initial_mesh,
            self.initial_mesh,
            self.gno_mlp_layers,
            self.branch_layers[0],
            self.n_neigbor,
        )
        self.layer_norm = nn.LayerNorm(self.branch_layers[0])
        layers = []

        self.branch_layers[0] = self.branch_layers[0] * self.input_grid.shape[0]
        for i in range(len(self.branch_layers) - 1):
            layers.append(nn.Linear(self.branch_layers[i], self.branch_layers[i + 1]))
            torch.nn.init.xavier_normal_(layers[-1].weight)
            if i != len(self.branch_layers) - 2:
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def get_trunk(self):
        """
        Returns the trunk layers of the DeepONet model.

        Returns:
            nn.Sequential: The trunk layers of the DeepONet model.
        """
        dim1 = self.n_dim + self.positional_encoding_dim * self.n_dim
        self.trunk_layers = [dim1] + self.trunk_layers
        self.trunk_layers[-1] = self.trunk_layers[-1] * self.out_dim
        layers = []
        for i in range(len(self.trunk_layers) - 1):
            layers.append(nn.Linear(self.trunk_layers[i], self.trunk_layers[i + 1]))
            torch.nn.init.xavier_normal_(layers[-1].weight)
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def get_pe(self, grid):
        """
        Calculates the potential energy (PE) for a given grid.

        Parameters:
        - grid: numpy.ndarray
            The input grid.

        Returns:
        - pe: numpy.ndarray
            The potential energy calculated for the grid.
        """
        pe = self.PE(grid.reshape(-1))
        pe = pe.reshape(grid.shape[0], -1)
        return pe

    def forward(self, inp, out_grid_displacement=None, in_grid_displacement=None):
        """
        Forward pass of the DeepONet model.

        Args:
            inp (torch.Tensor): Input tensor of shape (batch_size, n_points, in_dims/Channels).
                                Currently, only batch_size = 1 is supported.
            out_grid_displacement (torch.Tensor, optional): Displacement tensor for the output grid.
                                                            Default is None.
            in_grid_displacement (torch.Tensor, optional): Displacement tensor for the input grid.
                                                           Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_points, out_dim).

        Note:
            - If `out_grid_displacement` is not None, the input and output grids are updated using the
              displacement tensors before the forward pass.
            - The input data is concatenated with positional information.
            - The output tensor is computed using the GNN (Graph Neural Network) and the trunk network.
            - The output tensor is then multiplied with the bias and returned.

        """
        if out_grid_displacement is not None:
            with torch.no_grad():
                in_grid = self.initial_mesh + in_grid_displacement
                out_grid = self.initial_mesh + out_grid_displacement
                self.gnn.update_grid(in_grid.clone(), in_grid.clone())

        # data is concatenated with grid/positional information
        in_pe = self.get_pe(in_grid)
        in_data = torch.cat([inp, in_pe.unsqueeze(0)], dim=-1)

        bout = self.gnn(in_data[0])[None, ...]  # (batch, dim)

        bout = self.layer_norm(bout)

        bout = self.branch(bout.reshape(inp.shape[0], -1))

        bout = bout / np.sqrt(self.branch_layers[-1])

        pe = self.get_pe(out_grid)
        grid_pe = torch.cat([out_grid, pe], axis=1)

        tout = self.trunk(grid_pe)  # (ngrid, dim * out_dim)
        tout = tout.reshape(
            out_grid.shape[0], self.out_dim, -1
        )  # (ngrid, out_dim, dim)

        out = torch.einsum("bd,ncd->bnc", bout, tout)

        return out + self.bias
