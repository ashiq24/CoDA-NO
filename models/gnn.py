from layers.gnn_layer import GnnLayer
import torch.nn as nn
import torch.nn.functional as F
from neuralop.layers.mlp import MLPLinear
import torch


class GNN(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        input_grid,
        output_grid=None,
        n_neigbor=None,
        hidden_dim=None,
        lifting_dim=None,
        n_layers=4,
        initial_mesh=None,
        non_linearity=F.gelu,
        projection=True,
        gno_mlp_layers=None,
        lifting=True,
    ):
        super().__init__()
        self.n_layers = n_layers

        """
        Initialize the GNN model.

        Args:
            in_dim (int): The input dimension.
            out_dim (int): The output dimension.
            input_grid (torch.Tensor): The input grid.
            output_grid (torch.Tensor, optional): The output grid. Defaults to None.
            n_neigbor (int, optional): The number of neighbors. Defaults to None.
            hidden_dim (int, optional): The hidden dimension. Defaults to None.
            lifting_dim (int, optional): The lifting dimension. Defaults to None.
            n_layers (int, optional): The number of GNN layers. Defaults to 4.
            initial_mesh (torch.Tensor, optional): The initial mesh. Defaults to None.
            non_linearity (function, optional): The non-linearity function. Defaults to F.gelu.
            projection (bool, optional): Whether to use the projection layer. Defaults to True.
            gno_mlp_layers (list, optional): The layers of the GNO MLP. Defaults to None.
            lifting (bool, optional): Whether to use the lifting layer. Defaults to True.
        """
        super().__init__()
        self.n_layers = n_layers

        if output_grid is None:
            output_grid = input_grid.clone()

        self.n_dim = input_grid.shape[-1]

        self.in_dim = in_dim

        if hidden_dim is None:
            hidden_dim = in_dim
        if lifting_dim is None:
            lifting_dim = in_dim
        if out_dim is None:
            out_dim = in_dim

        self.input_grid = input_grid
        self.output_grid = output_grid

        self.hidden_dim = hidden_dim

        self.lifting = lifting
        self.projection = projection
        self.n_neigbor = n_neigbor
        self.gno_mlp_layers = gno_mlp_layers

        self.register_buffer("initial_mesh", initial_mesh)
        # Code for variable encoding

        # Initializing Components
        if self.lifting:
            print("Using lifting Layer")
        for i in range(self.n_layers):
            self.base.append(
                GnnLayer(
                    hidden_dim,
                    hidden_dim,
                    self.initial_mesh,
                    self.initial_mesh,
                    gno_mlp_layers,
                    lifting_dim,
                    n_neigbor,
                )
            )

        if self.projection:
            print("Using Projection Layer")
            self.projection = MLPLinear(layers=[self.hidden_dim, out_dim])

    def forward(self, inp, out_grid_displacement=None, in_grid_displacement=None):
        """
        Forward pass of the GNN model.

        Args:
            inp (torch.Tensor): Input tensor of shape (batch_size, n_points, in_dims/Channels).
                                Currently, only batch_size = 1 is supported.
            out_grid_displacement (torch.Tensor, optional): Tensor representing the displacement
                                                            of the output grid. Default is None.
            in_grid_displacement (torch.Tensor, optional): Tensor representing the displacement
                                                           of the input grid. Default is None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, n_points, out_dims/Channels).

        Note:
            - If `out_grid_displacement` is not None, the grid is updated using the displacement tensors.
            - If `lifting` is True, the input tensor is passed through the lifting layer before processing.
            - The input tensor is then passed through each layer of the GNN model.
            - If `projection` is True, the output tensor is passed through the projection layer before returning.
        """
        if out_grid_displacement is not None:
            with torch.no_grad():
                for i in range(self.n_layers):
                    if i == self.n_layers - 1:
                        in_grid = self.initial_mesh + in_grid_displacement
                        out_grid = self.initial_mesh + out_grid_displacement
                    else:
                        in_grid = self.initial_mesh + in_grid_displacement
                        out_grid = self.initial_mesh + in_grid_displacement
                    self.base[i].update_grid(in_grid, out_grid)

        if self.lifting:
            x = self.lifting(inp)
        else:
            x = inp
        x = x[0, ...]
        for layer_idx in range(self.n_layers):
            x = self.base[layer_idx](x)

        if self.projection:
            x = self.projection(x)
        return x[None, ...]
