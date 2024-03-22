from functools import reduce
from typing import Tuple
from neuralop.layers.mlp import MLPLinear
import numpy as np
import torch
from torch import nn
import torch_harmonics as th
from neuralop.layers.embeddings import PositionalEmbedding


class VariableEncoding2d(nn.Module):
    def __init__(self, channel, modes: Tuple[int, ...], basis="fft") -> None:
        """
        Variable Encoding 2D module.

        Args:
            channel (int): Number of input channels.
            modes (Tuple[int, ...]): Tuple of integers representing the modes for encoding.
            basis (str, optional): Basis for encoding. Can be "fft" or "sht". Defaults to "fft".
        """
        super().__init__()
        self.modes = modes
        self.coefficients_r = nn.Parameter(torch.empty(channel, *modes))
        self.coefficients_i = nn.Parameter(torch.empty(channel, *modes))
        self.reset_parameters()
        self.basis = basis
        if basis == "fft":
            self.transform = torch.fft.ifft2
        elif basis == "sht":
            self.transform = th.InverseRealSHT(
                *modes,
                lmax=modes[-2],
                mmax=modes[-1],
                grid="legendre-gauss",
                norm="backward",
            )

    def reset_parameters(self):
        """
        Reset the parameters of the module.
        """
        std = (1 / (self.modes[-1] * self.modes[-2])) ** 0.5
        torch.nn.init.normal_(self.coefficients_r, mean=0.0, std=std)
        torch.nn.init.normal_(self.coefficients_i, mean=0.0, std=std)

    def forward(self, x):
        """
        Forward pass of the module.

        Args:
            x: Input tensor.

        Returns:
            Tensor: Encoded tensor.
        """
        size_x, size_y = x.shape[-2], x.shape[-1]
        if self.basis == "sht":
            if self.transform.nlat == size_x and self.transform.nlon == size_y:
                return self.transform(self.coefficients_r + 1.0j * self.coefficients_i)

            self.transform = th.InverseRealSHT(
                size_x,
                size_y,
                lmax=self.modes[-2],
                mmax=self.modes[-1],
                grid="legendre-gauss",
                norm="backward",
            ).to(device=self.coefficients_i.device, dtype=self.coefficients_i.dtype)
            return self.transform

        else:
            return self.transform(
                self.coefficients_r + 1.0j * self.coefficients_i, s=(size_x, size_y)
            ).real


class FourierVariableEncoding3D(nn.Module):
    def __init__(self, n_features: int, modes: Tuple[int, ...]) -> None:
        """
        Initialize the FourierVariableEncoding3D module.

        Args:
            n_features (int): Number of input features.
            modes (Tuple[int, ...]): Tuple of three integers representing the frequency modes.

        Raises:
            ValueError: If the length of `modes` is not equal to 3.
        """
        super().__init__()
        if len(modes) != 3:
            raise ValueError(
                f"Expected 3 frequency modes, but got {len(modes)} modes:\n{modes=}"
            )

        self.modes = modes
        self.weights_re = nn.Parameter(torch.empty(n_features, *modes))
        self.weights_im = nn.Parameter(torch.empty(n_features, *modes))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the parameters of the module.
        """
        std = 1 / np.sqrt(reduce(lambda a, b: a * b, self.modes))
        torch.nn.init.normal_(self.weights_re, mean=0.0, std=std)
        torch.nn.init.normal_(self.weights_im, mean=0.0, std=std)

    def forward(self, x_shape):
        """
        Forward pass of the module.

        Args:
            x_shape: The shape of the input tensor.

        Returns:
            torch.Tensor: The positional encodings.
        """
        *_, size_t, size_x, size_y = x_shape
        return torch.fft.ifftn(
            self.weights_re + 1.0j * self.weights_im,
            s=(size_t, size_x, size_y),
            norm="forward",  # don't multiply by any normalization factor
        ).real


class VariableEncodingIrregularMesh(nn.Module):
    """
    A module for variable encoding on an irregular mesh.

    Args:
        n_variables (int): The number of variables.
        variable_encoding_size (int): The size of the variable encoding.
        n_dim (int, optional): The number of dimensions. Defaults to 2.
        positional_encoding_dim (int, optional): The dimension of the positional encoding. Defaults to 8.
    """

    def __init__(
        self,
        n_variables: int,
        variable_encoding_size: int,
        n_dim: int = 2,
        positional_encoding_dim: int = 8,
    ) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.variable_encoding_size = variable_encoding_size
        self.n_dim = n_dim
        self.positional_encoding_dim = positional_encoding_dim
        self.var_encoder = MLPLinear(
            [
                n_dim + self.n_dim * positional_encoding_dim,
                self.variable_encoding_size * n_variables,
            ]
        )
        self.PE = PositionalEmbedding(positional_encoding_dim)

    def forward(self, grid_poits):
        pe = self.PE(grid_poits.reshape(-1))
        pe = pe.reshape(grid_poits.shape[0], -1)
        grid_pe = torch.cat([grid_poits, pe], axis=1)
        var_encoding = self.var_encoder(grid_pe)
        return var_encoding


class VariableEncodingWrapper(nn.Module):
    def __init__(
        self,
        equation_dict: dict,
        variable_encoding_size: int,
        n_dim: int = 2,
        positional_encoding_dim: int = 8,
    ) -> None:
        """
        Wrapper class for variable encoding.

        Args:
            equation_dict (dict): A dictionary containing equations and the number of variables for each equation.
            variable_encoding_size (int): The size of the variable encoding.
            n_dim (int, optional): The number of dimensions for the variable encoding. Defaults to 2.
            positional_encoding_dim (int, optional): The dimension of the positional encoding. Defaults to 8.
        """
        super().__init__()
        self.n_dim = n_dim
        self.equation_dict = equation_dict
        self.variable_encoding_size = variable_encoding_size
        self.model_dict = nn.ModuleDict()
        for i in equation_dict.keys():
            self.model_dict[i] = VariableEncodingIrregularMesh(
                n_variables=equation_dict[i],
                variable_encoding_size=self.variable_encoding_size,
                n_dim=n_dim,
                positional_encoding_dim=positional_encoding_dim,
            )

    def load_encoder(self, equation: str, path: str):
        """
        Load the encoder for a specific equation from a given path.

        Args:
            equation (str): The equation for which to load the encoder.
            path (str): The path to the saved encoder model.
        """
        self.model_dict[equation].load_state_dict(
            torch.load(path, map_location=torch.device("cpu"))
        )

    def save_encoder(self, equation: str, path: str):
        """
        Save the encoder for a specific equation to a given path.

        Args:
            equation (str): The equation for which to save the encoder.
            path (str): The path to save the encoder model.
        """
        torch.save(self.model_dict[equation].state_dict(), path)

    def save_all_encoder(self, path: str):
        """
        Save all encoders to a given path.

        Args:
            path (str): The path to save the encoder models.
        """
        for i in self.equation_dict.keys():
            torch.save(self.model_dict[i].state_dict(), path + f"_{i}" + ".pt")

    def freeze(self, equation: str):
        """
        Freeze the parameters of the encoder for a specific equation.

        Args:
            equation (str): The equation for which to freeze the encoder parameters.
        """
        for param in self.model_dict[equation].parameters():
            param.requires_grad = False

    def forward(self, grid_poits, equation: str = None):
        """
        Forward pass of the variable encoding wrapper.

        Args:
            grid_poits: The input grid points.
            equation (str, optional): The equation for which to compute the variable encoding. If None, compute for all equations.

        Returns:
            torch.Tensor: The concatenated variable encodings.
        """
        encoding_list = []
        if equation is None:
            equation = list(self.equation_dict.keys())
        for i in equation:
            encoding_list.append(self.model_dict[i](grid_poits))
        return torch.cat(encoding_list, axis=1).unsqueeze(0)


def get_variable_encoder(params):
    """
    Returns a VariableEncodingWrapper object with the specified parameters.

    Args:
        params (dict): A dictionary containing the following parameters:
            - equation_dict (dict): A dictionary mapping variable names to equations.
            - n_encoding_channels (int): The number of encoding channels.
            - n_dim (int): The dimensionality of the encoding.
            - positional_encoding_dim (int): The dimensionality of the positional encoding.

    Returns:
        VariableEncodingWrapper: A VariableEncodingWrapper object.

    """
    return VariableEncodingWrapper(
        params.equation_dict,
        variable_encoding_size=params.n_encoding_channels,
        n_dim=params.n_dim,
        positional_encoding_dim=params.positional_encoding_dim,
    )
