from functools import reduce
from typing import Tuple, Literal, Optional

from numpy import sqrt as np_sqrt
import torch
from torch import nn
import torch_harmonics as th

from neuralop.layers.mlp import MLPLinear
from neuralop.layers.embeddings import PositionalEmbedding


class VariableEncoding2D(nn.Module):
    def __init__(
        self,
        channel: int,
        modes: Tuple[int, ...],
        basis: Literal["fft", "sht"] = 'fft',
    ):
        super().__init__()
        self.modes = modes
        self.weights = nn.Parameter(torch.empty(channel, *modes, dtype=torch.cfloat))
        self.reset_parameters()

        self.basis = basis
        if basis == 'fft':
            self.transform = torch.fft.ifft2
        elif basis == 'sht':
            self.transform = th.InverseRealSHT(
                *modes,  # assuming len(modes)==2, unpacks like nlat, nlon = modes
                lmax=modes[-2],
                mmax=modes[-1],
                grid="legendre-gauss",
                norm="backward",
            )
        else:
            raise ValueError(f'Expected one of "fft" or "sht". Got {basis=}')

    def reset_parameters(self):
        std = 1 / np_sqrt(self.modes[-1] * self.modes[-2])
        torch.nn.init.normal_(self.weights, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Take a resolution and outputs the positional encodings"""
        size_x, size_y = x.shape[-2], x.shape[-1]
        if self.basis == 'sht':
            if self.transform.nlat == size_x and self.transform.nlon == size_y:
                return self.transform(self.weights)

            self.transform = th.InverseRealSHT(
                size_x,
                size_y,
                lmax=self.modes[-2],
                mmax=self.modes[-1],
                grid="legendre-gauss",
                norm='backward'
            ).to(
                device=self.weights.device,
                dtype=self.weights.dtype
            )
            return self.transform(self.weights)

        if self.basis == 'fft':
            return torch.fft.ifft2(self.weights, s=(size_x, size_y)).real

        raise ValueError(f'Expected one of "fft" or "sht". Got {self.basis=}')


# SHT doesn't make sense for 3 variables
class FourierVariableEncoding3D(nn.Module):
    def __init__(self, n_features: int, modes: Tuple[int, ...]) -> None:
        super().__init__()
        if len(modes) != 3:
            raise ValueError(
                f"Expected 3 frequency modes, but got {len(modes)} modes:\n{modes=}")

        self.modes = modes
        self.weights = nn.Parameter(
            torch.empty(n_features, *modes, dtype=torch.cfloat)
        )
        self.reset_parameters()

    def reset_parameters(self):
        std = 1 / np_sqrt(reduce(lambda a, b: a * b, self.modes))
        torch.nn.init.normal_(self.weights, mean=0.0, std=std)

    def forward(self, x_shape: torch.Size):
        """Take a resolution and outputs the positional encodings"""
        *_, size_t, size_x, size_y = x_shape
        return torch.fft.ifftn(
            self.weights,
            s=(size_t, size_x, size_y),
            norm="forward",  # don't multiply by any normalization factor
        ).real


# NOTE: this doesn't actually use the above `*VariableEncoding*` transformations;
# instead it just uses the canonical sin/cos-based positional encoding -
# in particular, the one provided by `neuralop`.
class VariableEncodingIrregularMesh(nn.Module):
    def __init__(
        self,
        n_variables: int,
        variable_encoding_size: int,
        n_dim: int = 2,
        positional_encoding_dim: int = 8
    ) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.variable_encoding_size = variable_encoding_size
        self.n_dim = n_dim
        self.positional_encoding_dim = positional_encoding_dim

        self.variable_encoder = MLPLinear(
            [  # layers
                self.n_dim + self.n_dim * self.positional_encoding_dim,
                self.self.variable_encoding_size * self.n_variables
            ]
        )
        self.positional_embedding = PositionalEmbedding(
            num_channels=self.positional_encoding_dim)

    def forward(self, grid_points):
        positional_embedding = self.positional_embedding(grid_points.reshape(-1))
        positional_embedding = positional_embedding.reshape(grid_points.shape[0], -1)
        grid_and_embedding = torch.cat([grid_points, positional_embedding], dim=1)
        return self.variable_encoder(grid_and_embedding)


class VariableEncodingWrapper(nn.Module):
    def __init__(
            self,
            equation_dict: dict,
            variable_encoding_size: int,
            n_dim: int = 2,
            positional_encoding_dim: int = 8,
    ) -> None:
        """
        For each equation in ``equation_dict``, we create
        a ``VariableEncodingIrregularMesh`` dict is of the type:
        ``Dict[Equation, int]``:
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
                positional_encoding_dim=positional_encoding_dim
            )

    def load_encoder(self, equation: str, path: str):
        self.model_dict[equation].load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    def save_encoder(self, equation: str, path: str):
        torch.save(self.model_dict[equation].state_dict(), path)

    def save_all_encoders(self, path: str):
        for i in self.equation_dict.keys():
            torch.save(self.model_dict[i].state_dict(), path + f"_{i}"+".pt")

    def freeze(self, equation: str):
        for param in self.model_dict[equation].parameters():
            param.requires_grad = False
            
    def forward(self, grid_points, equation: Optional[str] = None):
        encoding_list = []
        if equation is None:
            equation = list(self.equation_dict.keys())
        for i in equation:
            encoding_list.append(self.model_dict[i](grid_points))
        return torch.cat(encoding_list, dim=1).unsqueeze(0)


def get_variable_encoder(params):
    return VariableEncodingWrapper(
        params.equation_dict,
        variable_encoding_size=params.n_encoding_channels,
        n_dim=params.n_dim,
        positional_encoding_dim=params.positional_encoding_dim,
    )
