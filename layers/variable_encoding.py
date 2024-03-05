from functools import reduce
import operator
from typing import Tuple, Literal, Optional, Callable

from numpy import sqrt as np_sqrt
import torch
from torch import nn
import torch_harmonics as th

from neuralop.layers.mlp import MLPLinear
from neuralop.layers.embeddings import PositionalEmbedding


class _VariableEncoding(nn.Module):
    sh_transform: Optional[th.InverseRealSHT]

    # Assumes subclass args are validated before being passed to super
    def __init__(
        self,
        n_features: int,
        modes: Tuple[int, ...],
        basis: Optional[Literal["fft", "sht"]],
    ):
        super().__init__()
        self.n_features = n_features
        self.modes = modes
        self.basis = basis

        self.weights = nn.Parameter(
            torch.empty(self.n_features, *self.modes, dtype=torch.cfloat)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        sigma = 1 / np_sqrt(reduce(operator.mul, self.modes))
        torch.nn.init.normal_(self.weights, mean=0.0, std=sigma)


class VariableEncoding2D(_VariableEncoding):
    def __init__(
        self,
        channel: int,
        modes: Tuple[int, ...],
        basis: Literal["fft", "sht"] = 'fft',
    ):
        if len(modes) != 2:
            raise ValueError(
                f"Expected 2 frequency modes, but got {len(modes)} modes:\n{modes=}")

        if basis not in ("fft", "sht"):
            raise ValueError(f'Expected one of "fft" or "sht", but got {basis=}')

        super().__init__(
            n_features=channel,
            modes=modes,
            basis=basis,
        )

        if basis == 'sht':
            # unpacks like nlat, nlon = self.modes
            self.compute_isht(*self.modes)
        # else basis == fft: leave transform == None

    def compute_isht(self, nlat: int, nlon: int) -> None:
        self.sh_transform = th.InverseRealSHT(
            nlat=nlat,
            nlon=nlon,
            lmax=self.modes[-2],
            mmax=self.modes[-1],
            grid="legendre-gauss",
            norm="backward",
        )

    def forward(self, x_shape: torch.Size) -> torch.Tensor:
        """Take a resolution and outputs the positional encodings"""
        *_, size_x, size_y = x_shape
        if self.basis == 'sht':
            if self.sh_transform.nlat == size_x and self.sh_transform.nlon == size_y:
                return self.sh_transform(self.weights)

            self.compute_isht(size_x, size_y)
            self.sh_transform.to(
                device=self.weights.device, dtype=self.weights.dtype
            )
            return self.sh_transform(self.weights)

        if self.basis == 'fft':
            return torch.fft.irfft2(self.weights, s=(size_x, size_y))

        raise ValueError(f'Expected one of "fft" or "sht". Got {self.basis=}')


# SHT doesn't make sense for 3 variables
class FourierVariableEncoding3D(_VariableEncoding):
    def __init__(self, n_features: int, modes: Tuple[int, ...]) -> None:
        if len(modes) != 3:
            raise ValueError(
                f"Expected 3 frequency modes, but got {len(modes)} modes:\n{modes=}")

        super().__init__(
            n_features=n_features,
            modes=modes,
            basis="fft",
        )

    def forward(self, x_shape: torch.Size):
        """Take a resolution and outputs the positional encodings"""
        *_, size_t, size_x, size_y = x_shape
        return torch.fft.irfftn(
            self.weights,
            s=(size_t, size_x, size_y),
            norm="forward",  # don't multiply by any normalization factor
        )


# NOTE: this doesn't actually use the above `*VariableEncoding*` transformations;
# instead it just uses the canonical sin/cos-based positional encoding:
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
            [  # `layers` - i.e. dimensionality per layer
                self.n_dim + self.n_dim * self.positional_encoding_dim,
                self.self.variable_encoding_size * self.n_variables
            ]
        )
        self.positional_embedding = PositionalEmbedding(
            num_channels=self.positional_encoding_dim)

    def forward(self, grid_points):
        """Mixes up positional embeddings according to a learned transformation."""
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
