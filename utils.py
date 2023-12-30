import os
import torch
import torch.nn as nn
import h5py
import numpy as np

def get_wandb_api_key(api_key_file="config/wandb_api_key.txt"):
    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()

class TokenExpansion(nn.Module):
    def __init__(self, n_variables: int, n_encoding_channels, n_static_channels: int) -> None:
        super().__init__()
        self.n_variables = n_variables
        self.n_encoding_channels = n_encoding_channels
        self.n_static_channels = n_static_channels

        expansion_factor = 1 + self.n_static_channels + self.n_encoding_channels

        self.variable_channels = [
            i * expansion_factor for i in range(n_variables)]
        self.static_channels = []
        if self.n_static_channels != 0:
            for v in self.variable_channels:
                self.static_channels.extend(
                    range(v + 1, v + self.n_static_channels + 1))
        self.encoding_channels = []
        if self.n_encoding_channels != 0:
            self.encoding_channels = sorted(list(
                set(range(n_variables * expansion_factor))
                - set(self.variable_channels)
                - set(self.static_channels)
            ))
        
        print(self.variable_channels)
        print(self.static_channels)
        print(self.encoding_channels)

    def forward(self, inp: torch.Tensor, variable_encodings: torch.tensor, static_channels: torch.tensor) -> torch.Tensor:
        """
        x: (batch_size, n_variables)
        """
        x = torch.zeros((inp.shape[0], inp.shape[1], len(
                self.variable_channels) + len(self.encoding_channels) + len(self.static_channels)), device=inp.device, dtype=inp.dtype)
        x[:, :, self.variable_channels] = inp
        if self.n_static_channels != 0:
            x[:, :, self.static_channels] = static_channels.repeat(x.shape[0], 1, 1)
        if self.n_encoding_channels != 0:
            x[:, :, self.encoding_channels] = variable_encodings.repeat(x.shape[0], 1, 1)

        return x

def get_mesh(location):
    """Get the mesh from a location."""
    h5f = h5py.File(location, 'r')
    mesh = h5f['mesh/coordinates']
    return mesh[:]