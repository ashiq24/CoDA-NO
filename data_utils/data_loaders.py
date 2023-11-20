import pickle
import random

import torch
from torch.utils import data

# TODO expose Normalizer in neuralop.datasets.transforms
class Normalizer:
    def __init__(self, mean, std, eps=1e-6):
        # print(mean)
        # print(std)
        self.mean = mean
        self.std = std
        if self.std < 0:
            raise ValueError(f"Cannot have a negative standard deviation: {std=}")
        self.eps = eps

    def __call__(self, data):
        return (data - self.mean) / (self.std + self.eps)


def get_onestep_dataloader(
    train_test_split=0.2,
    location='../Data/MP_data/',
    batch_size=1,
    dtype=torch.float,
    n_train=None,
    n_test=None,
):
    with open(location + 'displacements0-5000.pkl', 'rb') as file:
        displacements = torch.tensor(pickle.load(file), dtype=dtype)
    with open(location + 'pressures0-5000.pkl', 'rb') as file:
        # scaler variable
        pressure = torch.tensor(pickle.load(file), dtype=dtype)[:, :, None]
    with open(location + 'velocitoes0-5000.pkl', 'rb') as file:
        velocities = torch.tensor(pickle.load(file), dtype=dtype)

    combined = torch.cat([displacements, velocities, pressure], dim=-1)
    step_t0 = combined[:-1, ...]
    step_t1 = combined[1:, ...]

    indexes = [i for i in range(step_t0.shape[0])]
    if not n_train:
        n_train = int(train_test_split * len(indexes))
    if not n_test:
        n_test = len(indexes) - n_train
    random.shuffle(indexes)
    train_t0 = step_t0[indexes[:n_train]]
    test_t0 = step_t0[indexes[n_train:n_train + n_test]]
    train_t1 = step_t1[indexes[:n_train]]
    test_t1 = step_t1[indexes[n_train:n_train + n_test]]

    mean, var = torch.mean(train_t0, dim=(0, 1)), torch.var(train_t0, dim=(0, 1))
    normalizer = Normalizer(mean, var**0.5)
    train_loader = data.DataLoader(
        data.TensorDataset(normalizer(train_t0), normalizer(train_t1)),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = data.DataLoader(
        data.TensorDataset(normalizer(test_t0), normalizer(test_t1)),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def get_dummy_dataloaders(
    train_test_split=0.2,
    channels=7,
    resolution=256,
    location=None,
    batch_size=32,
    dtype=torch.float,
):
    train = data.DataLoader(
        data.TensorDataset(
            torch.rand(1000, channels, resolution, resolution),
            torch.rand(1000, channels, resolution, resolution),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test = data.DataLoader(
        data.TensorDataset(
            torch.rand(500, channels, resolution, resolution),
            torch.rand(500, channels, resolution, resolution),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train, test
