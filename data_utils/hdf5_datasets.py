"""
Author: Mogab Elleithy <github.com/m4e7>

Inspired by:
https://github.com/PolymathicAI/multiple_physics_pretraining/
blob/45918d1ac2c50a876a3aa36d837e3c199dfc08ba/
data_utils/hdf5_datasets.py#L259
"""
from typing import List, NamedTuple

import h5py
import numpy as np
import torch


class Pair(NamedTuple):
    input: torch.Tensor
    """Tensor to be consumed by ``model.forward(x)``"""
    output: torch.Tensor
    """Ground-truth tensor against model's prediction."""


class NSIncompressibleSample(NamedTuple):
    particles: np.ndarray
    velocity: np.ndarray
    force: np.ndarray


class NSIncompressibleDataset:
    """
    Represents one or more HDF5 datasets of incompressible Navier-Stokes
    trajectories from PDEBench.

    Parameters
    ---
    stride : int
        How many time frames to skip between items.
        Ex. ``stride=TRAJECTORY_LENGTH`` would skip every other item.
    offset : int
        Time-offset from which to start reading data. This is the same across
        all data files.
    """

    TRAJECTORY_LENGTH = 10
    """Allow for 5 frames in input and 5 in output."""

    TIME_DURATION = 1000
    """Each file contains 1,000 time frames."""

    # Try to use strides to "interleave" training and testing data.
    # That allows for 50 training datum in one 1000-step trajectory.
    # E.g. the following train/test would be interleaved:
    # train = NSDataset(stride=10, offset=0)
    # test  = NSDataset(stride=10, offset=10)
    # or interleaved 2:1
    # train1 = NSDataset(stride=20, offset=0)
    # train2 = NSDataset(stride=20, offset=10)
    # test   = NSDataset(stride=20, offset=20)

    def __init__(
        self,
        paths: List[str],
        train_test_split=1.0,
        subsampling_rate=None,
        stride=TRAJECTORY_LENGTH,
        offset=0,
    ):
        self.paths = paths
        self.files = [h5py.File(p) for p in paths]
        self.subsampling_rate = subsampling_rate
        self.stride = stride
        self.offset = offset

        # This isn't quite right (ex. with offset=stride=TRAJECTORY_LENGTH,
        # there should still be 50 items, but this would result in 49). Maybe
        # padding TIME_DURATION with an additional stride?
        self.items_per_file: int = np.floor(
            ((NSIncompressibleDataset.TIME_DURATION - self.offset) //
             (NSIncompressibleDataset.TRAJECTORY_LENGTH + self.stride))
            * train_test_split)
        # Each item within the file has 4 samples
        self.len = len(paths) * self.items_per_file * 4

    def __getitem__(self, index) -> Pair:
        if index >= self.len:
            raise IndexError(f"Cannot access item f{index} of f{self.len}")

        # Each item within the file has 4 samples
        index2, sample_idx = divmod(index, 4)
        # file_idx : which file we should read from,
        # local_idx : which index to address within that file.
        file_idx, local_idx = divmod(index2, self.items_per_file)

        time_idx = self.offset + local_idx * (
            NSIncompressibleDataset.TRAJECTORY_LENGTH + self.stride)

        try:
            # print(self.files[file_idx], sample_idx, time_idx, index)
            sample = self._reconstruct_sample(
                self.files[file_idx],
                sample_idx,
                time_idx,
                t_steps=NSIncompressibleDataset.TRAJECTORY_LENGTH,
            )
            trajectory = np.concatenate([sample.particles, sample.velocity], axis=-1)
        except:
            raise RuntimeError(
                f'Failed to reconstruct sample for file {self.paths[file_idx]} '
                f'sample {sample_idx} time {time_idx}')

        mid = NSIncompressibleDataset.TRAJECTORY_LENGTH // 2
        return Pair(
            torch.as_tensor(trajectory[:mid]),
            torch.as_tensor(trajectory[mid:])
        )

    def __len__(self):
        return self.len

    def _reconstruct_sample(
        self,
        h5_file,
        sample_idx,
        time_idx,
        t_steps,
    ) -> NSIncompressibleSample:
        """
        Reconstructs a sample with ordering: "particles", Vx, Vy

        Reconstructs a sample from the target H5 file from times
        ``[time_idx, time_idx + t_steps)`` (inclusive/exclusive, respectively).

        Shape: (time, x, y, channel)
        """
        subsampling_rate = (
            self.subsampling_rate
            if self.subsampling_rate is not None
            else 1
        )
        _slice = [
            sample_idx,
            slice(time_idx, time_idx + t_steps),
            slice(None, None, subsampling_rate),
            slice(None, None, subsampling_rate),
        ]
        particles = h5_file['particles'][_slice]
        velocity = h5_file['velocity'][_slice]
        force = h5_file['force'][sample_idx]

        return NSIncompressibleSample(particles, velocity, force)
