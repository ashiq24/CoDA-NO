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


class SWEDataset:
    """
    Represents one HDF5 dataset of 2D Shallow Water Equation trajectories
    from PDEBench.

    For each dataset on .h5 file:
    data=<HDF5 dataset "data": shape (101, 128, 128, 2), type "<f4">
    with shape (T, W, H, C), channels=["activator", "inhibitor"]
    T is linearly spaced in [0.0, 5.0], inclusive, with steps=101 and step_size=0.05.
    X and Y are between: [-0.9941406, 0.9902344]
    """

    TRAJECTORY_LENGTH = 10
    """Allow for 5 frames in input and 5 in output."""

    TIME_DURATION = 101
    """Each sample contains 101 time frames."""

    SAMPLE_SIZE = 1000
    """Each file contains 1,000 samples."""

    def __init__(
        self,
        path: str,
        train_test_split=1.0,
        subsampling_rate=None,
        stride=TRAJECTORY_LENGTH,
        offset=0,
    ):
        self.path = path
        self.file = h5py.File(path)
        self._subsampling_rate = subsampling_rate
        self.stride = stride
        self.offset = offset

        CLS = self.__class__
        self.items_per_sample: int = (
            (CLS.TIME_DURATION - self.offset) //
            (CLS.TRAJECTORY_LENGTH + self.stride))
        self.len = int(
            self.items_per_sample * np.floor(CLS.SAMPLE_SIZE * train_test_split))

        # expect strings like: "0000", ..., "0999"
        self.samples = list(self.file.keys())

    @property
    def subsampling_rate(self):
        return (
            self._subsampling_rate
            if self._subsampling_rate is not None
            else 1
        )

    @subsampling_rate.setter
    def subsampling_rate(self, value):
        self._subsampling_rate = value

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Pair:
        if index >= self.len:
            raise IndexError(f"Cannot access item {index} of {self.len}")

        CLS = self.__class__
        sample_idx, local_idx = divmod(index, self.items_per_sample)
        time_idx = self.offset + local_idx * (CLS.TRAJECTORY_LENGTH + self.stride)

        # try:
        sample = self._reconstruct_sample(
            self.file,
            sample_idx,
            time_idx,
            t_steps=CLS.TRAJECTORY_LENGTH,
        )
        # except:
        #    raise RuntimeError

        mid = CLS.TRAJECTORY_LENGTH // 2
        return Pair(torch.as_tensor(sample[:mid]), torch.as_tensor(sample[mid:]))

    def _reconstruct_sample(
        self,
        h5_file,
        sample_idx,
        time_idx,
        t_steps,
    ) -> np.ndarray:
        """
        Retrieves a sample from a SWE trajectory.

        Reconstructs a sample from the target H5 file from times
        ``[time_idx, time_idx + t_steps)`` (inclusive/exclusive, respectively).

        Shape: (time, x, y, channel)
        NOTE: channel is a singleton dimension in this case representing only
          water depth.
        """
        sample_key = self.samples[sample_idx]
        return h5_file[sample_key]['data'][
            time_idx : time_idx+t_steps,
            ::self.subsampling_rate,
            ::self.subsampling_rate
        ]


class DiffusionReaction2DDataset:
    """
    Represents one HDF5 dataset of 2D Diffusion-Reaction trajectories from PDEBench.

    For each dataset on .h5 file:
    data=<HDF5 dataset "data": shape (101, 128, 128, 2), type "<f4">
    with shape (T, W, H, C), channels=["activator", "inhibitor"]
    T is linearly spaced in [0.0, 5.0], inclusive, with steps=101 and step_size=0.05
    X and Y are between: [-0.9941406, 0.9902344]
    """

    TRAJECTORY_LENGTH = 10
    """Allow for 5 frames in input and 5 in output."""

    TIME_DURATION = 101
    """Each sample contains 101 time frames."""

    SAMPLE_SIZE = 1000
    """Each file contains 1,000 samples."""

    def __init__(
        self,
        path: str,
        train_test_split=1.0,
        subsampling_rate=None,
        stride=TRAJECTORY_LENGTH,
        offset=0,
    ):
        self.path = path
        self.file = h5py.File(path)
        self._subsampling_rate = subsampling_rate
        self.stride = stride
        self.offset = offset

        CLS = self.__class__
        self.items_per_sample: int = (
            (CLS.TIME_DURATION - self.offset) //
            (CLS.TRAJECTORY_LENGTH + self.stride))
        self.len = int(
            self.items_per_sample * np.floor(CLS.SAMPLE_SIZE * train_test_split))

        # expect strings like: "0000", ..., "0999"
        self.samples = list(self.file.keys())

    @property
    def subsampling_rate(self):
        return (
            self._subsampling_rate
            if self._subsampling_rate is not None
            else 1
        )

    @subsampling_rate.setter
    def subsampling_rate(self, value):
        self._subsampling_rate = value

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Pair:
        if index >= self.len:
            raise IndexError(f"Cannot access item {index} of {self.len}")

        CLS = self.__class__
        sample_idx, local_idx = divmod(index, self.items_per_sample)
        time_idx = self.offset + local_idx * (CLS.TRAJECTORY_LENGTH + self.stride)

        # try:
        sample = self._reconstruct_sample(
            self.file,
            sample_idx,
            time_idx,
            t_steps=CLS.TRAJECTORY_LENGTH,
        )
        # except:
        #    raise RuntimeError

        mid = CLS.TRAJECTORY_LENGTH // 2
        return Pair(torch.as_tensor(sample[:mid]), torch.as_tensor(sample[mid:]))

    def _reconstruct_sample(
        self,
        h5_file,
        sample_idx,
        time_idx,
        t_steps,
    ) -> np.ndarray:
        """
        Reconstructs a sample with ordering: "activator", "inhibitor"

        Reconstructs a sample from the target H5 file from times
        ``[time_idx, time_idx + t_steps)`` (inclusive/exclusive, respectively).

        Shape: (time, x, y, channel)
        """
        sample_key = self.samples[sample_idx]
        return h5_file[sample_key]['data'][
            time_idx : time_idx+t_steps,
            ::self.subsampling_rate,
            ::self.subsampling_rate
        ]


class NSIncompressibleSample(NamedTuple):
    particles: np.ndarray
    velocity: np.ndarray
    force: np.ndarray


class NSIncompressibleDataset:
    """
    Represents one or more HDF5 datasets of incompressible Navier-Stokes
    trajectories from PDEBench.

    For a .h5 file on disk:
    * f_ns['force']=
      <HDF5 dataset "force": shape (4, 512, 512, 2), type "<f4">
    * f_ns['particles']=
      <HDF5 dataset "particles": shape (4, 1000, 512, 512, 1), type "<f4">
    * f_ns['t']=
      <HDF5 dataset "t": shape (4, 1000), type "<f4">
    * f_ns['velocity']=
      <HDF5 dataset "velocity": shape (4, 1000, 512, 512, 2), type "<f4">

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
        self._subsampling_rate = subsampling_rate
        self.stride = stride
        self.offset = offset

        CLS = self.__class__
        # This isn't quite right (ex. with offset=stride=TRAJECTORY_LENGTH,
        # there should still be 50 items, but this would result in 49). Maybe
        # padding TIME_DURATION with an additional stride?
        self.items_per_file: int = np.floor(
            ((CLS.TIME_DURATION - self.offset) //
             (CLS.TRAJECTORY_LENGTH + self.stride))
            * train_test_split)
        # Each item within the file has 4 samples
        self.len = int(len(paths) * self.items_per_file * 4)

    @property
    def subsampling_rate(self):
        return (
            self._subsampling_rate
            if self._subsampling_rate is not None
            else 1
        )

    @subsampling_rate.setter
    def subsampling_rate(self, value):
        self._subsampling_rate = value

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Pair:
        if index >= self.len:
            raise IndexError(f"Cannot access item f{index} of f{self.len}")

        CLS = self.__class__
        # Each item within the file has 4 samples
        index2, sample_idx = divmod(index, 4)
        # file_idx : which file we should read from,
        # local_idx : which index to address within that file.
        file_idx, local_idx = divmod(index2, self.items_per_file)

        time_idx = int(self.offset + local_idx * (
            CLS.TRAJECTORY_LENGTH + self.stride))

        # try:
        # print(self.files[int(file_idx)], sample_idx, time_idx, CLS.TRAJECTORY_LENGTH)
        sample = self._reconstruct_sample(
            self.files[int(file_idx)],
            sample_idx,
            time_idx,
            t_steps=CLS.TRAJECTORY_LENGTH,
        )
        trajectory = np.concatenate([sample.particles, sample.velocity], axis=-1)
        # except:
        #    raise RuntimeError

        mid = CLS.TRAJECTORY_LENGTH // 2
        return Pair(
            torch.as_tensor(trajectory[:mid]),
            torch.as_tensor(trajectory[mid:])
        )

    def _reconstruct_sample(
        self,
        h5_file,
        sample_idx,
        time_idx,
        t_steps,
    ) -> NSIncompressibleSample:
        """
        Reconstructs a sample with ordering: "particles", [Vx, Vy], force

        Reconstructs a sample from the target H5 file from times
        ``[time_idx, time_idx + t_steps)`` (inclusive/exclusive, respectively).

        Shape (within each field) : (time, x, y, channel)
        """
        particles = h5_file['particles'][
            sample_idx,
            time_idx : time_idx+t_steps,
            ::self.subsampling_rate,
            ::self.subsampling_rate,
        ]
        velocity = h5_file['velocity'][
            sample_idx,
            time_idx : time_idx+t_steps,
            ::self.subsampling_rate,
            ::self.subsampling_rate,
        ]
        force = h5_file['force'][sample_idx]

        return NSIncompressibleSample(particles, velocity, force)
