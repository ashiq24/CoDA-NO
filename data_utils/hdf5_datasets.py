"""
Author: Mogab Elleithy <github.com/m4e7>

``SWEDataset``, ``DiffusionReactionDataset``, and ``NSDataset`` inspired by:
https://github.com/PolymathicAI/multiple_physics_pretraining/
blob/45918d1ac2c50a876a3aa36d837e3c199dfc08ba/
data_utils/hdf5_datasets.py#L259
"""
import enum
from typing import Dict, List, NamedTuple, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils import data

from data_utils.data_loaders import Normalizer

DEBUG = False


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
        filepath: str,
        subsampling_rate=None,
        strides_on=1,
        strides_off=1,
        offset=0,
        sample_size=None,
        predictive=True,
    ):
        """
        Given one sample (i.e. a time-trajectory from an initial condition),
        it may be divided up into one or more chunks of size
        ``TRAJECTORY_LENGTH * (stride_on + stride_off)``. In such a chunk the
        first ``TRAJECTORY_LENGTH * stride_on`` will be included (i.e. yielded
        when this dataset is iterated over), and the last
        ``TRAJECTORY_LENGTH * stride_off`` will be excluded, or skipped.

        Parameters
        ---
        strides_on: int
            How many items, in strides of size ``TRAJECTORY_LENGTH``, to include.
        strides_off: int
            How many items, in strides of size ``TRAJECTORY_LENGTH``, to exclude.
        offset: int
            Temporal offset to be applied to each sample.
        """
        self.path = filepath
        self.file = h5py.File(filepath)
        self._subsampling_rate = subsampling_rate

        self.strides_on = strides_on
        self.strides_off = strides_off
        self.offset = offset
        self.predictive = predictive

        CLS = self.__class__
        if sample_size is None:
            sample_size = CLS.SAMPLE_SIZE
        if 0 >= sample_size or sample_size > CLS.SAMPLE_SIZE:
            raise ValueError(
                f"Illegal value for {sample_size=}. "
                f"Expected `sample_size` to be within [1, {CLS.SAMPLE_SIZE}]"
            )
        self.sample_size = sample_size

        # We need to add this excluded "pace length" so that we don't
        # erroneously truncate the last included pace length if it's only the
        # excluded pace that doesn't fit. For example:
        # [ ..................... 50 ..................... ]
        # [ ...... 20 ...... ][ ...... 20 ...... ][ . 10 . ]
        # [ . 10 . ][ . 10 . ][ . 10 . ][ . 10 . ][ . 10 . ]
        # Consider the above scenario where:
        # * time_duration=50
        # * trajectory_length=10
        # * stride_on, stride_off=1, 1
        # We want the class to recognize 3 "on" strides. We do not want the 3rd
        # stride to be dropped because it couldn't fit the last "off" stride in.

        stride_length_on = CLS.TRAJECTORY_LENGTH * strides_on
        stride_length_off = CLS.TRAJECTORY_LENGTH * strides_off
        self.items_per_stride = stride_length_on + stride_length_off
        self.strides_per_sample: int = (
            (CLS.TIME_DURATION + stride_length_off - self.offset)
            // self.items_per_stride
        )
        self.len = int(
            self.strides_per_sample * self.strides_on * np.floor(self.sample_size)
        )

        # expect strings like: "0000", ..., "0999"
        self.samples: List[str] = list(self.file.keys())[: self.sample_size]
        self.normalizers: Dict[Tuple[str, int], Normalizer] = {}

    @property
    def subsampling_rate(self):
        return self._subsampling_rate if self._subsampling_rate is not None else 1

    @subsampling_rate.setter
    def subsampling_rate(self, value):
        self._subsampling_rate = value

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if index >= self.len:
            raise IndexError(f"Cannot access item {index} of {self.len}")

        CLS = self.__class__
        sample_idx, k = divmod(index, self.strides_per_sample * self.strides_on)
        stride_idx, local_idx = divmod(k, self.strides_on)
        time_idx = (
            self.offset
            + (stride_idx * self.items_per_stride)
            + (local_idx * CLS.TRAJECTORY_LENGTH)
        )
        if DEBUG:
            print(f"{sample_idx=}, {k=}, {stride_idx=}, {local_idx=}, {time_idx=}")

        if self.predictive:
            # Fetch the full trajectory:
            sample = self._reconstruct_sample(
                sample_idx, time_idx, t_steps=CLS.TRAJECTORY_LENGTH,
            )

            mid = CLS.TRAJECTORY_LENGTH // 2
            return torch.as_tensor(sample[:mid]), torch.as_tensor(sample[mid:])

        else:
            # Only fetch the first half of a trajectory.
            # As a reconstructive dataset, we want a model to learn an
            # encoder/decoder embedding of the data.
            sample = self._reconstruct_sample(
                sample_idx, time_idx, t_steps=CLS.TRAJECTORY_LENGTH // 2,
            )
            return torch.as_tensor(sample), torch.as_tensor(sample)

    def _reconstruct_sample(self, sample_idx, time_idx, t_steps,) -> np.ndarray:
        """
        Retrieves a sample from a SWE trajectory.

        Reconstructs a normalized sample from the target H5 file from times
        ``[time_idx, time_idx + t_steps)`` (inclusive/exclusive, respectively).

        Shape: (time, x, y, channel)
        NOTE: channel is a singleton dimension in this case representing only
          water depth.
        """
        sample_key = self.samples[sample_idx]
        norm = self.get_normalizer(sample_key, time_idx)
        datum = self.file[sample_key]["data"][
            time_idx : time_idx + t_steps,
            :: self.subsampling_rate,
            :: self.subsampling_rate,
        ]
        return norm(datum)

    def get_normalizer(self, key: str, index: int) -> Normalizer:
        if self.normalizers.get((key, index)) is None:
            # Normalize over only the data that is "input,"
            # including respecting subsampling:
            steps = self.__class__.TRAJECTORY_LENGTH // 2
            datum = self.file[key]["data"][
                index : index + steps,
                :: self.subsampling_rate,
                :: self.subsampling_rate,
            ]
            norm = Normalizer(np.mean(datum), np.std(datum))
            # Remember the normalizer to we can recover the original data.
            self.normalizers[(key, index)] = norm
        return self.normalizers.get((key, index))


# The first time frame of each sample is a noisy initial
# condition, but then all following time frames look substantially different.
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
        filepath: str,
        subsampling_rate=None,
        strides_on=1,
        strides_off=1,
        offset=0,
        sample_size=None,
        predictive=True,
    ):
        """
        Given one sample (i.e. a time-trajectory from an initial condition),
        it may be divided up into one or more chunks of size
        ``TRAJECTORY_LENGTH * (stride_on + stride_off)``. In such a chunk the
        first ``TRAJECTORY_LENGTH * stride_on`` will be included (i.e. yielded
        when this dataset is iterated over), and the last
        ``TRAJECTORY_LENGTH * stride_off`` will be excluded, or skipped.

        Parameters
        ---
        strides_on: int
            How many items, in strides of size ``TRAJECTORY_LENGTH``, to include.
        strides_off: int
            How many items, in strides of size ``TRAJECTORY_LENGTH``, to exclude.
        offset: int
            Temporal offset to be applied to each sample.
        """
        self.path = filepath
        self.file = h5py.File(filepath)
        self._subsampling_rate = subsampling_rate

        self.strides_on = strides_on
        self.strides_off = strides_off
        self.offset = offset
        self.predictive = predictive

        CLS = self.__class__
        if sample_size is None:
            sample_size = CLS.SAMPLE_SIZE
        if 0 >= sample_size or sample_size > CLS.SAMPLE_SIZE:
            raise ValueError(
                f"Illegal value for {sample_size=}. "
                f"Expected `sample_size` to be within [1, {CLS.SAMPLE_SIZE}]"
            )
        self.sample_size = sample_size

        # We need to add this excluded "pace length" so that we don't
        # erroneously truncate the last included pace length if it's only the
        # excluded pace that doesn't fit. For example:
        # [ ..................... 50 ..................... ]
        # [ ...... 20 ...... ][ ...... 20 ...... ][ . 10 . ]
        # [ . 10 . ][ . 10 . ][ . 10 . ][ . 10 . ][ . 10 . ]
        # Consider the above scenario where:
        # * time_duration=50
        # * trajectory_length=10
        # * stride_on, stride_off=1, 1
        # We want the class to recognize 3 "on" strides. We do not want the 3rd
        # stride to be dropped because it couldn't fit the last "off" stride in.

        CLS = self.__class__
        stride_length_on = CLS.TRAJECTORY_LENGTH * strides_on
        stride_length_off = CLS.TRAJECTORY_LENGTH * strides_off
        self.items_per_stride = stride_length_on + stride_length_off
        self.strides_per_sample: int = (
            (CLS.TIME_DURATION + stride_length_off - self.offset)
            // self.items_per_stride
        )
        self.len = int(
            self.strides_per_sample * self.strides_on * np.floor(self.sample_size)
        )

        # expect strings like: "0000", ..., "0999"
        self.samples: List[str] = list(self.file.keys())[:sample_size]
        self.normalizers: Dict[Tuple[str, int, int], Normalizer] = {}

    @property
    def subsampling_rate(self):
        return self._subsampling_rate if self._subsampling_rate is not None else 1

    @subsampling_rate.setter
    def subsampling_rate(self, value):
        self._subsampling_rate = value

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if index >= self.len:
            raise IndexError(f"Cannot access item {index} of {self.len}")

        CLS = self.__class__
        sample_idx, k = divmod(index, self.strides_per_sample * self.strides_on)
        stride_idx, local_idx = divmod(k, self.strides_on)
        time_idx = (
            self.offset
            + (stride_idx * self.items_per_stride)
            + (local_idx * CLS.TRAJECTORY_LENGTH)
        )
        if DEBUG:
            print(f"{sample_idx=}, {k=}, {stride_idx=}, {local_idx=}, {time_idx=}")

        if self.predictive:
            # Fetch the full trajectory:
            sample = self._reconstruct_sample(
                sample_idx, time_idx, t_steps=CLS.TRAJECTORY_LENGTH,
            )

            mid = CLS.TRAJECTORY_LENGTH // 2
            return torch.as_tensor(sample[:mid]), torch.as_tensor(sample[mid:])

        else:
            # Only fetch the first half of a trajectory.
            # As a reconstructive dataset, we want a model to learn an
            # encoder/decoder embedding of the data.
            sample = self._reconstruct_sample(
                sample_idx, time_idx, t_steps=CLS.TRAJECTORY_LENGTH // 2,
            )
            return torch.as_tensor(sample), torch.as_tensor(sample)

    def _reconstruct_sample(self, sample_idx, time_idx, t_steps,) -> np.ndarray:
        """
        Reconstructs a sample with ordering: "activator", "inhibitor"

        Reconstructs a sample from the target H5 file from times
        ``[time_idx, time_idx + t_steps)`` (inclusive/exclusive, respectively).

        Shape: (time, x, y, channel)
        """
        sample_key = self.samples[sample_idx]
        norm_activator = self.get_normalizer(sample_key, time_idx, 0)
        norm_inhibitor = self.get_normalizer(sample_key, time_idx, 1)
        data0 = self.file[sample_key]["data"][
            time_idx : time_idx + t_steps,
            :: self.subsampling_rate,
            :: self.subsampling_rate,
        ]
        activator = data0[..., 0]
        inhibitor = data0[..., 1]
        return np.concatenate(
            [
                np.expand_dims(norm_activator(activator), axis=-1),
                np.expand_dims(norm_inhibitor(inhibitor), axis=-1),
            ],
            axis=-1,
        )

    def get_normalizer(self, key: str, index: int, channel: int) -> Normalizer:
        if self.normalizers.get((key, index, channel)) is None:
            # Normalize over only the data that is "input,"
            # including respecting subsampling:
            steps = self.__class__.TRAJECTORY_LENGTH // 2
            datum = self.file[key]["data"][
                index : index + steps,
                :: self.subsampling_rate,
                :: self.subsampling_rate,
                channel,
            ]
            norm = Normalizer(np.mean(datum), np.std(datum))
            # Remember the normalizer to we can recover the original data.
            self.normalizers[(key, index, channel)] = norm
        return self.normalizers.get((key, index, channel))


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
    strides_on: int
        How many items, as units of size ``TRAJECTORY_LENGTH``, to include.
        Ex. ``stride_on=2`` would take 2 consecutive items per on/off chunk.
    strides_off: int
        How many items, as units of size ``TRAJECTORY_LENGTH``, to exclude.
        Ex. ``stride_off=1`` would skip 1 consecutive items per on/off chunk.
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
    # train = NSDataset(stride_on=1, stride_off=1, offset=0)
    # test  = NSDataset(stride_on=1, stride_off=1, offset=10)
    # or interleaved 2:1
    # train = NSDataset(stride_on=2, stride_off=1, offset=0)
    # test  = NSDataset(stride_on=1, stride_off=2, offset=20)

    def __init__(
        self,
        filepaths: List[str],
        subsampling_rate=None,
        strides_on=1,
        strides_off=1,
        offset=0,
        sample_size=None,
        predictive=True,
    ):
        self.paths = filepaths
        self.files = [h5py.File(p) for p in filepaths]
        self._subsampling_rate = subsampling_rate

        self.strides_on = strides_on
        self.strides_off = strides_off
        self.offset = offset
        self.predictive = predictive

        if sample_size is None:
            sample_size = 4
        if 0 >= sample_size or sample_size > 4:
            raise ValueError(
                f"Illegal value for {sample_size=}. "
                f"Expected `sample_size` to be within [1, {4}]"
            )
        self.sample_size = sample_size

        # We need to add this excluded "pace length" so that we don't
        # erroneously truncate the last included pace length if it's only the
        # excluded pace that doesn't fit. For example:
        # [ ..................... 50 ..................... ]
        # [ ...... 20 ...... ][ ...... 20 ...... ][ . 10 . ]
        # [ . 10 . ][ . 10 . ][ . 10 . ][ . 10 . ][ . 10 . ]
        # Consider the above scenario where:
        # * time_duration=50
        # * trajectory_length=10
        # * stride_on, stride_off=1, 1
        # We want the class to recognize 3 "on" strides. We do not want the 3rd
        # stride to be dropped because it couldn't fit the last "off" stride in.

        CLS = self.__class__
        stride_length_on = CLS.TRAJECTORY_LENGTH * strides_on
        stride_length_off = CLS.TRAJECTORY_LENGTH * strides_off
        self.items_per_stride = stride_length_on + stride_length_off
        self.strides_per_file: int = (
            (CLS.TIME_DURATION + stride_length_off - self.offset)
            // self.items_per_stride
        )
        # Each item within the file has 4 samples
        self.len = (
            len(filepaths) * self.strides_per_file * self.strides_on * self.sample_size
        )
        self.normalizers: Dict[Tuple[str, int, int, str], Normalizer] = {}

    @property
    def subsampling_rate(self):
        return self._subsampling_rate if self._subsampling_rate is not None else 1

    @subsampling_rate.setter
    def subsampling_rate(self, value):
        self._subsampling_rate = value

    def __len__(self):
        return self.len

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if index >= self.len:
            raise IndexError(f"Cannot access item f{index} of f{self.len}")

        # Each item within the file has 4 samples
        index2, sample_idx = divmod(index, 4)
        # file_idx : which file we should read from.
        file_idx, index3 = divmod(index2, self.strides_per_file * self.strides_on)
        # local_idx : which index to address within that file.
        stride_idx, local_idx = divmod(index3, self.strides_on)

        CLS = self.__class__
        time_idx = int(
            self.offset
            + (stride_idx * self.items_per_stride)
            + (local_idx * CLS.TRAJECTORY_LENGTH)
        )
        if DEBUG:
            print(
                f"{index2=}, "
                f"{sample_idx=}, "
                f"{file_idx=}, "
                f"{index3=}, "
                f"{stride_idx=}, "
                f"{local_idx=}, "
                f"{time_idx=}"
            )

        if self.predictive:
            sample = self._reconstruct_sample(
                int(file_idx), sample_idx, time_idx, t_steps=CLS.TRAJECTORY_LENGTH,
            )
            trajectory = np.concatenate([sample.particles, sample.velocity], axis=-1)

            mid = CLS.TRAJECTORY_LENGTH // 2
            return torch.as_tensor(trajectory[:mid]), torch.as_tensor(trajectory[mid:])

        else:
            sample = self._reconstruct_sample(
                int(file_idx), sample_idx, time_idx, t_steps=CLS.TRAJECTORY_LENGTH // 2,
            )
            trajectory = np.concatenate([sample.particles, sample.velocity], axis=-1)

            return torch.as_tensor(trajectory), torch.as_tensor(trajectory)

    def _reconstruct_sample(
        self, file_idx: int, sample_idx, time_idx, t_steps,
    ) -> NSIncompressibleSample:
        """
        Reconstructs a sample with ordering: "particles", [Vx, Vy], force

        Reconstructs a sample from the target H5 file from times
        ``[time_idx, time_idx + t_steps)`` (inclusive/exclusive, respectively).

        Shape (within each field) : (time, x, y, channel)
        """
        h5_file = self.files[file_idx]
        sample_idx = sample_idx % self.sample_size

        norm_p = self.get_normalizer(file_idx, sample_idx, time_idx, "particles")
        particles = h5_file["particles"][
            sample_idx,
            time_idx : time_idx + t_steps,
            :: self.subsampling_rate,
            :: self.subsampling_rate,
        ]

        norm_v = self.get_normalizer(file_idx, sample_idx, time_idx, "velocity")
        velocity = h5_file["velocity"][
            sample_idx,
            time_idx : time_idx + t_steps,
            :: self.subsampling_rate,
            :: self.subsampling_rate,
        ]

        # Skip normalizing the force. It has a different shape and is unused.
        # norm_f = self.get_normalizer(file_idx, sample_idx, 'force')
        force = h5_file["force"][
            sample_idx, :: self.subsampling_rate, :: self.subsampling_rate,
        ]

        return NSIncompressibleSample(norm_p(particles), norm_v(velocity), force,)

    def get_normalizer(self, path_idx, sample_idx: int, time_idx: int, channel: str):
        filepath = self.paths[path_idx]
        key = (filepath, sample_idx, time_idx, channel)
        t_steps = self.__class__.TRAJECTORY_LENGTH // 2

        if self.normalizers.get(key) is None:
            datum = self.files[path_idx][channel][
                sample_idx,
                time_idx : time_idx + t_steps,
                :: self.subsampling_rate,
                :: self.subsampling_rate,
            ]
            norm = Normalizer(np.mean(datum), np.std(datum))
            self.normalizers[key] = norm
        return self.normalizers.get(key)


def pad_with_noise(x, channels_before=0, channels_after=0, channel_dim=0):
    """Pad the given data with Gaussian noise in a given channel."""
    if channels_before > 0:
        prepended_shape = list(x.shape)
        prepended_shape[channel_dim] = channels_before
        prepended_noise = torch.randn(*prepended_shape)
        x = torch.concatenate([prepended_noise, x], dim=channel_dim)

    if channels_after > 0:
        appended_shape = list(x.shape)
        appended_shape[channel_dim] = channels_after
        appended_noise = torch.randn(*appended_shape)
        x = torch.concatenate([x, appended_noise], dim=channel_dim)

    return x


class Equation(enum.Enum):
    SWE = 0
    DIFF = 1
    NS = 2


class MultiPhysicsDataset(data.Dataset):
    """
    First approximation of a multi-physics dataset
    combining the Shallow Water equations, Diffusion Reaction,
    and (incompressible) Navier-Stokes.

    "First approximation" because each field is represented
    in its own input and output channel. Schematically:
            +--------+
    ---h--- |        | ------- water height
            |        |
    ---a--- |        | ------- activator
    ---i--- |  [GNO] | ------- inhibitor
            | CODANO |
    ---p--- |  [GNO] | ------- particle density
    --v_x-- |        | ------- velocity_x
    --v_x-- |        | ------- velocity_y
            +--------+

    Further refinements should allow for constructive mixing
    of all channels.
    """

    def __init__(
        self,
        swe_args,
        diff_args,
        ns_args,
        strides_on=1,
        strides_off=1,
        offset=0,
        channel_dim=0,
        predictive=True,
    ):
        """
        Each of the args dicts accepts optional keywords  "sample_size" and
        "subsampling_rate"

        Parameters
        ---
        swe_args : dict
            Requires keyword "filepath" with the absolute path to the SWE dataset.
        diff_args : dict
            Requires keyword "filepath" with the absolute path to the
            Diffusion-Reaction dataset.
        ns_args : dict
            Requires keyword "filepaths" with a list of absolute paths to
            Navier-Stokes dataset files.
        """
        if swe_args is None:
            swe_args = {}
        if diff_args is None:
            diff_args = {}
        if ns_args is None:
            ns_args = {}
        common_args = dict(
            strides_on=strides_on, strides_off=strides_off, predictive=predictive,
        )

        swe_args.update(offset=offset + swe_args.get("offset", default=0))
        diff_args.update(offset=offset + diff_args.get("offset", default=0))
        ns_args.update(offset=offset + ns_args.get("offset", default=0))

        self.swe_dataset = SWEDataset(**swe_args, **common_args,)
        self.diff_dataset = DiffusionReaction2DDataset(**diff_args, **common_args,)
        self.ns_dataset = NSIncompressibleDataset(**ns_args, **common_args,)
        self.channel_dim = channel_dim

    def __len__(self):
        return len(self.swe_dataset) + len(self.diff_dataset) + len(self.ns_dataset)

    def get_sampler_weights(self):
        """Returns a sequence of weights such that each dataset has a uniform
        chance of being drawn (despite differently sized datasets).
        """
        return (
            ([1 / (3 * len(self.swe_dataset))] * len(self.swe_dataset))
            + ([1 / (3 * len(self.diff_dataset))] * len(self.diff_dataset))
            + ([1 / (3 * len(self.ns_dataset))] * len(self.ns_dataset))
        )

    def __getitem__(
        self, idx
    ) -> Tuple[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int]]:
        """
        Returns fields A(x), U(x) across 6 variables.

        Shape: (C, T, W, H)
        Channels:
            height, (shallow water eqn)
            activator, inhibitor, (diffusion-reaction eqn)
            particle density, Vx, Vy (Navier-Stokes)

        "Pad" unused channels with Gaussian noise. This should teach
        the model to ignore these noisy channels without structure.
        """
        if idx < len(self.swe_dataset):
            # old shapes were like (T, W, H, C)
            swe_x, swe_y = self.swe_dataset[int(idx)]
            # new shapes will be like (C, T, W, H)
            swe_x = torch.permute(swe_x, (3, 0, 1, 2))
            swe_y = torch.permute(swe_y, (3, 0, 1, 2))

            return (
                # mark this datum as Shallow Water eqn
                (swe_x, Equation.SWE.value),
                (swe_y, Equation.SWE.value),
            )

        idx -= len(self.swe_dataset)
        if idx < len(self.diff_dataset):
            # old shape was like (T, W, H, C)
            diff_x, diff_y = self.diff_dataset[int(idx)]
            # new shapes will be like (C, T, W, H)
            diff_x = torch.permute(diff_x, (3, 0, 1, 2))
            diff_y = torch.permute(diff_y, (3, 0, 1, 2))

            # mark this datum as Diffusion-Reaction eqn
            return (
                (diff_x, Equation.DIFF.value),
                (diff_y, Equation.DIFF.value),
            )

        idx -= len(self.diff_dataset)
        if idx < len(self.ns_dataset):
            # old shape was like (T, W, H, C)
            ns_x, ns_y = self.ns_dataset[int(idx)]
            # new shapes will be like (C, T, W, H)
            ns_x = torch.permute(ns_x, (3, 0, 1, 2))
            ns_y = torch.permute(ns_y, (3, 0, 1, 2))

            return (
                (ns_x, Equation.NS.value),
                (ns_y, Equation.NS.value),
            )

        idx -= len(self.ns_dataset)
        raise IndexError(f"Cannot access item {idx + len(self)} of {len(self)}")
