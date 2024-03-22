import pickle
import random
import h5py
import os
import torch
import numpy as np
from torch.utils.data import ConcatDataset, random_split, DataLoader
from utils import *
from neuralop.datasets.tensor_dataset import TensorDataset
from data_utils import get_mesh_displacement


class IrregularMeshTensorDataset(TensorDataset):
    def __init__(
        self,
        x,
        y,
        transform_x=None,
        transform_y=None,
        equation=None,
        x1=0,
        x2=0,
        mu=0.1,
        mesh=None,
    ):
        super().__init__(x, y, transform_x, transform_y)
        self.x1 = x1
        self.x2 = x2

        self.mu = mu
        self.mesh = mesh
        self.equation = equation
        self._creat_static_features()

    def _creat_static_features(self, d_grid=None):
        """
        creating static channels for inlet and reynolds number 
        """
        n_grid_points = self.x.shape[1]
        if len(self.equation) == 1:
            # equation can be either  ['NS'] or ['NS', 'ES']
            # of 3 or 5 channels/varibales
            n_variables = 3
        else:
            n_variables = self.x.shape[-1]
        if d_grid is not None:
            positional_enco = self.mesh + d_grid
        else:
            positional_enco = self.mesh

        raynolds = torch.ones(n_grid_points, 1) * self.mu
        inlet = (
            (-self.x1 / 2 + positional_enco[:, 1])
            * (-self.x2 / 2 + positional_enco[:, 1])
        )[:, None] ** 2

        self.static_features = torch.cat(
            [raynolds, inlet, positional_enco], dim=-1
        ).repeat(1, n_variables)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        d_grid_x = get_mesh_displacement(x)
        d_grid_y = get_mesh_displacement(y)

        self._creat_static_features(d_grid_x)

        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            y = self.transform_y(y)

        if len(self.equation) == 1:
            # Assumeing equaiion is NS
            x = x[:, :3]
            y = y[:, :3]

        return {
            "x": x,
            "y": y,
            "d_grid_x": d_grid_x,
            "d_grid_y": d_grid_y,
            "static_features": self.static_features,
            "equation": self.equation,
        }


class Normalizer:
    """
    A class that performs normalization and denormalization operations on data.

    Args:
        mean (torch.Tensor): The mean values used for normalization.
        std (torch.Tensor): The standard deviation values used for normalization.
        eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-6.
        persample (bool, optional): If True, performs instance normalization. Defaults to False.
    """

    def __init__(self, mean, std, eps=1e-6, persample=False):
        self.persample = persample
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        """
        Normalize the input data.

        Args:
            data (torch.Tensor): The input data to be normalized.

        Returns:
            torch.Tensor: The normalized data.
        """
        if self.persample:
            self.mean = torch.mean(data, dim=(0))
            self.std = torch.var(data, dim=(0)) ** 0.5
        return (data - self.mean) / (self.std + self.eps)

    def denormalize(self, data):
        """
        Denormalize the input data.

        Args:
            data (torch.Tensor): The input data to be denormalized.

        Returns:
            torch.Tensor: The denormalized data.
        """
        return data * (self.std + self.eps) + self.mean

    def cuda(self):
        """
        Move the mean and std tensors to the GPU.

        Returns:
            None
        """
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()


class NsElasticDataset:
    """
    A class representing a dataset for solving the Navier-Stokes equations on an elastic domain.

    Args:
        location (str): The location of the dataset.
        equation (str): The equation to be solved.
        mesh_location (str): The location of the mesh.
        params (object): An object containing various parameters.

    Attributes:
        location (str): The location of the dataset.
        _x1 (list): The x1 values.
        _x2 (list): The x2 values.
        _mu (list): The mu values.
        equation (str): The equation to be solved.
        input_mesh (torch.Tensor): The input mesh.
        normalizer (Normalizer): The normalizer object.

    Methods:
        _readh5(h5f, dtype=torch.float32): Reads data from an h5 file.
        get_data(mu, x1, x2): Gets the data for a given mu, x1, and x2.
        get_dataloader(mu_list, dt, normalize=True, batch_size=1, train_test_split=0.2, sample_per_inlet=200, ntrain=None, ntest=None, data_loader_kwargs={'num_workers': 2}): Gets the dataloader for the dataset.
        get_tensor_dataset(mu, dt, normalize=True, min_max_normalize=False, train_test_split=0.2, sample_per_inlet=200, x1_list=None, x2_list=None): Gets the tensor dataset for a given mu.

    """

    def __init__(self, location, equation, mesh_location, params):
        self.location = location
        self._x1 = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
        self._x2 = [-4.0, -2.0, 0, 2.0, 4.0, 6.0]
        self._mu = [0.1, 0.01, 0.5, 5, 1, 10]
        if params.data_partition == "supervised":
            self._x1 = params.supervised_inlets_x1
            self._x2 = params.supervised_inlets_x2
        elif params.data_partition == "self-supervised":
            self._x1 = list(set(self._x1) - set(params.supervised_inlets_x1))
            self._x2 = list(set(self._x2) - set(params.supervised_inlets_x2))
        else:
            raise ValueError(f"Data partition {params.data_partition} not supported")

        self.equation = equation

        mesh = get_mesh(mesh_location)
        self.input_mesh = torch.from_numpy(mesh).type(torch.float)

        self.normalizer = Normalizer(None, None, persample=True)

    def _readh5(self, h5f, dtype=torch.float32):
        """
        Reads data from an h5 file.

        Args:
            h5f (h5py.File): The h5 file object.
            dtype (torch.dtype, optional): The data type of the tensor. Defaults to torch.float32.

        Returns:
            torch.Tensor: The tensor containing the data.
        """
        a_dset_keys = list(h5f["VisualisationVector"].keys())
        size = len(a_dset_keys)
        readings = [None for i in range(size)]
        for dset in a_dset_keys:
            ds_data = h5f["VisualisationVector"][dset]
            if ds_data.dtype == "float64":
                csvfmt = "%.18e"
            elif ds_data.dtype == "int64":
                csvfmt = "%.10d"
            else:
                csvfmt = "%s"
            readings[int(dset)] = torch.tensor(np.array(ds_data), dtype=dtype)

        readings_tensor = torch.stack(readings, dim=0)
        print(f"Loaded tensor Size: {readings_tensor.shape}")
        return readings_tensor

    def get_data(self, mu, x1, x2):
        """
        Gets the data for a given mu, x1, and x2.

        Args:
            mu (float): The value of mu.
            x1 (float): The value of x1.
            x2 (float): The value of x2.

        Returns:
            tuple: A tuple containing the velocity tensor, pressure tensor, and displacements tensor.
        """
        if mu not in self._mu:
            raise ValueError(f"Value of mu must be one of {self._mu}")
        if x1 not in self._x1 or x2 not in self._x2:
            raise ValueError(
                f"Value of is must be one of {self._ivals3} and {self._ivals12} "
            )
        path = os.path.join(
            self.location,
            "mu=" + str(mu),
            "x1=" + str(x1),
            "x2=" + str(x2),
            "Visualization",
        )

        filename = os.path.join(path, "displacement.h5")

        h5f = h5py.File(filename, "r")
        displacements_tensor = self._readh5(h5f)

        filename = os.path.join(path, "pressure.h5")
        h5f = h5py.File(filename, "r")
        pressure_tensor = self._readh5(h5f)

        filename = os.path.join(path, "velocity.h5")
        h5f = h5py.File(filename, "r")
        velocity_tensor = self._readh5(h5f)

        return velocity_tensor, pressure_tensor, displacements_tensor

    def get_dataloader(
        self,
        mu_list,
        dt,
        normalize=True,
        batch_size=1,
        train_test_split=0.2,
        sample_per_inlet=200,
        ntrain=None,
        ntest=None,
        data_loader_kwargs={"num_workers": 2},
    ):
        """
        Gets the dataloader for the dataset.

        Args:
            mu_list (list): The list of mu values.
            dt (int): The time step.
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            batch_size (int, optional): The batch size. Defaults to 1.
            train_test_split (float, optional): The train-test split ratio. Defaults to 0.2.
            sample_per_inlet (int, optional): The number of samples per inlet. Defaults to 200.
            ntrain (int, optional): The number of training samples. Defaults to None.
            ntest (int, optional): The number of testing samples. Defaults to None.
            data_loader_kwargs (dict, optional): Additional keyword arguments for the DataLoader. Defaults to {'num_workers': 2}.

        Returns:
            tuple: A tuple containing the training dataloader and testing dataloader.
        """
        train_datasets = []
        test_datasets = []

        for mu in mu_list:
            train, test = self.get_tensor_dataset(
                mu,
                dt,
                normalize,
                train_test_split=train_test_split,
                sample_per_inlet=sample_per_inlet,
            )
            train_datasets.append(train)
            test_datasets.append(test)
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)
        print("****Train dataset size***: ", len(train_dataset))
        print("***Test dataset size***: ", len(test_dataset))
        if ntrain is not None:
            train_dataset = random_split(
                train_dataset, [ntrain, len(train_dataset) - ntrain]
            )[0]
        if ntest is not None:
            test_dataset = random_split(
                test_dataset, [ntest, len(test_dataset) - ntest]
            )[0]

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, **data_loader_kwargs
        )
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, **data_loader_kwargs
        )

        return train_dataloader, test_dataloader

    def get_tensor_dataset(
        self,
        mu,
        dt,
        normalize=True,
        min_max_normalize=False,
        train_test_split=0.2,
        sample_per_inlet=200,
        x1_list=None,
        x2_list=None,
    ):
        """
        Gets the tensor dataset for a given mu.

        Args:
            mu (float): The value of mu.
            dt (int): The time step.
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            min_max_normalize (bool, optional): Whether to use min-max normalization. Defaults to False.
            train_test_split (float, optional): The train-test split ratio. Defaults to 0.2.
            sample_per_inlet (int, optional): The number of samples per inlet. Defaults to 200.
            x1_list (list, optional): The list of x1 values. Defaults to None.
            x2_list (list, optional): The list of x2 values. Defaults to None.

        Returns:
            tuple: A tuple containing the training dataset and testing dataset.
        """
        if x1_list is None:
            x1_list = self._x1
        if x2_list is None:
            x2_list = self._x2

        train_datasets = []
        test_datasets = []
        for x1 in x1_list:
            for x2 in x2_list:
                try:
                    velocities, pressure, displacements = self.get_data(mu, x1, x2)
                except FileNotFoundError:
                    print(f"File not found for mu={mu}, x1={x1}, x2={x2}")
                    continue
                # keeping vx,xy, P, dx,dy
                varable_idices = [0, 1, 3, 4, 5]
                combined = torch.cat([velocities, pressure, displacements], dim=-1)[
                    :sample_per_inlet, :, varable_idices
                ]

                step_t0 = combined[:-dt, ...]
                step_t1 = combined[dt:, ...]

                indexs = [i for i in range(step_t0.shape[0])]

                ntrain = int((1 - train_test_split) * len(indexs))
                ntest = len(indexs) - ntrain

                random.shuffle(indexs)
                train_t0, test_t0 = (
                    step_t0[indexs[:ntrain]],
                    step_t0[indexs[ntrain : ntrain + ntest]],
                )
                train_t1, test_t1 = (
                    step_t1[indexs[:ntrain]],
                    step_t1[indexs[ntrain : ntrain + ntest]],
                )

                if not normalize:
                    normalizer = None
                else:
                    if not min_max_normalize:
                        mean, var = (
                            torch.mean(train_t0, dim=(0, 1)),
                            torch.var(train_t0, dim=(0, 1)) ** 0.5,
                        )
                    else:
                        mean = torch.min(train_t0.view(-1, train_t0.shape[-1]), dim=0)[
                            0
                        ]
                        var = (
                            torch.max(train_t0.view(-1, train_t0.shape[-1]), dim=0)[0]
                            - torch.min(train_t0.view(-1, train_t0.shape[-1]), dim=0)[0]
                        )

                    normalizer = Normalizer(mean, var)

                train_datasets.append(
                    IrregularMeshTensorDataset(
                        train_t0,
                        train_t1,
                        normalizer,
                        normalizer,
                        x1=x1,
                        x2=x2,
                        mu=mu,
                        equation=self.equation,
                        mesh=self.input_mesh,
                    )
                )
                test_datasets.append(
                    IrregularMeshTensorDataset(
                        test_t0,
                        test_t1,
                        normalizer,
                        normalizer,
                        x1=x1,
                        x2=x2,
                        mu=mu,
                        equation=self.equation,
                        mesh=self.input_mesh,
                    )
                )

        return ConcatDataset(train_datasets), ConcatDataset(test_datasets)


class DatasetSimple:
    """
    A class representing a simple dataset.

    Attributes:
        normalizer: An object used for normalizing the data.

    Methods:
        get_onestep_dataloader: Returns train and test data loaders for one-step prediction.
    """

    def __init__(self):
        self.normalizer = None

    def get_onestep_dataloader(
        self,
        train_test_split=0.2,
        location="../Data/MP_data/",
        batch_size=1,
        dtype=torch.float,
        dt=1,
        ntrain=None,
        ntest=None,
    ):
        """
        Returns train and test data loaders for one-step prediction.

        Args:
            train_test_split (float): The ratio of data to be used for training. Default is 0.2.
            location (str): The location of the data files. Default is '../Data/MP_data/'.
            batch_size (int): The batch size for the data loaders. Default is 1.
            dtype (torch.dtype): The data type of the tensors. Default is torch.float.
            dt (int): The time step difference between t0 and t1. Default is 1.
            ntrain (int): The number of training samples. If not provided, it is calculated based on train_test_split.
            ntest (int): The number of test samples. If not provided, it is calculated based on train_test_split.

        Returns:
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            test_loader (torch.utils.data.DataLoader): The data loader for test data.
        """
        with open(location + "displacements0-5000.pkl", "rb") as file:
            displacements = torch.tensor(pickle.load(file), dtype=dtype)
        with open(location + "pressures0-5000.pkl", "rb") as file:
            # scaler variable
            pressure = torch.tensor(pickle.load(file), dtype=dtype)[:, :, None]
        with open(location + "velocitoes0-5000.pkl", "rb") as file:
            velocities = torch.tensor(pickle.load(file), dtype=dtype)
        varable_idices = [0, 1, 3, 4, 5]
        combined = torch.cat([velocities, pressure, displacements], dim=-1)[
            :, :, varable_idices
        ]
        step_t0 = combined[:-dt, ...]
        step_t1 = combined[dt:, ...]

        indexs = [i for i in range(step_t0.shape[0])]
        if not ntrain:
            ntrain = int(train_test_split * len(indexs))
        if not ntest:
            ntest = len(indexs) - ntrain

        random.shuffle(indexs)
        train_t0, test_t0 = (
            step_t0[indexs[:ntrain]],
            step_t0[indexs[ntrain : ntrain + ntest]],
        )
        train_t1, test_t1 = (
            step_t1[indexs[:ntrain]],
            step_t1[indexs[ntrain : ntrain + ntest]],
        )

        mean, var = (
            torch.mean(train_t0, dim=(0, 1)),
            torch.mean(torch.var(train_t0, dim=(1)), dim=0),
        )

        normalizer = Normalizer(mean, var ** 0.5)

        # setting normalizer
        self.normalizer = normalizer

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_t0, train_t1),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_t0, test_t1),
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
    """
    Create dummy data loaders for training and testing.

    Args:
        train_test_split (float): The ratio of training data to testing data.
        channels (int): The number of channels in the input data.
        resolution (int): The resolution of the input data.
        location (str): The location of the data.
        batch_size (int): The batch size for the data loaders.
        dtype (torch.dtype): The data type of the input data.

    Returns:
        tuple: A tuple containing the training and testing data loaders.
    """
    train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.rand(1000, channels, resolution, resolution),
            torch.rand(1000, channels, resolution, resolution),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    test = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.rand(500, channels, resolution, resolution),
            torch.rand(500, channels, resolution, resolution),
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    return train, test
