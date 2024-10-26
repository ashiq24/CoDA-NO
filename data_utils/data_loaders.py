import pickle
import random
import h5py
import os
import torch
import numpy as np
from torchvision.transforms import Normalize
from torch.utils.data import ConcatDataset, random_split, DataLoader
import itertools
from utils import *
from neuralop.datasets.tensor_dataset import TensorDataset
from data_utils import get_mesh_displacement


# data loaders of Releigh-Berneard and Navier-Stokes


class regularMeshTensorDataset(TensorDataset):
    def __init__(
            self,
            x,
            y,
            transform_x=None,
            transform_y=None,
            equation=None):
        '''
        data in format
        x: samples x channels x x_grid_size x y_grid_size
        '''
        super().__init__(x, y, transform_x, transform_y)
        self.equation = equation
        x_grid_size = x[0].shape[-2]
        y_grid_size = x[0].shape[-1]

        self.static_features = self.generate_static_grid(
            x_grid_size, y_grid_size)

    def generate_static_grid(self, x_grid_size, y_grid_size):
        '''
        creat grid of resolution x_grid_size x y_grid_size
        '''
        x = torch.linspace(-1, 1, x_grid_size)
        y = torch.linspace(-1, 1, y_grid_size)
        x, y = torch.meshgrid(x, y)
        return torch.permute(torch.stack([x, y], dim=-1), (2, 0, 1))

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            y = self.transform_y(y)

        return {'x': x, 'y': y, 'equation': self.equation,
                'static_features': self.static_features}


def load_NS_dataset(path, n_samples, subsamplingrate=2):
    data_1 = torch.load(path[0])
    data_2 = torch.load(path[1])

    x1_t0 = data_1[:, :-1, :, :]
    x1_t1 = data_1[:, 1:, :, :]
    x2_t0 = data_2[:, :-1, :, :]
    x2_t1 = data_2[:, 1:, :, :]

    # flatten the data
    x1_t0 = x1_t0.reshape(-1, x1_t0.shape[-3],
                          x1_t0.shape[-2], x1_t0.shape[-1])
    x1_t1 = x1_t1.reshape(-1, x1_t0.shape[-3],
                          x1_t1.shape[-2], x1_t1.shape[-1])
    x2_t0 = x2_t0.reshape(-1, x1_t0.shape[-3],
                          x2_t0.shape[-2], x2_t0.shape[-1])
    x2_t1 = x2_t1.reshape(-1, x1_t0.shape[-3],
                          x2_t1.shape[-2], x2_t1.shape[-1])

    # shuffel the data
    indices = torch.randperm(x1_t0.shape[0])
    x1_t0 = x1_t0[indices]
    x1_t1 = x1_t1[indices]
    idx = torch.randperm(x2_t0.shape[0])
    x2_t0 = x2_t0[indices]
    x2_t1 = x2_t1[indices]

    ntain1 = int(n_samples / 2)
    ntain2 = n_samples - ntain1

    x1_train = x1_t0[:ntain1]
    y1_train = x1_t1[:ntain1]
    x2_train = x2_t0[:ntain2]
    y2_train = x2_t1[:ntain2]

    # concatenate the data
    x = torch.cat([x1_train, x2_train], dim=0)
    y = torch.cat([y1_train, y2_train], dim=0)

    return x.permute(0, 3, 1, 2), y.permute(0, 3, 1, 2)


def load_NS_dataset_hdf5(path, n_samples, subsamplingrate=2):
    '''
    loading only velocity field data
    '''
    samples = 0
    i = 0
    data_x = []
    data_y = []

    while samples < n_samples + 100:
        if i > 999:
            break
        data_path = path[0] + 'data_' + str(i) + '.hdf5'
        x = h5py.File(data_path, "r")['data']
        data_x.append(x[:: subsamplingrate, :: subsamplingrate, :-1, :2])
        data_y.append(x[:: subsamplingrate, :: subsamplingrate, 1:, :2])

        data_path = path[1] + 'data_' + str(i) + '.hdf5'
        x = h5py.File(data_path, "r")['data']
        data_x.append(x[:: subsamplingrate, :: subsamplingrate, :-1, :2])
        data_y.append(x[:: subsamplingrate, :: subsamplingrate, 1:, :2])

        i += 1
        samples += (x.shape[2] - 1)

    x = torch.tensor(np.concatenate(data_x, axis=2))
    y = torch.tensor(np.concatenate(data_y, axis=2))

    print("Data Loaded: ", x.shape, y.shape)

    return torch.permute(x, (2, 3, 0, 1)), torch.permute(y, (2, 3, 0, 1))


def get_NS_dataloader(params):
    n_samples = params.n_train + params.n_test
    x, y = load_NS_dataset(params.data_location,
                           n_samples, params.subsampling_rate)

    # shuffel and split test train
    indices = torch.randperm(x.shape[0])
    x = x[indices]
    x_train = x[:params.n_train]
    y_train = y[:params.n_train]

    x_test = x[params.n_train:]
    y_test = y[params.n_train:]

    # get the mean and std of the data
    mean = torch.mean(x_train, dim=(0, 1, 2, 3))
    std = torch.std(x_train, dim=(0, 1, 2, 3))
    normalizer = Normalize(mean, std)

    dataset_train = regularMeshTensorDataset(
        x_train,
        y_train,
        transform_x=normalizer,
        transform_y=normalizer,
        equation=['NS'])
    dataset_test = regularMeshTensorDataset(
        x_test,
        y_test,
        transform_x=normalizer,
        transform_y=normalizer,
        equation=['NS'])

    dat_train = DataLoader(
        dataset_train, batch_size=params.batch_size, shuffle=True)
    dat_test = DataLoader(
        dataset_test, batch_size=params.batch_size, shuffle=False)
    return dat_train, dat_test


def get_RB_dataloader(params):
    data = np.load(params.data_location)
    # files ['vx', 'vy', 'temp', 'time']
    vx = torch.tensor(data['vx']).type(torch.float)
    vy = torch.tensor(data['vy']).type(torch.float)
    temp = torch.tensor(data['temp']).type(torch.float)
    time = torch.tensor(data['time']).type(torch.float)

    # stack the data
    data = torch.stack([vx, vy, temp], dim=1)
    data = data[params.skip_start:]
    x = data[:int(-1 * params.dt)]
    y = data[int(params.dt):]

    # shuffel and split test train
    # fix the seed
    torch.manual_seed(params.random_seed)
    indices = torch.randperm(x.shape[0])
    x = x[indices]
    y = y[indices]

    x_train = x[:params.n_train, :,
                ::params.subsampling_rate, ::params.subsampling_rate]
    y_train = y[:params.n_train, :,
                ::params.subsampling_rate, ::params.subsampling_rate]

    x_test = x[-params.n_test:]
    y_test = y[-params.n_test:]
    print("len test data", len(x_test), len(y_test), params.n_test, x.shape)

    # get the mean and std of the data
    mean = torch.mean(x_train, dim=(0, 2, 3))
    std = torch.std(x_train, dim=(0, 2, 3))
    normalizer = Normalize(mean, std)

    dataset_train = regularMeshTensorDataset(
        x_train,
        y_train,
        transform_x=normalizer,
        transform_y=normalizer,
        equation=['ES'])
    dataset_test = regularMeshTensorDataset(
        x_test,
        y_test,
        transform_x=normalizer,
        transform_y=normalizer,
        equation=['ES'])

    dat_train = DataLoader(
        dataset_train, batch_size=params.batch_size, shuffle=True)
    dat_test = DataLoader(
        dataset_test, batch_size=params.batch_size, shuffle=False)

    return dat_train, dat_test


# dataloader for Fluid Sturctur Interaction (FSI) problems

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
            mesh=None):
        super().__init__(x, y, transform_x, transform_y)
        self.x1 = x1
        self.x2 = x2

        self.mu = mu
        self.mesh = mesh
        self.equation = equation
        print("Inside Dataset :", self.mesh.dtype, x.dtype, x.dtype)
        self._creat_static_features()

    def _creat_static_features(self, d_grid=None):
        '''
        creating static channels for inlet and reynolds number
        '''
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
        inlet = ((-self.x1 / 2 + positional_enco[:, 1]) *
                 (-self.x2 / 2 + positional_enco[:, 1]))[:, None]**2

        self.static_features = torch.cat(
            [raynolds, inlet, positional_enco], dim=-1).repeat(1, n_variables)

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
            x = x[:, :3]
            y = y[:, :3]

        return {'x': x, 'y': y, 'd_grid_x': d_grid_x,
                'd_grid_y': d_grid_y, 'static_features': self.static_features,
                'equation': self.equation}


class Normalizer():
    def __init__(self, mean, std, eps=1e-6, persample=False):
        self.persample = persample
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        if self.persample:
            self.mean = torch.mean(data, dim=(0))
            self.std = torch.var(data, dim=(0))**0.5
        return (data - self.mean) / (self.std + self.eps)

    def denormalize(self, data):
        return data * (self.std + self.eps) + self.mean

    def cuda(self,):
        if self.mean is not None and self.std is not None:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()


class NsElasticDataset():
    def __init__(self, location, equation, mesh_location, params):
        self.location = location

        # _x1 and _x2 are the paraemters for the inlets condtions
        # _mu is the visocity
        self._x1 = [-4.0, -2.0, 0.0, 2.0, 4.0, 6.0]
        self._x2 = [-4.0, -2.0, 0, 2.0, 4.0, 6.0]
        self._mu = [0.1, 0.01, 0.5, 5, 1, 10]
        if params.data_partition == 'supervised':
            # held out 2 inlets for finetuning
            # there not introduced in the self-supevised
            # pretraining
            self._x1 = params.supervised_inlets_x1
            self._x2 = params.supervised_inlets_x2
        elif params.data_partition == 'self-supervised':
            self._x1 = list(set(self._x1) - set(params.supervised_inlets_x1))
            self._x2 = list(set(self._x2) - set(params.supervised_inlets_x2))
        else:
            raise ValueError(
                f"Data partition {params.data_partition} not supported")

        self.equation = equation

        mesh = get_mesh(params)
        self.input_mesh = torch.from_numpy(mesh).type(torch.float)
        print("Mesh Shape: ", self.input_mesh.shape)
        self.params = params

        self.normalizer = Normalizer(None, None, persample=True)

    def _readh5(self, h5f, dtype=torch.float32):
        a_dset_keys = list(h5f['VisualisationVector'].keys())
        size = len(a_dset_keys)
        readings = [None for i in range(size)]
        for dset in a_dset_keys:
            ds_data = (h5f['VisualisationVector'][dset])
            readings[int(dset)] = torch.tensor(np.array(ds_data), dtype=dtype)

        readings_tensor = torch.stack(readings, dim=0)
        print(f"Loaded tensor Size: {readings_tensor.shape}")
        return readings_tensor

    def get_data(self, mu, x1, x2):
        if mu not in self._mu:
            raise ValueError(f"Value of mu must be one of {self._mu}")
        if x1 not in self._x1 or x2 not in self._x2:
            raise ValueError(
                f"Value of is must be one of {self._ivals3} and {self._ivals12} ")
        if mu == 0.5:
            path = os.path.join(
                self.location,
                'mu=' + str(mu),
                'x1=' + str(-4.0),
                'x2=' + str(x2),
                '1',
                'Visualization')
            print(path)
        else:
            path = os.path.join(
                self.location,
                'mu=' + str(mu),
                'x1=' + str(x1),
                'x2=' + str(x2),
                'Visualization')

        filename = os.path.join(path, 'displacement.h5')

        h5f = h5py.File(filename, 'r')
        displacements_tensor = self._readh5(h5f)

        filename = os.path.join(path, 'pressure.h5')
        h5f = h5py.File(filename, 'r')
        pressure_tensor = self._readh5(h5f)

        filename = os.path.join(path, 'velocity.h5')
        h5f = h5py.File(filename, 'r')
        velocity_tensor = self._readh5(h5f)

        return velocity_tensor, pressure_tensor, displacements_tensor

    def get_data_txt(self, mu, x1, x2):
        if mu not in self._mu:
            raise ValueError(f"Value of mu must be one of {self._mu}")
        if x1 not in self._x1 or x2 not in self._x2:
            raise ValueError(
                f"Value of is must be one of {self._ivals3} and {self._ivals12} ")
        path = os.path.join(
            self.location,
            'mu=' + str(mu),
            'x1=' + str(x1),
            'x2=' + str(x2),
            '1')

        velocity_x = torch.tensor(np.loadtxt(os.path.join(path, 'vel_x.txt')))
        velocity_y = torch.tensor(np.loadtxt(os.path.join(path, 'vel_y.txt')))
        if len(self.params.equation_dict) != 1:
            dis_x = torch.tensor(np.loadtxt(os.path.join(path, 'dis_x.txt')))
            dis_y = torch.tensor(np.loadtxt(os.path.join(path, 'dis_y.txt')))
            pressure = torch.tensor(np.loadtxt(os.path.join(path, 'pres.txt')))
        else:
            # just copying values as place holder when only NS equation is used
            dis_x = velocity_x
            dis_y = velocity_y
            pressure = velocity_x

        # reshape each tensor into 2d by keeping 876 entries in each row
        dis_x = dis_x.view(-1, 876, 1)
        dis_y = dis_y.view(-1, 876, 1)
        pressure = pressure.view(-1, 876, 1)
        velocity_x = velocity_x.view(-1, 876, 1)
        velocity_y = velocity_y.view(-1, 876, 1)

        velocity = torch.cat([velocity_x, velocity_y], dim=-1)
        displacement = torch.cat([dis_x, dis_y], dim=-1)

        return velocity.to(
            torch.float), pressure.to(
            torch.float), displacement.to(
            torch.float)

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
            data_loader_kwargs={'num_workers': 2}):

        train_datasets = []
        test_datasets = []

        for mu in mu_list:
            train, test = self.get_tensor_dataset(
                mu, dt, normalize, train_test_split=train_test_split, sample_per_inlet=sample_per_inlet)
            train_datasets.append(train)
            test_datasets.append(test)
        train_dataset = ConcatDataset(train_datasets)
        test_dataset = ConcatDataset(test_datasets)
        print("****Train dataset size***: ", len(train_dataset))
        print("***Test dataset size***: ", len(test_dataset))
        if ntrain is not None:
            train_dataset = random_split(
                train_dataset, [ntrain, len(train_dataset) - ntrain])[0]
        if ntest is not None:
            test_dataset = random_split(
                test_dataset, [ntest, len(test_dataset) - ntest])[0]

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, **data_loader_kwargs)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, **data_loader_kwargs)

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
            x2_list=None):

        if x1_list is None:
            x1_list = self._x1
        if x2_list is None:
            x2_list = self._x2
        train_datasets = []
        test_datasets = []
        # for the given mu
        # loop over all given inlets
        for x1 in x1_list:
            for x2 in x2_list:
                try:
                    if mu == 0.5:
                        velocities, pressure, displacements = self.get_data_txt(
                            mu, x1, x2)
                    else:
                        velocities, pressure, displacements = self.get_data(
                            mu, x1, x2)
                except FileNotFoundError as e:
                    print(e)
                    print(
                        f"Original file not found for mu={mu}, x1={x1}, x2={x2}")
                    continue

                # keeping vx,xy, P, dx,dy
                varable_idices = [0, 1, 3, 4, 5]
                if mu == 0.5:
                    combined = torch.cat(
                        [velocities, pressure, displacements], dim=-1)[:sample_per_inlet, :, :]
                else:
                    combined = torch.cat(
                        [velocities, pressure, displacements], dim=-1)[:sample_per_inlet, :, varable_idices]

                if hasattr(
                        self.params,
                        'sub_sample_size') and self.params.sub_sample_size is not None:
                    mesh_size = combined.shape[1]
                    indexs = [i for i in range(mesh_size)]
                    np.random.seed(self.params.random_seed)
                    sub_indexs = np.random.choice(
                        indexs, self.params.sub_sample_size, replace=False)
                    combined = combined[:, sub_indexs, :]

                if self.params.super_resolution:
                    new_quieries = self.get_data_txt(
                        mu, x1, x2).to(dtype=combined.dtype)
                    new_quieries = new_quieries[:sample_per_inlet, :]

                    print("shape of old data", combined.shape)
                    print("shape of new data", new_quieries.shape)

                    combined = torch.cat([combined, new_quieries], dim=-2)
                    print("shape of combined data", combined.shape)

                step_t0 = combined[:-dt, ...]
                step_t1 = combined[dt:, ...]

                indexs = [i for i in range(step_t0.shape[0])]

                ntrain = int((1 - train_test_split) * len(indexs))
                ntest = len(indexs) - ntrain

                random.shuffle(indexs)
                train_t0, test_t0 = step_t0[indexs[:ntrain]
                                            ], step_t0[indexs[ntrain:ntrain + ntest]]
                train_t1, test_t1 = step_t1[indexs[:ntrain]
                                            ], step_t1[indexs[ntrain:ntrain + ntest]]

                if not normalize:
                    normalizer = None
                else:
                    if not min_max_normalize:
                        mean, var = torch.mean(train_t0, dim=(
                            0, 1)), torch.var(train_t0, dim=(0, 1))**0.5
                    else:
                        mean = torch.min(
                            train_t0.view(-1, train_t0.shape[-1]), dim=0)[0]
                        var = torch.max(train_t0.view(-1,
                                                      train_t0.shape[-1]),
                                        dim=0)[0] - torch.min(train_t0.view(-1,
                                                                            train_t0.shape[-1]),
                                                              dim=0)[0]

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
                        mesh=self.input_mesh))
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
                        mesh=self.input_mesh))

        return ConcatDataset(train_datasets), ConcatDataset(test_datasets)
