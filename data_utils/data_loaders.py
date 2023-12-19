import pickle
import random
import h5py
import os
import torch
import numpy as np
from torchvision.transforms import Normalize
from torch.utils.data import ConcatDataset, random_split, DataLoader
import itertools
from neuralop.datasets.tensor_dataset import TensorDataset
from data_utils import get_mesh_displacement


class IrregularMeshTensorDataset(TensorDataset):
    def __init__(self, x, y, transform_x=None, transform_y=None, equation=None, i1=0,i2=0,i3=0, mu=0.1, mesh=None):
        super().__init__(x, y, transform_x, transform_y)
        self.i1 = i1
        self.i2 = i2
        self.i3 = i3
        self.mu = mu
        self.mesh = mesh
        self.equation = equation
        self._creat_static_features()
    
    def _creat_static_features(self,):
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
        raynolds = torch.ones(n_grid_points, 1)*self.mu
        inlet = (self.i1*torch.sin(self.mesh[:,1]) + self.i2*torch.sin(2*self.mesh[:,1]) + self.i3*torch.sin(3*self.mesh[:,1]))[:,None]**2
        self.static_features =  torch.cat([raynolds, inlet], dim=-1).repeat(1, n_variables)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        
        d_grid_x = get_mesh_displacement(x)
        d_grid_y = get_mesh_displacement(y)

        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            y = self.transform_y(y)

        if len(self.equation) == 1:
            # Assumeing equaiion is NS
            x = x[:,:3]
            y = y[:,:3]

        return {'x': x, 'y': y, 'd_grid_x': d_grid_x, 'd_grid_y': d_grid_y, 'static_features': self.static_features, 'equation': self.equation}


class Normalizer():
    def __init__(self, mean, std, eps=1e-6, persample=False):
        print("Means: ", mean)
        print("stds ", std)
        self.persample = persample # if true, instance norm type normalizer
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
    def __init__(self, location, equation, mesh_location='../Data/test_data/mesh.csv'):
        self.location = location
        self._ivals12 = [-0.5, 0, 1.0] # values related to inlet condition
        self._ivals3 = [-0.1, 0, 0.05]
        self._mu = [0.1, 0.01, 0.5, 1, 10] # Reynolds number
        self.equation = equation

        mesh = np.loadtxt(mesh_location, delimiter=',')
        self.input_mesh = torch.transpose(
            torch.stack([torch.tensor(mesh[0, :]), torch.tensor(mesh[1, :])]),
            dim0=0,
            dim1=1,
        ).type(torch.float)

        self.normalizer = Normalizer(None, None, persample=True)

    def _readh5(self, h5f, dtype=torch.float32):
        a_dset_keys = list(h5f['VisualisationVector'].keys())
        size = len(a_dset_keys)
        readings = [None for i in range(size)]
        for dset in a_dset_keys:
            ds_data = (h5f['VisualisationVector'][dset])
            if ds_data.dtype == 'float64':
                csvfmt = '%.18e'
            elif ds_data.dtype == 'int64':
                csvfmt = '%.10d'
            else:
                csvfmt = '%s'
            readings[int(dset)] = torch.tensor(np.array(ds_data), dtype=dtype)

        readings_tensor = torch.stack(readings, dim=0)
        print(f"Loaded tensor Size: {readings_tensor.shape}")
        return readings_tensor

    def get_data(self, mu, i1, i2, i3):
        if mu not in self._mu:
            raise ValueError(f"Value of mu must be one of {self._mu}")
        if i1 not in self._ivals12 or i2 not in self._ivals12 or i3 not in self._ivals3:
            raise ValueError(
                f"Value of is must be one of {self._ivals3} and {self._ivals12} ")
        path = os.path.join(
            self.location,
            'mu='+str(mu),
            'i1='+str(i1),
            'i2='+str(i2),
            'i3='+str(i3),
            'Visualization')

        filename = os.path.join(path, 'displacement.h5')
        #print(filename)
        h5f = h5py.File(filename, 'r')
        displacements_tensor = self._readh5(h5f)
        #print("Displacement Tensor", displacements_tensor.shape, np.max(displacements_tensor.numpy()), np.min(displacements_tensor.numpy()))

        filename = os.path.join(path, 'pressure.h5')
        h5f = h5py.File(filename, 'r')
        pressure_tensor = self._readh5(h5f)

        filename = os.path.join(path, 'velocity.h5')
        h5f = h5py.File(filename, 'r')
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

        if ntrain is not None:
            train_dataset = random_split(
                train_dataset, [ntrain, len(train_dataset)-ntrain])[0]
        if ntest is not None:
            test_dataset = random_split(
                test_dataset, [ntest, len(test_dataset)-ntest])[0]

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, **data_loader_kwargs)
        test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, **data_loader_kwargs)

        return train_dataloader, test_dataloader

    def get_tensor_dataset(self, mu, dt, normalize=True, train_test_split=0.2, sample_per_inlet=200):
        train_datasets = []
        test_datasets = []
        for i1 in self._ivals12:
            for i2 in self._ivals12:
                for i3 in self._ivals3:
                    velocities, pressure, displacements = self.get_data(
                        mu, i1, i2, i3)
                    # keeping vx,xy, P, dx,dy
                    varable_idices = [0, 1, 3, 4, 5]
                    combined = torch.cat(
                        [velocities, pressure, displacements], dim=-1)[:sample_per_inlet, :, varable_idices]

                    # print("sample data", combined[50,500,:])
                    step_t0 = combined[:-dt, ...]
                    step_t1 = combined[dt:, ...]

                    indexs = [i for i in range(step_t0.shape[0])]

                    ntrain = int((1-train_test_split) * len(indexs))
                    ntest = len(indexs) - ntrain

                    random.shuffle(indexs)
                    train_t0, test_t0 = step_t0[indexs[:ntrain]
                                                ], step_t0[indexs[ntrain:ntrain + ntest]]
                    train_t1, test_t1 = step_t1[indexs[:ntrain]
                                                ], step_t1[indexs[ntrain:ntrain + ntest]]

                    if not normalize:
                        normalizer = None
                    else:
                        mean, var = torch.mean(train_t0, dim=(
                            0, 1)), torch.var(train_t0, dim=(0, 1))**0.5
#                         mean= torch.min(train_t0.view(-1, train_t0.shape[-1]), dim=0)[0]
#                         var = torch.max(train_t0.view(-1, train_t0.shape[-1]), dim=0)[0]\
#                         - torch.min(train_t0.view(-1, train_t0.shape[-1]), dim=0)[0]

                        normalizer = Normalizer(mean, var)

                    train_datasets.append(
                        IrregularMeshTensorDataset(train_t0, train_t1, normalizer, normalizer, i1=i1, i2=i2, i3=i3, mu=mu, equation=self.equation, mesh=self.input_mesh))
                    test_datasets.append(
                        IrregularMeshTensorDataset(test_t0, test_t1, normalizer, normalizer, i1=i1, i2=i2, i3=i3, mu=mu, equation=self.equation, mesh=self.input_mesh))
        #####
        return ConcatDataset(train_datasets), ConcatDataset(test_datasets)



####
## Following is test codes 
###

class DatasetSimple():
    def __init__(self,):
        self.normalizer = None

    def get_onestep_dataloader(self,
                               train_test_split=0.2,
                               location='../Data/MP_data/',
                               batch_size=1,
                               dtype=torch.float,
                               dt=1,
                               ntrain=None,
                               ntest=None):
        with open(location + 'displacements0-5000.pkl', 'rb') as file:
            displacements = torch.tensor(pickle.load(file), dtype=dtype)
        with open(location + 'pressures0-5000.pkl', 'rb') as file:
            # scaler variable
            pressure = torch.tensor(pickle.load(file), dtype=dtype)[:, :, None]
        with open(location + 'velocitoes0-5000.pkl', 'rb') as file:
            velocities = torch.tensor(pickle.load(file), dtype=dtype)
        # keeping vx,xy, P, dx,dy
        varable_idices = [0, 1, 3, 4, 5]
        combined = torch.cat(
            [velocities, pressure, displacements], dim=-1)[:, :, varable_idices]
        step_t0 = combined[:-dt, ...]
        step_t1 = combined[dt:, ...]

        indexs = [i for i in range(step_t0.shape[0])]
        if not ntrain:
            ntrain = int(train_test_split * len(indexs))
        if not ntest:
            ntest = len(indexs) - ntrain

        random.shuffle(indexs)
        train_t0, test_t0 = step_t0[indexs[:ntrain]
                                    ], step_t0[indexs[ntrain:ntrain + ntest]]
        train_t1, test_t1 = step_t1[indexs[:ntrain]
                                    ], step_t1[indexs[ntrain:ntrain + ntest]]

        mean, var = torch.mean(
            train_t0, dim=(
                0, 1)), torch.mean(
            torch.var(
                train_t0, dim=(1)), dim=0)

        normalizer = Normalizer(mean, var**0.5)

        # setting normalizer
        self.normalizer = normalizer

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                train_t0, train_t1), batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                test_t0, test_t1), batch_size=batch_size, shuffle=False)

        return train_loader, test_loader


def get_dummy_dataloaders(train_test_split=0.2, channels=7, resolution=256,
                          location=None, batch_size=32, dtype=torch.float):
    train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.rand(
                1000,
                channels,
                resolution,
                resolution),
            torch.rand(
                1000,
                channels,
                resolution,
                resolution)),
        batch_size=batch_size,
        shuffle=True)

    test = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.rand(
                500,
                channels,
                resolution,
                resolution),
            torch.rand(
                500,
                channels,
                resolution,
                resolution)),
        batch_size=batch_size,
        shuffle=False)

    return train, test
