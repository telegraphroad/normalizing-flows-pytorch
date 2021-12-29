import os

import hydra
import numpy as np
import torch
import torchvision
import sklearn.datasets
from torchvision import transforms
from scipy.stats import gennorm,lognorm,t
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
N_DATASET_SIZE = 65536


def _sample_circles(n):
    samples, _ = sklearn.datasets.make_circles(N_DATASET_SIZE, noise=0.08, factor=0.5)
    return samples * 0.6


def _sample_moons(n):
    samples, _ = sklearn.datasets.make_moons(N_DATASET_SIZE, noise=0.08)
    samples = (samples - 0.5) / 2.0
    return samples

def _sample_ggd(n, beta, dim):
    return gennorm(beta=beta).rvs(size=[n,dim])

def _sample_lognorm(n, s, dim):
    return lognorm(s=s).rvs(size=[n,dim])

def _sample_t(n, df, dim):
    return t(df=df).rvs(size=[n,dim])

def _sample_normals(n):
    radius = 0.7
    n_normals = 8
    k = np.random.randint(n_normals, size=(n, ))
    cx = radius * np.cos(2.0 * np.pi * k / n_normals)
    cy = radius * np.sin(2.0 * np.pi * k / n_normals)
    dx, dy = np.random.normal(size=(2, n)) * 0.1
    x = cx + dx
    y = cy + dy
    samples = np.stack([x, y], axis=1)
    return samples


def _sample_swiss(n):
    samples, _ = sklearn.datasets.make_swiss_roll(n, noise=0.08)
    samples[:, 0] = samples[:, 0] * 0.07
    samples[:, 1] = samples[:, 1] * 0.07 - 1.0
    samples[:, 2] = samples[:, 2] * 0.07
    return samples


def _sample_s_curve(n):
    samples, _ = sklearn.datasets.make_s_curve(n, noise=0.08)
    samples[:, 0] = samples[:, 0] * 0.7
    samples[:, 1] = (samples[:, 1] - 1.0) * 0.7
    samples[:, 2] = samples[:, 2] * 0.35
    return samples


class FlowDataLoader(object):
    def __init__(self, name='moons', batch_size=1024, total_steps=100000, shuffle=True, **args):
        super(FlowDataLoader, self).__init__()
        self.name = name
        self.batch_size = batch_size
        self.total_steps = total_steps
        self.shuffle = shuffle

        self.iter = 0
        self.indices = None
        self._args = args

        self._initialize()

    def _initialize(self):
        data_root = os.path.join(hydra.utils.get_original_cwd(), 'data')
        if self.name == 'mnist':
            self.dset = torchvision.datasets.MNIST(root=os.path.join(data_root, 'mnist'),
                                                   train=True,
                                                   download=True,
                                                   transform=transforms.Pad((2, 2)))
            self.dims = (1, 32, 32)
            self.dtype = 'image'
        elif self.name == 'credit':
            ds = pd.read_csv('creditcard.csv')
            ds = shuffle(ds)
            num_cols = ds.columns[:-1]
            target_col = 'Class'
            X = ds.loc[:, num_cols]
            y = ds.loc[:, target_col]
            X_train_num = ds[num_cols]
            minmax = MinMaxScaler()
            X_train_num_trans = minmax.fit_transform(X_train_num)
            #X_train_num_trans = X_train_num
            X_train_num_trans = pd.DataFrame(data=X_train_num_trans)
            amnt_num_trans_cols = X_train_num_trans.shape[1]
            X_train_final = X_train_num_trans
            self.dset = X_train_final.values
            self.dims = (3, )
            self.dtype = '2d'
         
        elif self.name == 'cifar10':
            self.dset = torchvision.datasets.CIFAR10(root=os.path.join(data_root, 'cifar10'),
                                                     train=True,
                                                     download=True)
            self.dims = (3, 32, 32)
            self.dtype = 'image'
        elif self.name == 'circles':
            self.dset = _sample_circles(N_DATASET_SIZE)
            self.dims = (2, )
            self.dtype = '2d'
          
        elif self.name == 'ggd':
            self.dset = _sample_ggd(N_DATASET_SIZE,self._args['beta'],self._args['dim'])
            self.dims = (self._args['dim'], )
            self.dtype = str(self._args['dim']) + 'd'
        elif self.name == 't':
            self.dset = _sample_t(N_DATASET_SIZE,self._args['df'],self._args['dim'])
            self.dims = (self._args['dim'], )
            self.dtype = str(self._args['dim']) + 'd'
          
        elif self.name == 'lognorm':
            self.dset = _sample_lognorm(N_DATASET_SIZE,self._args['s'],self._args['dim'])
            self.dims = (self._args['dim'], )
            self.dtype = str(self._args['dim']) + 'd'

        elif self.name == 'moons':
            self.dset = _sample_moons(N_DATASET_SIZE)
            self.dims = (2, )
            self.dtype = '2d'
        elif self.name == 'normals':
            self.dset = _sample_normals(N_DATASET_SIZE)
            self.dims = (2, )
            self.dtype = '2d'
        elif self.name == 'swiss':
            self.dset = _sample_swiss(N_DATASET_SIZE)
            self.dims = (3, )
            self.dtype = '3d'
        elif self.name == 's_curve':
            self.dset = _sample_s_curve(N_DATASET_SIZE)
            self.dims = (3, )
            self.dtype = '3d'
        else:
            raise Exception('unsupported type: "%s"' % self.name)

        self.iter = 0
        self.indices = np.arange(len(self.dset))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.dset)

    def __iter__(self):
        for _ in range(self.total_steps):
            if self.__len__() <= self.iter + self.batch_size:
                self._initialize()

            idx = self.indices[self.iter:self.iter + self.batch_size]
            self.iter += self.batch_size

            if self.dtype == 'image':
                data = [np.asarray(self.dset[i][0], dtype='float32') / 255.0 for i in idx]
                data = np.reshape(data, (self.batch_size, self.dims[1], self.dims[2], -1))
                data = np.transpose(data, axes=(0, 3, 1, 2))
            else:
                data = self.dset[idx]
                data = data.astype('float32')

            yield torch.from_numpy(data)
    def sample(self, n):
        if self.name == 'circles':
            return _sample_circles(n)
        elif self.name == 'ggd':
            return _sample_ggd(n,self._args['beta'],self._args['dim'])
        elif self.name == 't':
            return _sample_t(n,self._args['df'],self._args['dim'])
        elif self.name == 'lognorm':
            return _sample_lognorm(n,self._args['s'],self._args['dim'])

        elif self.name == 'moons':
            return _sample_moons(n)
        elif self.name == 'normals':
            return _sample_normals(n)
        elif self.name == 'swiss':
            return _sample_swiss(n)
        elif self.name == 's_curve':
            return _sample_s_curve(n)
        else:
            raise Exception('unsupported type: "%s"' % self.name)
