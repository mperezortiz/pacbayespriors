import torch
import numpy as np
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


def loaddataset(name):
    """Function to load the datasets (mnist and cifar10)

    Parameters
    ----------
    name : string
        name of the dataset ('mnist' or 'cifar10')

    """
    torch.manual_seed(7)

    if name == 'mnist':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(
            'mnist-data/', train=True, download=True, transform=transform)
        test = datasets.MNIST(
            'mnist-data/', train=False, download=True, transform=transform)
    elif name == 'cifar10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010)),
             ])
        train = datasets.CIFAR10(
            './data', train=True, download=True, transform=transform)
        test = datasets.CIFAR10(
            './data', train=False, download=True, transform=transform)
    else:
        raise RuntimeError(f'Wrong dataset chosen {name}')

    return train, test


def loadbatches(train, test, loader_kargs, batch_size, prior=False, perc_train=1.0, perc_prior=0.2, perc_val=0.0, self_cert=False):
    """Function to load the batches for the dataset

    Parameters
    ----------
    train : torch dataset object
        train split

    test : torch dataset object
        test split 

    loader_kargs : dictionary
        loader arguments

    batch_size : int
        size of the batch

    prior : bool
        boolean indicating the use of a learnt prior (e.g. this would be False for a random prior)

    perc_train : float
        percentage of data used for training (set to 1.0 if not intending to do data scarcity experiments)

    perc_prior : float
        percentage of data to use for building the prior (1-perc_prior is used to estimate the risk)

    """

    ntrain = len(train.data)
    ntest = len(test.data)

    if prior == False:
        indices = list(range(ntrain))
        split = int(np.round((perc_train)*ntrain))
        random_seed = 10
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        if perc_val > 0.0:
            # compute number of data points
            indices = list(range(split))
            split_val = int(np.round((perc_val)*split))
            train_idx, val_idx = indices[split_val:], indices[:split_val]
        else:
            train_idx = indices[:split]
            val_idx = None

        # we assume that if there is a validation set, that should not be included in posterior set
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        bound_loader_1batch = torch.utils.data.DataLoader(
            train, batch_size=len(train_idx), sampler=train_sampler, **loader_kargs)
        test_1batch = torch.utils.data.DataLoader(
            test, batch_size=ntest, shuffle=True, **loader_kargs)
        posterior_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=train_sampler, **loader_kargs)
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=True, **loader_kargs)
        prior_loader = None
        if perc_val > 0.0:
            val_loader = torch.utils.data.DataLoader(
                train, batch_size=batch_size, sampler=val_sampler, shuffle=False)
        else:
            val_loader = None
        bound_loader = posterior_loader
    else:
        if self_cert:
            n = len(train.data) + len(test.data)

            # reduce training data if needed
            new_num_train = int(np.round((perc_train)*n))
            indices = list(range(new_num_train))
            split = int(np.round((perc_prior)*new_num_train))
            random_seed = 10
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            all_train_sampler = SubsetRandomSampler(indices)
            if perc_val > 0.0:
                bound_idx = indices[split:]
                indices_prior = list(range(split))
                all_prior_sampler = SubsetRandomSampler(indices_prior)
                split_val = int(np.round((perc_val)*split))
                prior_idx, val_idx = indices_prior[split_val:], indices_prior[:split_val]
            else:
                bound_idx, prior_idx = indices[split:], indices[:split]
                val_idx = None

            # bound always needs to be evaluated on data that is separate from prior
            bound_sampler = SubsetRandomSampler(bound_idx)
            prior_sampler = SubsetRandomSampler(prior_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            final_dataset = torch.utils.data.ConcatDataset([train, test])

            bound_loader_1batch = torch.utils.data.DataLoader(
                final_dataset, batch_size=len(bound_idx), sampler=bound_sampler, **loader_kargs)
            bound_loader = torch.utils.data.DataLoader(
                final_dataset, batch_size=batch_size, sampler=bound_sampler, shuffle=False)
            # posterior can be trained with all data!
            posterior_loader = torch.utils.data.DataLoader(
                final_dataset, batch_size=batch_size, sampler=all_train_sampler, shuffle=False)
            prior_loader = torch.utils.data.DataLoader(
                final_dataset, batch_size=batch_size, sampler=prior_sampler, shuffle=False)

            if perc_val > 0.0:
                val_loader = torch.utils.data.DataLoader(
                    final_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False)
            else:
                val_loader = None
            test_loader = None
            test_1batch = None

        else:
            # reduce training data if needed
            new_num_train = int(np.round((perc_train)*ntrain))
            indices = list(range(new_num_train))
            split = int(np.round((perc_prior)*new_num_train))
            random_seed = 10
            np.random.seed(random_seed)
            np.random.shuffle(indices)

            all_train_sampler = SubsetRandomSampler(indices)
            #train_idx, valid_idx = indices[split:], indices[:split]
            if perc_val > 0.0:
                bound_idx = indices[split:]
                indices_prior = list(range(split))
                all_prior_sampler = SubsetRandomSampler(indices_prior)
                split_val = int(np.round((perc_val)*split))
                prior_idx, val_idx = indices_prior[split_val:], indices_prior[:split_val]
            else:
                bound_idx, prior_idx = indices[split:], indices[:split]
                val_idx = None

            # bound always needs to be evaluated on data that is separate from prior
            bound_sampler = SubsetRandomSampler(bound_idx)
            prior_sampler = SubsetRandomSampler(prior_idx)
            val_sampler = SubsetRandomSampler(val_idx)

            bound_loader_1batch = torch.utils.data.DataLoader(
                train, batch_size=len(bound_idx), sampler=bound_sampler, **loader_kargs)
            bound_loader = torch.utils.data.DataLoader(
                train, batch_size=batch_size, sampler=bound_sampler, shuffle=False)
            test_1batch = torch.utils.data.DataLoader(
                test, batch_size=ntest, shuffle=True, **loader_kargs)
            # posterior can be trained with all data!
            posterior_loader = torch.utils.data.DataLoader(
                train, batch_size=batch_size, sampler=all_train_sampler, shuffle=False)
            prior_loader = torch.utils.data.DataLoader(
                train, batch_size=batch_size, sampler=prior_sampler, shuffle=False)
            test_loader = torch.utils.data.DataLoader(
                test, batch_size=batch_size, shuffle=True, **loader_kargs)
            if perc_val > 0.0:
                val_loader = torch.utils.data.DataLoader(
                    train, batch_size=batch_size, sampler=val_sampler, shuffle=False)
            else:
                val_loader = None

    # train_loader comprises all the data used in training and prior_loader the data used to build
    # the prior
    # set_bound_1batch and set_bound are the set of data points used to evaluate the bound.
    # the only difference between these two is that onf of them is splitted in multiple batches
    # while the 1batch one is only one batch. This is for computational efficiency with some
    # of the large architectures used.
    # The same is done for test_1batch
    return posterior_loader, test_loader, prior_loader, bound_loader_1batch, test_1batch, bound_loader, val_loader


def loaddatafrompath(path_data, pickle=False):
    if pickle:
        data = pickle.load(open(path_data, "rb"))
        X = data['X']
        y = data['y']
    else:
        df = pd.read_csv(path_data)
        aux = df.to_numpy()
        dims = aux.shape
        data = {}
        X = aux[:, 0:dims[1]-2]
        y = aux[:, dims[1]-1]
    return X, y


def holdout_data(X, y, random_state=1, test_size=10000):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def standardize(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)
    return X_train, X_test


def load_batches(X, y, loader_kargs, batch_size=250):
    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.Tensor(y)
    data = TensorDataset(tensor_x, tensor_y)  # create your datset
    if batch_size == -1:
        loader = DataLoader(data, batch_size=X.shape[0], **loader_kargs)
    else:
        loader = DataLoader(data, batch_size=batch_size, **loader_kargs)
    return loader


def concatenate_sets(X_1, X_2, y_1, y_2):
    X_augm = np.concatenate((X_1, X_2), axis=0)
    y_augm = np.concatenate((y_1, y_2), axis=0)
    return X_augm, y_augm
