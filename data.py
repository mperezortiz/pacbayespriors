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
from oversampling import Oversampling
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


# class ConcatDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets

#     def __getitem__(self, i):
#         return tuple(d[i] for d in self.datasets)

#     def __len__(self):
#         return min(len(d) for d in self.datasets)

# class ConcatDataset(torch.utils.data.Dataset):
#     def __init__(self, *datasets):
#         self.datasets = datasets

#     def __getitem__(self, i):
#         return tuple(d[i %len(d)] for d in self.datasets)

#     def __len__(self):
#         return max(len(d) for d in self.datasets)

# train_loader = torch.utils.data.DataLoader(
#              ConcatDataset(
#                  datasets.ImageFolder(traindir_A),
#                  datasets.ImageFolder(traindir_B)
#              ),
#              batch_size=args.batch_size, shuffle=True,
#              num_workers=args.workers, pin_memory=True)

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

            #import ipdb; ipdb.set_trace()

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



# def loadbatches_selfcert(train, test, loader_kargs, batch_size, prior=False, perc_train=1.0, perc_prior=0.2, perc_val=0.0):
#     """Function to load the batches for the dataset

#     Parameters
#     ----------
#     train : torch dataset object
#         train split
    
#     test : torch dataset object
#         test split 

#     loader_kargs : dictionary
#         loader arguments
    
#     batch_size : int
#         size of the batch

#     prior : bool
#         boolean indicating the use of a learnt prior (e.g. this would be False for a random prior)

#     perc_train : float
#         percentage of data used for training (set to 1.0 if not intending to do data scarcity experiments)

#     perc_prior : float
#         percentage of data to use for building the prior (1-perc_prior is used to estimate the risk)

#     """

#     n = len(train.data) + len(test.data)

#     # reduce training data if needed
#     new_num_train = int(np.round((perc_train)*n))
#     indices = list(range(new_num_train))
#     split = int(np.round((perc_prior)*new_num_train))
#     random_seed = 10
#     np.random.seed(random_seed)
#     np.random.shuffle(indices)

#     all_train_sampler = SubsetRandomSampler(indices)
#     if perc_val > 0.0:
#         bound_idx = indices[split:]
#         indices_prior = list(range(split))
#         all_prior_sampler = SubsetRandomSampler(indices_prior)
#         split_val = int(np.round((perc_val)*split))
#         prior_idx, val_idx = indices_prior[split_val:], indices_prior[:split_val]
#     else: 
#         bound_idx, prior_idx = indices[split:], indices[:split]
#         val_idx = None
        
#     # bound always needs to be evaluated on data that is separate from prior
#     bound_sampler = SubsetRandomSampler(bound_idx)
#     prior_sampler = SubsetRandomSampler(prior_idx)
#     val_sampler = SubsetRandomSampler(val_idx)

#     bound_loader_1batch = torch.utils.data.DataLoader(
#                  ConcatDataset(train, test), batch_size=len(bound_idx), sampler=bound_sampler, **loader_kargs)
#     bound_loader = torch.utils.data.DataLoader(
#                  ConcatDataset(train, test), batch_size=batch_size, sampler=bound_sampler, shuffle=False)
#     # posterior can be trained with all data!
#     posterior_loader = torch.utils.data.DataLoader(
#                  ConcatDataset(train, test), batch_size=batch_size, sampler=all_train_sampler, shuffle=False)
#     prior_loader = torch.utils.data.DataLoader(
#                  ConcatDataset(train, test), batch_size=batch_size, sampler=prior_sampler, shuffle=False)

#     if perc_val > 0.0:
#         val_loader = torch.utils.data.DataLoader(
#                     ConcatDataset(train, test), batch_size=batch_size, sampler=val_sampler, shuffle=False)
#     else: 
#         val_loader = None

#     # train_loader comprises all the data used in training and prior_loader the data used to build
#     # the prior
#     # set_bound_1batch and set_bound are the set of data points used to evaluate the bound.
#     # the only difference between these two is that onf of them is splitted in multiple batches
#     # while the 1batch one is only one batch. This is for computational efficiency with some
#     # of the large architectures used.
#     # The same is done for test_1batch
#     return posterior_loader, None, prior_loader, bound_loader_1batch, None, bound_loader, val_loader

def loadbatches_relay(train, test, loader_kargs, batch_size, perc_train=1.0, perc_relays=[0.2,0.8], perc_val=0.0):
    """Function to load the batches for the dataset using a bayes-relay technique:

        -first, shaves off extra data if perc_train != 1
        -next, sets aside validation data
        -with remaining data, splits according to perc_relays(must add up to 1!)
            -ex: perc_relays = [0.1, 0.1 ,0.8]
                -first relay's bound = first 10% of data
                -first relay's training data = first 10% of data
                -second relay's bound = second 10% of data
                -second relay's training data = first 20% of data
                -third relay's bound = last 80% of data
                -third relay's training data = all 100% of data

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
    
    perc_relays : list of float
        each entry represents the portion of data that its corresponding relay iteration gets access to(in addition to 
        all data available to its prior)
    """

    assert sum(perc_relays) ==1 , "Relay data split must add to 1!"

    ntrain = len(train.data)
    ntest = len(test.data)

    #prior will never be false in relay case so removed that logic

    #remove last (perc_train %) of indices (only for data scarcity experiments)
    new_num_train = int(np.round((perc_train)*ntrain))
    indices = list(range(new_num_train))

    #compute the allocation per relay of data. posterior gains all data from prior + an additional portion
    data_allocations = np.cumsum(perc_relays)

    #TODO: added a variable that determines whether posterior has access to prior's data when training. Might be
    #      interesting to experiment with
    posterior_access_prior_data = True

    random_seed = 10
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    if perc_val > 0.0:
        #shave off the back (perc_val * new_num_train) data samples for validation
        new_num_train = int(np.round((1-perc_val)*new_num_train))
        train_indices = indices[0:new_num_train]
        val_indices = indices[new_num_train:]
    else:
        val_indices = None

    #create val sampler + loader
    val_sampler = SubsetRandomSampler(val_indices)
    val_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=val_sampler, shuffle=False)


    #create dictionaries to hold relay training data loader and relay bound data loader. keys are integers corresponding
    # to the relay iteration
    relay_train_loader = {key: None for key in range(len(perc_relays))}
    relay_bound_loader = {key: None for key in range(len(perc_relays))}
    start_relay_data_index = 0

    for i in range(len(data_allocations)):
        #the last data index acessible to this relay
        end_relay_data_index = int(np.round((data_allocations[i])*new_num_train))
        
        #pull relay's indices - bound always needs to be evaluated on data that is separate from prior
        curr_bound_indices = train_indices[start_relay_data_index:end_relay_data_index]
        
        #see above note on 'variable that determines whether posterior has access to prior's data'
        if posterior_access_prior_data:
            curr_train_indices = train_indices[:end_relay_data_index]
        else:
            curr_train_indices = curr_bound_indices
        
        #set the next relay to start pulling data from the end of the previous one
        start_relay_data_index = end_relay_data_index

        #defines current train data sampler and loader
        curr_relay_sampler = SubsetRandomSampler(curr_train_indices)
        curr_relay_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=curr_relay_sampler, shuffle=False)
        
        #defines current bound data sampler and loader
        curr_bound_sampler = SubsetRandomSampler(curr_bound_indices)
        curr_bound_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=curr_bound_sampler, shuffle=False)
        
        relay_train_loader[i] = curr_relay_loader
        relay_bound_loader[i] = curr_bound_loader 
    
    
    bound_loader_1batch = torch.utils.data.DataLoader(
            train, batch_size=len(curr_bound_indices), sampler=curr_bound_sampler, **loader_kargs)
    test_1batch = torch.utils.data.DataLoader(
            test, batch_size=ntest, shuffle=True, **loader_kargs)
        # posterior can be trained with all data!
    test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=True, **loader_kargs)

    return relay_train_loader, test_loader, bound_loader_1batch, test_1batch, relay_bound_loader, val_loader
    

#TODO: remove this wrapper!

def loadbatches_relay_wrapper(train, test, loader_kargs, batch_size, prior=False, perc_train=1.0, perc_prior=0.2, perc_val=0.0):
    """ Temporary wrapper for the loadbatches_relay function, for backward compatibility.
        Once I confirm with Maria that lodabatches is a special case of loadbatches_relay, delete + rework 'utils' 
        calls to this fct
            -only case I'm unsure about is Prior=False
    """

    perc_relays = [perc_prior, 1-perc_prior]

    relay_train_loader, test_loader, bound_loader_1batch, test_1batch, relay_bound_loader, val_loader = loadbatches_relay(train, test, loader_kargs, batch_size, perc_train=perc_train, perc_relays=perc_relays, perc_val=perc_val)
    return relay_train_loader[1], test_loader, relay_train_loader[0], bound_loader_1batch, test_1batch, relay_bound_loader[1], val_loader



def loaddatafrompath(path_data, pickle=False):
    if pickle:
        data = pickle.load( open(path_data, "rb" ) )
        X = data['X']
        y = data['y']
    else: 
        df = pd.read_csv (path_data)
        aux = df.to_numpy()
        dims = aux.shape
        data = {}
        X = aux[:,0:dims[1]-2]
        y = aux[:,dims[1]-1]
    return X, y

def holdout_data(X, y, random_state = 1, test_size=10000):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)  
    return X_train, X_test, y_train, y_test

def standardize(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)
    return X_train, X_test

def oversample(X_train, y_train, k_oversampling=3, random_state=1, perc_oversampling=1, iid=False):
    oversample = Oversampling(k_neighbors=k_oversampling, random_state=random_state, perc_oversampling=perc_oversampling, iid=iid)
    X_synth, y_synth = oversample.fit_resample(X_train, y_train)
    X_augm, y_augm = concatenate_sets(X_train, X_synth, y_train, y_synth)
    return X_synth, y_synth, X_augm, y_augm

def load_batches(X, y, loader_kargs, batch_size=250):
    tensor_x = torch.Tensor(X) # transform to torch tensor
    tensor_y = torch.Tensor(y)
    data = TensorDataset(tensor_x,tensor_y) # create your datset
    if batch_size == -1:
        loader = DataLoader(data, batch_size=X.shape[0], **loader_kargs) 
    else: 
        loader = DataLoader(data, batch_size=batch_size, **loader_kargs) 
    return loader

def concatenate_sets(X_1, X_2, y_1, y_2):
    X_augm = np.concatenate((X_1,X_2), axis=0)
    y_augm = np.concatenate((y_1,y_2), axis=0)
    return X_augm, y_augm