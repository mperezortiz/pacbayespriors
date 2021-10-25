import torch
import time
from pbb.utils import runexp
import os
scratchdir = os.environ.get("scratchdir")


prior_args = dict(
    lr = 0.001,
    momentum = 0.95,
    objective = 'erm', #or 'fquad' or 'bbb'
    weight_distribution = 'gaussian',
    epochs = 100,
    kl_penalty = 0.001,
    sigma = 0.01, 
    initialize_prior = 'random', #or 'zeros'
    dropout_prob = 0.2,
    stopping_prior = True
)


posterior_args = dict(
    lr = 0.001,
    momentum = 0.90,
    objective = 'fquad', # or 'bbb' or 'flamb' or 'fclassic'
    epochs = 100,
    kl_penalty = 1,
    initial_lamb = None #6.0
)


experiment_args = dict(
    dataset = 'mnist',
    prior_type = 'learnt',
    model_class = 'fcn', # cnn
    layers = 3,
    neurons = 100,
    batch_size = 250,

    delta = 0.025,
    delta_test = 0.01,
    pmin = 1e-5,
    mc_samples = 1000,
    
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    device = "cpu",
    verbose = True,
    verbose_test = True,
    
    samples_ensemble = 100,
    perc_train = 1.0,
    perc_val = 0.01,
    perc_prior = 0.5,
    freq_test = 3,
    
    oversampling = False,
    alpha_mixup = 1.0,
    perc_oversampling = 1,
    k_oversampling = 3,
    val_posterior = True,
    self_cert = True,
)

"""
Prior Parameters
    ----------
    lr: float
        learning rate used in the optimiser for learning the prior (only
        applicable if prior is learnt)
    momentum : float
        momentum used in the optimiser for learning the prior (only
        applicable if prior is learnt)
    objective : string
        training objective to use (only applicable if prior is learnt)
    epochs : int
        numer of epochs for training
    kl_penalty : float
        penalty for the kl coefficient in the training objective for the 
        prior - scales the kl coefficient.
    sigma : float
        scale hyperparameter for the prior - scales its standard deviation
        std of prior = log(exp(sigma_prior)-1)
    initialize_prior : string
        specifies whether to initialize prior's prior to zeros or truncated
        gaussian
    dropout_prob : float
        probability of an element to be zeroed. only used in training of the prior

-----------------------------------------------------------------------------

Posterior Parameters
    ----------
    lr: float
        learning rate used in the optimiser for learning the posterior
    momentum : float
        momentum used in the optimiser for learning the posterior 
    objective : string
        training objective to use
    epochs : int
        numer of epochs for training
    kl_penalty : float
        penalty for the kl coefficient in the training objective for the 
        posterior - scales the kl coefficient.
    initial_lamb : float
        initial value for the lambda variable used in flamb objective
        (scaled later)

-----------------------------------------------------------------------------

Experiment Parameters
    ----------
    dataset : string
        name of the dataset to use - either 'mnist', 'cifar10', or path to 
        dataset (check data file for more info)
    prior_type : string
        could be rand or learnt depending on whether the prior 
        is data-free or data-dependent
    model_class : string
        could be cnn or fcn
    layers : int
        integer indicating the number of layers (applicable for CIFAR-10, 
        to choose between 9, 13 and 15)
    neurons : int
        integer indicating the number of neurons per layer ( only applicable
        for fcn on mnist currently)
    batch_size : int
        batch size for experiments
    delta : float
        confidence value for the risk certificate training objective
    delta_test : float
        confidence value for the risk certificate chernoff bound (used when 
        computing the risk)
    pmin : float
        minimum probability to clamp the output of the cross entropy loss
    mc_samples : int
        number of monte carlo samples for estimating the risk certificate
        (set to 1000 by default as it is more computationally efficient, 
        although larger values lead to tighter risk certificates)
    device : string
        parameter for pytorch. cnn architectures with higher # of layers tend
        to run faster on GPU(device='cuda'), otherwise use 'cpu'
    verbose : bool
        whether to print metrics during training
    verbose_test : bool
        whether to print test and risk certificate stats during training epochs
    samples_ensemble : int
        number of members for the ensemble predictor
    perc_train : float
        percentage of train data to use for the entire experiment (can be used to run
        experiments with reduced datasets to test small data scenarios)
    perc_val : float
        percentage of data to be used to determine early stopping(saving) of
        the model. TODO: Needs to be deprecated
    perc_prior : float
        percentage of data to be used to learn the prior
    freq_test : int
        number of epochs to go between each posterior checkpoint
#TODO: add documentation for following parameters:
    alpha_mixup = 1.0,
    perc_oversampling = 1,
    self_cert
"""

runexp(experiment_args, prior_args, posterior_args)
