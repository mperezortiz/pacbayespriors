import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from pbb.models import NNet4l, CNNet4l, ProbNNet4l, ProbCNNet4l, ProbCNNet9l, CNNet9l, NNet3l, ProbNNet3l, CNNet13l, ProbCNNet13l, ProbCNNet15l, CNNet15l, trainNNet, testNNet, Lambda_var, trainPNNet, computeRiskCertificates, testPosteriorMean, testStochastic, testEnsemble, trainNNetmixup
from pbb.bounds import PBBobj
from pbb import data
import wandb
import copy
import os

# TODOS: 
# combine prior_args and posterior_args to a single argument 'bayes_relay_args', for iterative pac-bayes
# expand hyperparameter tuning to raytune option
# Clean up stuff like "learning_rate_prior = prior_net_params['lr']" so that code is cleaner
# If prior is probabilistic we should be reporting and logging as well the risk of the prior and using computeRiskCertificates


def runexp(experiment_args, prior_args, posterior_args, wandb_args=None):

    """Run an experiment with PAC-Bayes inspired training objectives
    Parameters
    ----------
    experiment_args : dictionary
        contains arguments that define the structure of the overall experiment
    prior_args : dictionary
        contains arguments that define the structure of the prior network
    posterior_args : dictionary
        contains arguments that define the structure of the posterior network
    wandb_args : dictionary
        contains arguments that define the method of logging to Weights & 
        Biases
    """
    prior_net_params = prior_args

    learning_rate = posterior_args['lr']
    momentum = posterior_args['momentum']
    objective = posterior_args['objective']
    train_epochs = posterior_args['epochs']
    kl_penalty_posterior = posterior_args['kl_penalty']
    initial_lamb = posterior_args['initial_lamb']
    
    device=experiment_args['device']
    name_data = experiment_args['dataset']
    prior_type = experiment_args['prior_type']
    model = experiment_args['model_class']
    layers = experiment_args['layers']
    neurons = experiment_args['neurons']
    batch_size = experiment_args['batch_size']
    delta = experiment_args['delta']
    delta_test = experiment_args['delta_test']
    pmin = experiment_args['pmin']
    mc_samples = experiment_args['mc_samples']
    verbose = experiment_args['verbose']
    verbose_test = experiment_args['verbose_test']
    samples_ensemble = experiment_args['samples_ensemble']
    perc_train = experiment_args['perc_train']
    perc_val = experiment_args['perc_val']
    perc_prior = experiment_args['perc_prior']
    freq_test = experiment_args['freq_test']
    oversampling = experiment_args['oversampling']
    alpha_mixup = experiment_args['alpha_mixup']
    perc_oversampling = experiment_args['perc_oversampling']
    k_oversampling = experiment_args['k_oversampling']
    val_posterior = experiment_args['val_posterior']
    self_cert = experiment_args['self_cert']

    #log training status with weights & biases
    #check if W&B turned on and haven't initialized run(if it's already been initialized then this is a sweep experiment)
    if wandb_args['log_wandb'] and wandb_args['run_type']!='sweep':
        if wandb_args['scratch_dir'] is not None:
            run = wandb.init(settings=wandb.Settings(start_method="fork"), reinit=True, entity=wandb_args['entity'], dir = wandb_args['scratch_dir'], project=wandb_args['wandb_project_name'], config=locals())
        else:
            run = wandb.init(settings=wandb.Settings(start_method="fork"), reinit=True, entity=wandb_args['entity'], project=wandb_args['wandb_project_name'], config=locals())
        #if a specific name for the run is passed, set it. Otherwise use auto-generated run name
        if wandb_args.get('wandb_run_name') is not None:
            wandb.run.name = wandb_args['wandb_run_name']

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(7)
    np.random.seed(0)
    RANDOM_STATE = 10

    vision_dataset = False
    if name_data in {'mnist', 'cifar10'}:
        vision_dataset = True
    
    learn_prior = False
    if prior_type == 'learnt':
        learn_prior = True

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_kargs = {'num_workers': 1,
                    'pin_memory': True} if torch.cuda.is_available() else {}

    # set the hyperparameter of the prior 
    rho_prior = math.log(math.exp(prior_net_params['sigma'])-1.0)

    # if basically we just want ERM (usually to be used as a baseline)
    if prior_net_params["objective"] in ['erm', 'mixup'] and objective in ['erm', 'mixup']:
        if vision_dataset:
            train, test = data.loaddataset(name_data)
            
            prior_net_params["loader"], test_loader, _, _, _, _, val_loader = data.loadbatches(
                train, test, loader_kargs, batch_size, prior=False, perc_train=perc_train, perc_prior=0.0, perc_val=perc_val)
        
            train_size = len(prior_net_params["loader"].dataset)
            num_classes = len(prior_net_params["loader"].dataset.classes)
            n_features = None 
            
        else:
            X, y = data.loaddatafrompath(name_data)
            n_features = X.shape[1]

            # partition the data in train and test
            X_train, X_test, y_train, y_test = data.holdout_data(X, y, random_state = RANDOM_STATE, test_size=0.2)
            X_train, X_test = data.standardize(X_train, X_test)

            # if we are doing experiments in the data starvation regime
            if perc_train < 1.0:
                X_train, _, y_train, _ = data.holdout_data(X_train, y_train, random_state = RANDOM_STATE, test_size=1-perc_train)
        
            if perc_val > 0.0:
                X_train, X_val, y_train, y_val = data.holdout_data(X_train, y_train, random_state = RANDOM_STATE, test_size=perc_val)
                val_loader = data.load_batches(X_val, y_val, loader_kargs)
            else:
                val_loader = None
            # if not prior to be learnt
            if oversampling: 
                _, _, X_post, y_post = data.oversample(X_train, y_train, k_oversampling=k_oversampling, random_state=RANDOM_STATE,  perc_oversampling=perc_oversampling)
            else: 
                X_post = X_train
                y_post = y_train
            
            prior_net_params["loader"] = data.load_batches(X_post, y_post, loader_kargs)
            test_loader = data.load_batches(X_test, y_test, loader_kargs)

            train_size = X_post.shape[0]
            num_classes = np.unique(y_post).shape[0] 

        prior_net = initialise_prior(model, name_data, layers, prior_net_params['dropout_prob'], device, prior_net_params["objective"], rho_prior, prior_net_params['weight_distribution'], init_priors_prior = prior_net_params["initialize_prior"], features=n_features, classes=num_classes, neurons=neurons)


        experiment_settings = {
                'num_classes' : num_classes,
                'delta' : delta,
                'pmin' : pmin,
                'delta_test' : delta_test,
                'mc_samples' : mc_samples,
                'device' : device,
                'wandb_args' : wandb_args,
                'test_loader' : test_loader,
                'verbose' : verbose,
                'verbose_test' : verbose_test,
                'samples_ensemble' : samples_ensemble,
                'name_data' : name_data,
                'perc_train' : perc_train,
                'freq_test' : freq_test,
                'alpha_mixup': alpha_mixup, 
                'val_posterior': val_posterior, 
                'self_cert': self_cert
        }

        prior_net = train_prior_net(prior_net, prior_net_params, experiment_settings, val_loader=val_loader)
        loss_prior_net, error_prior_net = testNNet(prior_net, test_loader, device=device)
        print(f"Prior test error: {error_prior_net :.5f}")
        # TODO: NEED TO ADD WANDB LOGGING HERE
    else:

        if vision_dataset:
            train, test = data.loaddataset(name_data)
            # PRIOR NET IS ALREADY INITIALISED BELOW!
            #prior_net = initialise_prior(model, name_data, layers, dropout_prob, device, objective_prior, rho_prior, prior_dist, init_priors_prior = init_priors_prior)
            posterior_loader, test_loader, prior_net_params["loader"], bound_loader_1batch, _, bound_loader, val_loader = data.loadbatches(
                train, test, loader_kargs, batch_size, prior=learn_prior, perc_train=perc_train, perc_prior=perc_prior, perc_val=perc_val, self_cert=self_cert)
            #import ipdb; ipdb.set_trace()

            train_size = len(posterior_loader)*batch_size 
            num_classes = 10#len(prior_net_params["loader"].dataset.classes)
            n_features = None
        else:
            X, y = data.loaddatafrompath(name_data)
            n_features = X.shape[1]

            # partition the data in train and test
            X_train, X_test, y_train, y_test = data.holdout_data(X, y, random_state = RANDOM_STATE, test_size=0.2)
            X_train, X_test = data.standardize(X_train, X_test)

            # if we are doing experiments in the data starvation regime
            if perc_train < 1.0:
                X_train, _, y_train, _ = data.holdout_data(X_train, y_train, random_state = RANDOM_STATE, test_size=1-perc_train)
        
            # if we want to learn the prior 
            if prior_type == 'learnt':
                # we divide data for the prior and the bound
                X_eval, X_prior, y_eval, y_prior = data.holdout_data(X_train, y_train, random_state = RANDOM_STATE, test_size=perc_prior)

                if perc_val>0.0:
                    # we divide data for the prior again to set aside some validation data
                    X_prior, X_val, y_prior, y_val = data.holdout_data(X_prior, y_prior, random_state = RANDOM_STATE, test_size=perc_val)
                    val_loader = data.load_batches(X_val, y_val, loader_kargs)
                else:
                    val_loader = None

                if oversampling:
                    _, _, X_prior, y_prior = data.oversample(X_prior, y_prior, k_oversampling=k_oversampling, random_state=RANDOM_STATE, perc_oversampling=perc_oversampling)
                    # the posterior is learnt with all the data + synthetic data from all the data
                    _, _, X_post, y_post = data.oversample(X_train, y_train, k_oversampling=k_oversampling, random_state=RANDOM_STATE, perc_oversampling=perc_oversampling)
                    # ONLY from part of the real data
                else:    
                    X_post = X_train
                    y_post = y_train

                prior_net_params["loader"] = data.load_batches(X_prior, y_prior, loader_kargs)
                
            # if not prior to be learnt
            else:
                if oversampling: 
                    _, _, X_post, y_post = data.oversample(X_train, y_train, k_oversampling=k_oversampling, random_state=RANDOM_STATE,  perc_oversampling=perc_oversampling)
                else: 
                    X_post = X_train
                    y_post = y_train
                    X_eval = X_train
                    y_eval = y_train
        
            posterior_loader = data.load_batches(X_post, y_post, loader_kargs)
            bound_loader = data.load_batches(X_eval, y_eval, loader_kargs)
            test_loader = data.load_batches(X_test, y_test, loader_kargs)
            
            bound_loader_1batch = data.load_batches(X_eval, y_eval, loader_kargs, batch_size=-1)
            train_size = X_post.shape[0]
            num_classes = np.unique(y_post).shape[0]

        if prior_type == 'learnt':
            prior_net_params['n_prior'] = len(prior_net_params["loader"])*batch_size 
        else: 
            prior_net_params['n_prior'] = 0

        prior_net_params['bound_loader'] = bound_loader   
        prior_net_params['bound_loader_1batch'] = bound_loader_1batch  
        prior_net_params['n_bound'] = len(bound_loader)*batch_size 

    
        prior_net = initialise_prior(model, name_data, layers, prior_net_params['dropout_prob'], device, prior_net_params["objective"], rho_prior, prior_net_params['weight_distribution'], init_priors_prior = prior_net_params["initialize_prior"], features=n_features, classes=num_classes, neurons=neurons)

        if prior_net_params['objective'] == 'flamb':
            prior_net_params['lambda_var'] = Lambda_var(initial_lamb, train_size).to(device)
            prior_net_params['optimizer_lambda'] = optim.SGD(lambda_var.parameters(), lr=learning_rate, momentum=momentum)
        else:
            prior_net_params['optimizer_lambda'] = None
            prior_net_params['lambda_var'] = None

        prior_net_params['toolarge'] = False
        if model == 'cnn':
            prior_net_params['toolarge'] = True
        
        prior_net_params['bound'] = PBBobj(prior_net_params['objective'], pmin, num_classes, delta,
                        delta_test, mc_samples, prior_net_params['kl_penalty'], device, n_posterior=prior_net_params['n_prior'], n_bound=prior_net_params['n_bound'])

        experiment_settings = {
                'num_classes' : num_classes,
                'delta' : delta,
                'pmin' : pmin,
                'delta_test' : delta_test,
                'mc_samples' : mc_samples,
                'device' : device,
                'wandb_args' : wandb_args,
                'test_loader' : test_loader,
                'verbose' : verbose,
                'verbose_test' : verbose_test,
                'samples_ensemble' : samples_ensemble,
                'name_data' : name_data,
                'perc_train' : perc_train,
                'freq_test' : freq_test,
                'alpha_mixup': alpha_mixup, 
                'val_posterior': val_posterior
        }

        if prior_type == 'learnt':   
            prior_net = train_prior_net(prior_net, prior_net_params, experiment_settings, val_loader=val_loader)
            if prior_net_params['objective'] == 'bbb' or prior_net_params['objective'] == 'fquad':
                loss_prior_net = -99
                error_prior_net = -99
                error_prior_net_std = -99
            else:
                loss_prior_net, error_prior_net = testNNet(prior_net, test_loader, device=device)
        else: 
            loss_prior_net, error_prior_net = testNNet(prior_net, test_loader, device=device)
            #print(f"Prior test error: {error_prior_net :.5f}")


        if prior_net_params["objective"] in ['bbb', 'fquad', 'fclassic']:
            to_log_prior = validate_posterior_net(prior_net, prior_net_params, experiment_settings)
        
        posterior_bound = PBBobj(objective, pmin, num_classes, delta,
                        delta_test, mc_samples, kl_penalty_posterior, device, n_posterior=train_size, n_bound=prior_net_params['n_bound'])

        posterior_net, toolarge = initialise_posterior(model, name_data, layers, rho_prior, prior_net_params['weight_distribution'], prior_net, device, features=n_features, classes=num_classes, neurons=neurons)
        _, stch_err_mean_prior, stch_err_std_prior = testStochastic(posterior_net, test_loader, posterior_bound, device=device)


        if objective == 'flamb':
            lambda_var = Lambda_var(initial_lamb, train_size).to(device)
            optimizer_lambda = optim.SGD(lambda_var.parameters(), lr=learning_rate, momentum=momentum)
        else:
            optimizer_lambda = None
            lambda_var = None

        
        posterior_net_params = {
            'lr' : learning_rate,
            'momentum' : momentum,
            'objective' : objective,
            'epochs' : train_epochs,
            'loader' : posterior_loader,
            'bound' : posterior_bound,
            'bound_loader' : bound_loader,
            'bound_loader_1batch' : bound_loader_1batch,
            'toolarge' : toolarge,
            'lambda_var' : lambda_var,
            'optimizer_lambda' : optimizer_lambda,
            'kl_penalty' : kl_penalty_posterior
        }
        posterior_net = train_posterior_net(posterior_net, posterior_net_params, experiment_settings)

        #experiment_settings['mc_samples'] = 10000
        to_log = validate_posterior_net(posterior_net, posterior_net_params, experiment_settings)
        print(f"{to_log['Risk_01'] :.5f}, {to_log['Stch 01 error mean']:.5f}, {to_log['Stch 01 error std']:.5f}, {to_log['KL']:.12f}, {to_log['Post mean 01 error']:.5f}, {to_log['Train 01 error']:.5f}, {to_log['Ens 01 error']:.5f}, {error_prior_net:.5f}, {stch_err_mean_prior:.5f}, {stch_err_std_prior:.5f}")
        
        #print(to_log)
        # to_log.update( {'Epoch' : train_epochs} )
        # to_log.update( {'error_prior_net' : error_prior_net} )
        # to_log.update( {'loss_prior_net' : loss_prior_net} )
        # if prior_net_params["objective"] in ['bbb', 'fquad', 'fclassic']:
        #     to_log.update( {'risk_01_prior' : to_log_prior['Risk_01']} )
        #     to_log.update( {'stch_01_error_prior' : to_log_prior['Stch 01 error']} )
        #     to_log.update( {'postmean_01_error_prior' : to_log_prior['Post mean 01 error']} )
        # TO DO: we should log the results from training the prior (if it's a normal net just error_net_0
        # but if it's a PNN we should log stochastic errors, bound, etc)

        # if wandb_args['log_wandb']:
        #     to_log = {"Final/"+k: v for k, v in to_log.items()}
        #     wandb.log(to_log)


def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialise_prior(model, name_data, layers, dropout_prob, device, objective_prior, rho_prior, prior_dist, init_priors_prior='zeros', features=28*28, classes=10, neurons=100):

    prob_prior = True
    if objective_prior in {'erm', 'mixup'}: 
        prob_prior = False

    if model == 'cnn':
        if name_data == 'cifar10':
            # only cnn models are tested for cifar10, fcns are only used 
            # with mnist
            if not prob_prior: 
                if layers == 9:
                    prior_net = CNNet9l(dropout_prob=dropout_prob).to(device)
                elif layers == 13:
                    prior_net = CNNet13l(dropout_prob=dropout_prob).to(device)
                elif layers == 15:
                    prior_net = CNNet15l(dropout_prob=dropout_prob).to(device)
                else: 
                    raise RuntimeError(f'Wrong number of layers {layers}')
            else:
                if layers == 9:
                    prior_net = ProbCNNet9l(rho_prior, prior_dist=prior_dist,
                        device=device, init_prior = init_priors_prior).to(device)
                elif layers == 13:
                    prior_net = ProbCNNet13l(rho_prior, prior_dist=prior_dist,
                        device=device, init_prior = init_priors_prior).to(device)
                elif layers == 15:
                    prior_net = ProbCNNet15l(rho_prior, prior_dist=prior_dist,
                        device=device, init_prior = init_priors_prior).to(device)
                else: 
                    raise RuntimeError(f'Wrong number of layers {layers}')
        elif name_data == 'mnist':
            if not prob_prior:
                prior_net = CNNet4l(dropout_prob=dropout_prob).to(device)
            else:
                prior_net = ProbCNNet4l(rho_prior, prior_dist=prior_dist,
                        device=device, init_prior = init_priors_prior).to(device)
        else: 
            raise RuntimeError(f'Wrong dataset')
    else:
        if name_data == 'mnist':
            if prob_prior:
                if layers == 4:
                    prior_net = ProbNNet4l(rho_prior, prior_dist=prior_dist,
                                device=device, init_prior = init_priors_prior).to(device)
                else:
                    prior_net = ProbNNet3l(rho_prior, prior_dist=prior_dist,
                                device=device, init_prior = init_priors_prior).to(device)
            else:
                if layers == 4:
                    prior_net = NNet4l(dropout_prob=dropout_prob, device=device).to(device)
                else: 
                    prior_net = NNet3l(dropout_prob=dropout_prob, device=device, features=784, classes=10, neurons=neurons).to(device)
        else: 
            if prob_prior:
                if layers == 4:
                    prior_net = ProbNNet4l(rho_prior, prior_dist='gaussian',
                                    device=device, features=features, classes=classes, neurons=neurons, init_prior='random').to(device)
                else: 
                    prior_net = ProbNNet3l(rho_prior, prior_dist='gaussian',
                                    device=device, features=features, classes=classes, neurons=neurons, init_prior='random').to(device)
            else:
                if layers == 4:
                    prior_net = NNet4l(dropout_prob=dropout_prob, device=device, features=features, classes=classes, neurons=neurons).to(device)
                else:
                    prior_net = NNet3l(dropout_prob=dropout_prob, device=device, features=features, classes=classes, neurons=neurons).to(device)
                
            
    return prior_net

def initialise_posterior(model, name_data, layers, rho_prior, prior_dist, prior_net, device, features=28*28, classes=10, neurons=100):
    toolarge = False

    if model == 'cnn':
        toolarge = True
        if name_data == 'cifar10':
            if layers == 9:
                posterior_net = ProbCNNet9l(rho_prior, prior_dist=prior_dist,
                                    device=device, init_net=prior_net).to(device)
            elif layers == 13:
                posterior_net = ProbCNNet13l(rho_prior, prior_dist=prior_dist,
                                   device=device, init_net=prior_net).to(device)
            elif layers == 15: 
                posterior_net = ProbCNNet15l(rho_prior, prior_dist=prior_dist,
                                   device=device, init_net=prior_net).to(device)
            else: 
                raise RuntimeError(f'Wrong number of layers {layers}')
        else:
            posterior_net = ProbCNNet4l(rho_prior, prior_dist=prior_dist,
                          device=device, init_net=prior_net).to(device)
    elif model == 'fcn':
        if name_data == 'cifar10':
            raise RuntimeError(f'Cifar10 not supported with given architecture {model}')
        elif name_data == 'mnist':
            if layers == 4:
                posterior_net = ProbNNet4l(rho_prior, prior_dist=prior_dist,
                            device=device, init_net=prior_net, features=784, classes=10, neurons=neurons).to(device)
            else: 
                posterior_net = ProbNNet3l(rho_prior, prior_dist=prior_dist,
                                    device=device, features=784, classes=10, neurons=neurons, init_net=prior_net).to(device)
        else: 
            if layers == 4:
                posterior_net = ProbNNet4l(rho_prior, prior_dist='gaussian',
                                    device=device, features=features, classes=classes, neurons=neurons, init_net=prior_net).to(device)
            else:
                posterior_net = ProbNNet3l(rho_prior, prior_dist='gaussian',
                                    device=device, features=features, classes=classes, neurons=neurons, init_net=prior_net).to(device)
    else:
        raise RuntimeError(f'Architecture {model} not supported')

    return posterior_net, toolarge

def count_parameters(model): return sum(p.numel()
                                        for p in model.parameters() if p.requires_grad)

def train_prior_net(prior_net, prior_net_params, experiment_settings, val_loader=None):
    #TODO: is there a cleaner/more concise way to do this rather than 20 lines of pulling from dictionary into local vars?
    learning_rate_prior = prior_net_params['lr']
    momentum_prior = prior_net_params['momentum']
    objective_prior = prior_net_params['objective']
    epochs = prior_net_params['epochs']
    loader = prior_net_params['loader']
    kl_penalty = prior_net_params['kl_penalty']

    num_classes = experiment_settings['num_classes']
    delta = experiment_settings['delta']
    pmin = experiment_settings['pmin']
    delta_test = experiment_settings['delta_test']
    mc_samples = experiment_settings['mc_samples']
    device = experiment_settings['device']
    wandb_args = experiment_settings['wandb_args']
    test_loader = experiment_settings['test_loader']
    verbose = experiment_settings['verbose']
    alpha_mixup = experiment_settings['alpha_mixup']

    prior_optimizer = optim.SGD(
        prior_net.parameters(), lr=learning_rate_prior, momentum=momentum_prior)
    #end copied from below
        
    # we select best prior based on validation loss
    best_prior = None
    loss_best_prior = float('inf')
    
    if objective_prior == 'bbb' or objective_prior == 'fquad':
        bound_prior = PBBobj(objective_prior, pmin, num_classes, delta,
                delta_test, mc_samples, kl_penalty, device, n_posterior=prior_net_params['n_prior'], n_bound=prior_net_params['n_bound'])
        optimizer_lambda = None
        lambda_var = None

        for epoch in trange(epochs):
            trainPNNet(prior_net, prior_optimizer, bound_prior, epoch, loader, lambda_var, optimizer_lambda, True, log_wandb = wandb_args['log_wandb'], wandb_is_prior=True)
            if val_loader:
                val_loss_prior_net, _, _ = testStochastic(prior_net, val_loader, bound_prior, device=device, samples=1)
                #log the validation loss and the bound loss
                #TODO: remove this temporary code
                if wandb_args['log_wandb']:
                    test_logging = {
                        "Prior/Epoch" : epoch,
                        "Prior/val_loss" : val_loss_prior_net,
                    }
                    wandb.log(test_logging)
                if val_loss_prior_net < loss_best_prior:
                    
                    best_prior = copy.deepcopy(prior_net)
                    loss_best_prior = val_loss_prior_net
    elif objective_prior == 'erm':

        for epoch in trange(epochs):
            trainNNet(prior_net, prior_optimizer, epoch, loader,
                    device=device, verbose=verbose, log_wandb = wandb_args['log_wandb'], wandb_is_prior=True)
            if val_loader:
                val_loss_prior_net, _ = testNNet(prior_net, val_loader, device=device)
                #log the validation loss and the bound loss
                #TODO: remove this temporary code
                if wandb_args['log_wandb']:
                    test_logging = {
                        "Prior/Epoch" : epoch,
                        "Prior/val_loss" : val_loss_prior_net,
                    }
                    wandb.log(test_logging)
                if val_loss_prior_net < loss_best_prior:
                    best_prior = copy.deepcopy(prior_net)
                    loss_best_prior = val_loss_prior_net
    elif objective_prior == 'mixup':

        for epoch in trange(epochs):
            trainNNetmixup(prior_net, prior_optimizer, epoch, loader,
                    device=device, verbose=verbose, alpha=alpha_mixup)
            if val_loader:
                val_loss_prior_net, _ = testNNet(prior_net, val_loader, device=device)
                if val_loss_prior_net < loss_best_prior:
                    best_prior = copy.deepcopy(prior_net)
                    loss_best_prior = val_loss_prior_net
                
    # if no stopping, we return the last prior, not the best one!
    if not val_loader:
        best_prior = copy.deepcopy(prior_net)

    return best_prior


def train_posterior_net(posterior_net, posterior_net_params, experiment_settings):
    #TODO: is there a cleaner/more concise way to do this rather than 30 lines of pulling from dictionary into local vars?
    learning_rate_posterior = posterior_net_params['lr']
    momentum_posterior = posterior_net_params['momentum']
    objective_posterior = posterior_net_params['objective']
    epochs = posterior_net_params['epochs']
    loader = posterior_net_params['loader']
    bound = posterior_net_params['bound']
    bound_loader = posterior_net_params['bound_loader']
    bound_loader_1batch = posterior_net_params['bound_loader_1batch']
    toolarge = posterior_net_params['toolarge']
    kl_penalty = posterior_net_params['kl_penalty']

    lambda_var = posterior_net_params['lambda_var']
    optimizer_lambda = posterior_net_params['optimizer_lambda']

    num_classes = experiment_settings['num_classes']
    delta = experiment_settings['delta']
    pmin = experiment_settings['pmin']
    delta_test = experiment_settings['delta_test']
    mc_samples = experiment_settings['mc_samples']
    
    device = experiment_settings['device']
    wandb_args = experiment_settings['wandb_args']
    test_loader = experiment_settings['test_loader']
    verbose = experiment_settings['verbose']
    verbose_test = experiment_settings['verbose_test']
    samples_ensemble = experiment_settings['samples_ensemble']
    name_data = experiment_settings['name_data']
    perc_train = experiment_settings['perc_train']
    wandb_args = experiment_settings['wandb_args']
    freq_test = experiment_settings['freq_test']
    val_posterior = experiment_settings['val_posterior']
    
    posterior_optimizer = optim.SGD(posterior_net.parameters(), lr=learning_rate_posterior, momentum=momentum_posterior)

    best_posterior = None
    risk_best_posterior = float('inf')
    stats_best_posterior = {}
    epoch_best_posterior = 1

    if wandb_args['log_wandb']:
        wandb.watch(posterior_net, log_freq=100)
    for epoch in trange(epochs):
        trainPNNet(posterior_net, posterior_optimizer, bound, epoch, loader, lambda_var, optimizer_lambda, verbose, wandb_args['log_wandb'])
        if verbose_test and ((epoch+1) % freq_test == 0):
            #import ipdb; ipdb.set_trace()
            to_log = validate_posterior_net(posterior_net, posterior_net_params, experiment_settings, compute_all_metrics=False)
            to_log.update( {'Epoch' : epoch} )
            print(
            f"-Epoch {epoch :.5f}, risk: {to_log['Risk_01'] :.5f}, test error:  {to_log['Stch 01 error mean']:.5f}")

            if to_log['Risk_01'] < risk_best_posterior:
                best_posterior = copy.deepcopy(posterior_net)
                risk_best_posterior = to_log['Risk_01']
                stats_best_posterior = to_log
                epoch_best_posterior = epoch
                #import ipdb; ipdb.set_trace()

            if wandb_args['log_wandb']:
                #log training metrics. Experiment hyperparams(like momentum) are logged to the config.yaml file in W&B
                to_log = {"Posterior Checkpoint/"+k: v for k, v in to_log.items()}
                wandb.log(to_log)
    if wandb_args['save_model']:
        torch.save(best_posterior,os.path.join(wandb.run.dir, 'best_posterior_model.pth' ))
    
    #print('Best posterior...')
    #print(stats_best_posterior)
    if val_posterior:
        #print(f"{stats_best_posterior['Risk_01'] :.5f}, {stats_best_posterior['Stch 01 error']:.5f}, {stats_best_posterior['KL']:.5f}, {stats_best_posterior['Post mean 01 error']:.5f}, {stats_best_posterior['Train 01 error']:.5f}, {stats_best_posterior['Ens 01 error']:.5f}")
        return best_posterior
    else:
        return posterior_net

def validate_posterior_net(posterior_net, posterior_net_params, experiment_settings, compute_all_metrics=True): 
    learning_rate_posterior = posterior_net_params['lr']
    momentum_posterior = posterior_net_params['momentum']
    objective_posterior = posterior_net_params['objective']
    epochs = posterior_net_params['epochs']
    loader = posterior_net_params['loader']
    bound = posterior_net_params['bound']
    bound_loader = posterior_net_params['bound_loader']
    bound_loader_1batch = posterior_net_params['bound_loader_1batch']
    toolarge = posterior_net_params['toolarge']
    kl_penalty = posterior_net_params['kl_penalty']

    lambda_var = posterior_net_params['lambda_var']
    optimizer_lambda = posterior_net_params['optimizer_lambda']

    num_classes = experiment_settings['num_classes']
    delta = experiment_settings['delta']
    pmin = experiment_settings['pmin']
    delta_test = experiment_settings['delta_test']
    mc_samples = experiment_settings['mc_samples']
    device = experiment_settings['device']
    wandb_args = experiment_settings['wandb_args']
    test_loader = experiment_settings['test_loader']
    verbose = experiment_settings['verbose']
    verbose_test = experiment_settings['verbose_test']
    samples_ensemble = experiment_settings['samples_ensemble']
    name_data = experiment_settings['name_data']
    perc_train = experiment_settings['perc_train']
    wandb_args = experiment_settings['wandb_args']

    train_obj, risk_ce, risk_01, kl, loss_ce_train, loss_01_train = computeRiskCertificates(posterior_net, toolarge, bound, device=device,
    lambda_var=lambda_var, train_loader=bound_loader, whole_train=bound_loader_1batch)

    #import ipdb; ipdb.set_trace()
    if compute_all_metrics:
        stch_loss, stch_err_mean, stch_err_std = testStochastic(posterior_net, test_loader, bound, device=device, samples=10)
        post_loss, post_err = testPosteriorMean(posterior_net, test_loader, bound, device=device)
        ens_loss, ens_err = testEnsemble(posterior_net, test_loader, bound, device=device, samples=samples_ensemble)
    else: 
        stch_loss, stch_err_mean, stch_err_std = -99, -99, -99
        post_loss, post_err = -99, -99
        ens_loss, ens_err = -99, -99


    to_log = {
                    "Obj_train" : train_obj , 
                    "Risk_CE" : risk_ce , 
                    "Risk_01" : risk_01 , 
                    "KL" : kl , 
                    "Train NLL loss" : loss_ce_train , 
                    "Train 01 error" : loss_01_train , 
                    "Stch loss" : stch_loss , 
                    "Stch 01 error mean" : stch_err_mean , 
                    "Stch 01 error std" : stch_err_std , 
                    "Post mean loss" : post_loss , 
                    "Post mean 01 error" : post_err , 
                    "Ens loss" : ens_loss , 
                    "Ens 01 error" : ens_err }
    #print(to_log)

    return to_log