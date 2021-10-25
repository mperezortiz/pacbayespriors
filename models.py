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
import wandb
from torch.autograd import Variable
import copy

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used works best if :math:`\text{mean}` is
    near the center of the interval.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
    
    
   
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Fill tensor with uniform values from [l, u]
        tensor.uniform_(l, u)

        # Use inverse cdf transform from normal distribution
        tensor.mul_(2)
        tensor.sub_(1)

        # Ensure that the values are strictly between -1 and 1 for erfinv
        eps = torch.finfo(tensor.dtype).eps
        tensor.clamp_(min=-(1. - eps), max=(1. - eps))
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp one last time to ensure it's still in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

class Gaussian(nn.Module):
    """Implementation of a Gaussian random variable, using softplus for
    the standard deviation and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Gaussian.

    rho : Tensor of floats
        Scale parameter of the Gaussian (to be transformed to std
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the Gaussian is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def sigma(self):
        # Computation of standard deviation:
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. Specifically, we use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))
        

    def sample(self):
        # Return a sample from the Gaussian distribution
        epsilon = torch.randn(self.sigma.size()).to(self.device)
        return self.mu + self.sigma * epsilon

    def compute_kl(self, other):
        # Compute KL divergence between two Gaussians (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = torch.pow(self.sigma, 2)
        b0 = torch.pow(other.sigma, 2)

        term1 = torch.log(torch.div(b0, b1))
        term2 = torch.div(
            torch.pow(self.mu - other.mu, 2), b0)
        term3 = torch.div(b1, b0)
        kl_div = (torch.mul(term1 + term2 + term3 - 1, 0.5)).sum()
        return kl_div


class Laplace(nn.Module):
    """Implementation of a Laplace random variable, using softplus for
    the scale parameter and with implementation of sampling and KL
    divergence computation.

    Parameters
    ----------
    mu : Tensor of floats
        Centers of the Laplace distr.

    rho : Tensor of floats
        Scale parameter for the distribution (to be transformed
        via the softplus function)

    device : string
        Device the code will run in (e.g. 'cuda')

    fixed : bool
        Boolean indicating whether the distribution is supposed to be fixed
        or learnt.

    """

    def __init__(self, mu, rho, device='cuda', fixed=False):
        super().__init__()
        self.mu = nn.Parameter(mu, requires_grad=not fixed)
        self.rho = nn.Parameter(rho, requires_grad=not fixed)
        self.device = device

    @property
    def scale(self):
        # We use rho instead of sigma so that sigma is always positive during
        # the optimisation. We use sigma = log(exp(rho)+1)
        return torch.log(1 + torch.exp(self.rho))
        #equivalent to m = nn.Softplus(); return m(self.rho)
        
        

    def sample(self):
        # Return a sample from the Laplace distribution
        # we do scaling due to numerical issues
        epsilon = (0.999*torch.rand(self.scale.size())-0.49999).to(self.device)
        result = self.mu - torch.mul(torch.mul(self.scale, torch.sign(epsilon)),
                                     torch.log(1-2*torch.abs(epsilon)))
        return result

    def compute_kl(self, other):
        # Compute KL divergence between two Laplaces distr. (self and other)
        # (refer to the paper)
        # b is the variance of priors
        b1 = self.scale
        b0 = other.scale
        term1 = torch.log(torch.div(b0, b1))
        aux = torch.abs(self.mu - other.mu)
        term2 = torch.div(aux, b0)
        term3 = torch.div(b1, b0) * torch.exp(torch.div(-aux, b1))

        kl_div = (term1 + term2 + term3 - 1).sum()
        return kl_div


class Linear(nn.Module):
    """Implementation of a Linear layer (reimplemented to use
    truncated normal as initialisation for fair comparison purposes)

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, in_features, out_features, device='cuda'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        # same initialisation as before for the prob layer
        self.weight = nn.Parameter(trunc_normal_(torch.Tensor(
            out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(
            out_features), requires_grad=True)

    def forward(self, input):
        weight = self.weight
        bias = self.bias
        return F.linear(input, weight, bias)


class Lambda_var(nn.Module):
    """Class for the lambda variable included in the objective
    flambda

    Parameters
    ----------
    lamb : float
        initial value

    n : int
        Scaling parameter (lamb_scaled is between 1/sqrt(n) and 1)

    """

    def __init__(self, lamb, n):
        super().__init__()
        self.lamb = nn.Parameter(torch.tensor([lamb]), requires_grad=True)
        self.min = 1/np.sqrt(n)

    @property
    def lamb_scaled(self):
        # We restrict lamb_scaled to be between 1/sqrt(n) and 1.
        m = nn.Sigmoid()
        return (m(self.lamb) * (1-self.min) + self.min)


class ProbLinear(nn.Module):
    """Implementation of a Probabilistic Linear layer.

    Parameters
    ----------
    in_features : int
        Number of input features for the layer

    out_features : int
        Number of output features for the layer

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_layer : Linear object
        Linear layer object used to initialise the prior

    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """

    def __init__(self, in_features, out_features, rho_prior, prior_dist='gaussian', device='cuda', init_prior='random', init_layer=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Set sigma for the truncated gaussian of weights
        sigma_weights = 1/np.sqrt(in_features)

        # INITIALISE PRIOR
        # this means prior is uninformed
        if not init_layer:
            # initalise prior to zeros and rho_prior
            if init_prior == 'zeros':
                bias_mu_prior = torch.zeros(out_features) 
                weights_mu_prior = torch.zeros(out_features, in_features)
                weights_rho_prior = torch.ones(out_features, in_features) * rho_prior
                bias_rho_prior = torch.ones(out_features) * rho_prior
            # initialise prior to random weights and rho_prior
            elif init_prior == 'random':
                weights_mu_prior = trunc_normal_(torch.Tensor(
                    out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
                bias_mu_prior = torch.zeros(out_features) 
                weights_rho_prior = torch.ones(out_features, in_features) * rho_prior
                bias_rho_prior = torch.ones(out_features) * rho_prior
            else: 
                raise RuntimeError(f'Wrong type of prior initialisation!')
        # informed prior
        else: 
            # if init layer is probabilistic
            if hasattr(init_layer.weight, 'rho'):
                weights_mu_prior = init_layer.weight.mu
                bias_mu_prior = init_layer.bias.mu
                weights_rho_prior = init_layer.weight.rho
                bias_rho_prior = init_layer.bias.rho
            # if init layer for prior is not probabilistic
            else:
                weights_mu_prior = init_layer.weight
                bias_mu_prior = init_layer.bias
                weights_rho_prior = torch.ones(out_features, in_features) * rho_prior
                bias_rho_prior = torch.ones(out_features) * rho_prior

        # INITIALISE POSTERIOR
        # WE ASSUME THAT ALWAYS POSTERIOR WILL BE INITIALISED TO PRIOR (UNLESS PRIOR IS INITIALISED TO ALL ZEROS)
        if init_prior == 'zeros':
            weights_mu_init = trunc_normal_(torch.Tensor(
                    out_features, in_features), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_features)
            weights_rho_init = torch.ones(out_features, in_features) * rho_prior
            bias_rho_init = torch.ones(out_features) * rho_prior
        # initialise to prior
        else: 
            weights_mu_init = weights_mu_prior
            bias_mu_init = bias_mu_prior
            weights_rho_init = weights_rho_prior
            bias_rho_init = bias_rho_prior

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_prior.clone(), device=device, fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_prior.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            self.kl_div = self.weight.compute_kl(self.weight_prior) + \
                self.bias.compute_kl(self.bias_prior)

        return F.linear(input, weight, bias)
    
    def __deepcopy__(self,memo):
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        self.kl_div=self.kl_div.detach().clone()
        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        cp.__deepcopy__ = deepcopy_method
        return cp


class ProbConv2d(nn.Module):
    """Implementation of a Probabilistic Convolutional layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels for the layer

    out_channels : int
        Number of output channels for the layer

    kernel_size : int
        size of the convolutional kernel

    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    stride : int
        Stride of the convolution

    padding: int
        Zero-padding added to both sides of the input

    dilation: int
        Spacing between kernel elements

    init_layer : Linear object
        Linear layer object used to initialise the prior

    """

    def __init__(self, in_channels, out_channels, kernel_size, rho_prior, prior_dist='gaussian',
                 device='cuda', stride=1, padding=0, dilation=1, init_prior='random', init_layer=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.init_prior = init_prior

        # Compute and set sigma for the truncated gaussian of weights
        in_features = self.in_channels
        out_features = self.out_channels
        for k in self.kernel_size:
            in_features *= k
        sigma_weights = 1/np.sqrt(in_features)

        # INITIALISE PRIOR
        # this means prior is uninformed
        if not init_layer:
            # initalise prior to zeros and rho_prior
            if init_prior == 'zeros':
                bias_mu_prior = torch.zeros(out_channels) 
                weights_mu_prior = torch.zeros(out_channels, in_channels, *self.kernel_size)
                weights_rho_prior = torch.ones(out_channels, in_channels, *self.kernel_size) * rho_prior
                bias_rho_prior = torch.ones(out_channels) * rho_prior
            # initialise prior to random weights and rho_prior
            elif init_prior == 'random':
                weights_mu_prior = trunc_normal_(torch.Tensor(
                out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
                bias_mu_prior = torch.zeros(out_channels) 
                weights_rho_prior = torch.ones(out_channels, in_channels, *self.kernel_size) * rho_prior
                bias_rho_prior = torch.ones(out_channels) * rho_prior
            else: 
                raise RuntimeError(f'Wrong type of prior initialisation!')
        # informed prior
        else: 
            # if init layer is probabilistic
            if hasattr(init_layer.weight, 'rho'):
                weights_mu_prior = init_layer.weight.mu
                bias_mu_prior = init_layer.bias.mu
                weights_rho_prior = init_layer.weight.rho
                bias_rho_prior = init_layer.bias.rho
            # if init layer for prior is not probabilistic
            else:
                weights_mu_prior = init_layer.weight
                bias_mu_prior = init_layer.bias
                weights_rho_prior = torch.ones(out_channels, in_channels, *self.kernel_size) * rho_prior
                bias_rho_prior = torch.ones(out_channels) * rho_prior

        # INITIALISE POSTERIOR
        # WE ASSUME THAT ALWAYS PRIOR WILL BE INITIALISED TO PRIOR (UNLESS PRIOR IS INITIALISED TO ALL ZEROS)
        if init_prior == 'zeros':
            weights_mu_init = trunc_normal_(torch.Tensor(
                out_channels, in_channels, *self.kernel_size), 0, sigma_weights, -2*sigma_weights, 2*sigma_weights)
            bias_mu_init = torch.zeros(out_channels)
            weights_rho_init = torch.ones(out_channels, in_channels, *self.kernel_size) * rho_prior
            bias_rho_init = torch.ones(out_channels) * rho_prior
        # initialise to prior
        else: 
            weights_mu_init = weights_mu_prior
            bias_mu_init = bias_mu_prior
            weights_rho_init = weights_rho_prior
            bias_rho_init = bias_rho_prior

        if prior_dist == 'gaussian':
            dist = Gaussian
        elif prior_dist == 'laplace':
            dist = Laplace
        else:
            raise RuntimeError(f'Wrong prior_dist {prior_dist}')

        self.weight = dist(weights_mu_init.clone(),
                           weights_rho_init.clone(), device=device, fixed=False)
        self.bias = dist(bias_mu_init.clone(),
                         bias_rho_init.clone(), device=device, fixed=False)
        self.weight_prior = dist(
            weights_mu_prior.clone(), weights_rho_prior.clone(), device=device, fixed=True)
        self.bias_prior = dist(
            bias_mu_prior.clone(), bias_rho_prior.clone(), device=device, fixed=True)

        self.kl_div = 0

    def forward(self, input, sample=False):
        if self.training or sample:
            # during training we sample from the model distribution
            # sample = True can also be set during testing if we
            # want to use the stochastic/ensemble predictors
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            # otherwise we use the posterior mean
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training:
            # sum of the KL computed for weights and biases
            #ipdb.set_trace()
            self.kl_div = self.weight.compute_kl(
                self.weight_prior) + self.bias.compute_kl(self.bias_prior)

        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    # def __deepcopy__(self,memo):
    #     self.kl_div=0
    #     return copy.deepcopy(self,memo)

    def __deepcopy__(self,memo):
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        self.kl_div=self.kl_div.detach().clone()
        cp = copy.deepcopy(self, memo)
        self.__deepcopy__ = deepcopy_method
        cp.__deepcopy__ = deepcopy_method
        return cp
        
        
        

        # custom treatments
        # for instance: cp.id = None

        return cp

class NNet4l(nn.Module):
    """Implementation of a standard Neural Network with 4 layers and dropout
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    def __init__(self, dropout_prob=0.0, device='cuda', features=28*28, classes = 10, neurons=600):
        super().__init__()
        self.l1 = Linear(features, neurons, device)
        self.l2 = Linear(neurons, neurons, device)
        self.l3 = Linear(neurons, neurons, device)
        self.l4 = Linear(neurons, classes, device)
        self.d = nn.Dropout(dropout_prob)
        self.features = features
        self.classes = classes

    def forward(self, x):
        # forward pass for the network
        x = x.view(-1, self.features)
        x = self.d(self.l1(x))
        x = F.relu(x)
        x = self.d(self.l2(x))
        x = F.relu(x)
        x = self.d(self.l3(x))
        x = F.relu(x)
        x = output_transform(self.l4(x), clamping=False)
        return x


class CNNet4l(nn.Module):
    """Implementation of a standard Convolutional Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.d = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        x = self.d(self.conv1(x))
        x = F.relu(x)
        x = self.d(self.conv2(x))
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.d(self.fc1(x))
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class ProbNNet4l(nn.Module):
    """Implementation of a Probabilistic Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : NNet object
        Network object used to initialise the prior
    
    init_prior : string
        string that indicates the way to initialise the prior:
        *"weights" = initialise with init_layer
        *"zeros" = initialise with zeros and rho prior
        *"random" = initialise with random weights and rho prior
        *""

    """


    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', features=28*28, classes=10, neurons=600, init_net=None, init_prior='weights'):
        super().__init__()

        self.l1 = ProbLinear(features, neurons, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l1 if init_net else None, init_prior = init_prior)
        self.l2 = ProbLinear(neurons, neurons, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l2 if init_net else None, init_prior = init_prior)
        self.l3 = ProbLinear(neurons, neurons, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l3 if init_net else None, init_prior = init_prior)
        self.l4 = ProbLinear(neurons, classes, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l4 if init_net else None, init_prior = init_prior)
        self.features = features
        self.classes = classes


    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        x = x.view(-1, self.features)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = F.relu(self.l3(x, sample))
        x = output_transform(self.l4(x, sample), clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div + self.l4.kl_div


class ProbCNNet4l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size,
    number of classes and kernel size).

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : CNNet object
        Network object used to initialise the prior

    """
    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None, init_prior='weights'):
        super().__init__()

        self.conv1 = ProbConv2d(
            1, 32, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv1 if init_net else None, init_prior = init_prior)
        self.conv2 = ProbConv2d(
            32, 64, 3, rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.conv2 if init_net else None, init_prior = init_prior)
        self.fc1 = ProbLinear(9216, 128, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc1 if init_net else None, init_prior=init_prior)
        self.fc2 = ProbLinear(128, 10, rho_prior, prior_dist=prior_dist,
                              device=device, init_layer=init_net.fc2 if init_net else None, init_prior = init_prior)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        # forward pass for the network
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x, sample))
        x = output_transform(self.fc2(x, sample), clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.fc1.kl_div + self.fc2.kl_div


class CNNet9l(nn.Module):
    """Implementation of a Convolutional Neural Network with 9 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    """

    def __init__(self, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(4096, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        # conv layers
        x = self.d(F.relu(self.conv1(x)))
        x = self.d(F.relu(self.conv2(x)))
        x = (F.max_pool2d(x, kernel_size=2, stride=2))
        x = self.d(F.relu(self.conv3(x)))
        x = self.d(F.relu(self.conv4(x)))
        x = (F.max_pool2d(x, kernel_size=2, stride=2))
        x = self.d(F.relu(self.conv5(x)))
        x = self.d(F.relu(self.conv6(x)))
        x = (F.max_pool2d(x, kernel_size=2, stride=2))
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ProbCNNet9l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 9 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : CNNet object
        Network object used to initialise the prior

    """

    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None, init_prior='weights'):
        super().__init__()
        self.conv1 = ProbConv2d(
            in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None, init_prior = init_prior)
        self.conv2 = ProbConv2d(
            in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None, init_prior = init_prior)
        self.conv3 = ProbConv2d(
            in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None, init_prior = init_prior)
        self.conv4 = ProbConv2d(
            in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None, init_prior = init_prior)
        self.conv5 = ProbConv2d(
            in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None, init_prior = init_prior)
        self.conv6 = ProbConv2d(
            in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None, init_prior = init_prior)
        self.fcl1 = ProbLinear(4096, 1024, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl1 if init_net else None, init_prior = init_prior)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl2 if init_net else None, init_prior = init_prior)
        self.fcl3 = ProbLinear(
            512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.fcl3 if init_net else None, init_prior = init_prior)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div


class NNet3l(nn.Module):
    """Implementation of a standard Neural Network with 4 layers and dropout
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(self, dropout_prob=0.0, device='cuda', features=28*28, classes = 10, neurons=100):
        super().__init__()
        self.l1 = Linear(features, neurons, device)
        self.l2 = Linear(neurons, neurons, device)
        self.l3 = Linear(neurons, classes, device)
        self.d = nn.Dropout(dropout_prob)
        self.features = features
        self.classes = classes

    def forward(self, x):
        # forward pass for the network
        x = x.view(-1, self.features)
        x = self.d(self.l1(x))
        x = F.relu(x)
        x = self.d(self.l2(x))
        x = F.relu(x)
        x = output_transform(self.l3(x), clamping=False)
        return x



class ProbNNet3l(nn.Module):
    """Implementation of a Probabilistic Neural Network with 4 layers
    (used for the experiments on MNIST so it assumes a specific input size and
    number of classes)

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : NNet object
        Network object used to initialise the prior

    """

    def __init__(self, rho_prior, prior_dist='gaussian', device='cuda', init_net=None, features=28*28, classes=10, neurons=100, init_prior='weights', init_prior_net=None):
        super().__init__()
        self.l1 = ProbLinear(features, neurons, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l1 if init_net else None, init_prior=init_prior)
        self.l2 = ProbLinear(neurons, neurons, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l2 if init_net else None, init_prior=init_prior)
        self.l3 = ProbLinear(neurons, classes, rho_prior, prior_dist=prior_dist,
                             device=device, init_layer=init_net.l3 if init_net else None, init_prior=init_prior)
        self.features = features
        self.classes = classes

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        x = x.view(-1, self.features)
        x = F.relu(self.l1(x, sample))
        x = F.relu(self.l2(x, sample))
        x = output_transform(self.l3(x, sample), clamping, pmin)
        return x

    def compute_kl(self,recompute=False):
        # KL as a sum of the KL for each individual layer
        if recompute:
            return self.l1.compute_kl() + self.l2.compute_kl() + self.l3.compute_kl() 
        else:
            return self.l1.kl_div + self.l2.kl_div + self.l3.kl_div 



class CNNet13l(nn.Module):
    """Implementation of a Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    """

    def __init__(self, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(2048, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        # conv layers
        x = F.relu(self.d(self.conv1(x)))
        x = F.relu(self.d(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv3(x)))
        x = F.relu(self.d(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv5(x)))
        x = F.relu(self.d(self.conv6(x)))
        x = F.relu(self.d(self.conv7(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv8(x)))
        x = F.relu(self.d(self.conv9(x)))
        x = F.relu(self.d(self.conv10(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ProbCNNet13l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 13 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : CNNet object
        Network object used to initialise the prior

    """

    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None):
        super().__init__()
        self.conv1 = ProbConv2d(in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.conv3 = ProbConv2d(in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.conv4 = ProbConv2d(in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.conv5 = ProbConv2d(in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.conv6 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        self.conv7 = ProbConv2d(in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv7 if init_net else None)
        self.conv8 = ProbConv2d(in_channels=256, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv8 if init_net else None)
        self.conv9 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                device=device, kernel_size=3, padding=1, init_layer=init_net.conv9 if init_net else None)
        self.conv10 = ProbConv2d(in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist,
                                 device=device, kernel_size=3, padding=1, init_layer=init_net.conv10 if init_net else None)
        self.fcl1 = ProbLinear(2048, 1024, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(512, 10, rho_prior=rho_prior, prior_dist=prior_dist,
                               device=device, init_layer=init_net.fcl3 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.relu(self.conv7(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv8(x, sample))
        x = F.relu(self.conv9(x, sample))
        x = F.relu(self.conv10(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div + self.conv6.kl_div + self.conv7.kl_div + self.conv8.kl_div + self.conv9.kl_div + self.conv10.kl_div + self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div


class CNNet15l(nn.Module):
    """Implementation of a Convolutional Neural Network with 15 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).

    Parameters
    ----------
    dropout_prob : float
        probability of an element to be zeroed.

    """

    def __init__(self, dropout_prob):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(
            in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(
            in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.fcl1 = nn.Linear(2048, 1024)
        self.fcl2 = nn.Linear(1024, 512)
        self.fcl3 = nn.Linear(512, 10)
        self.d = nn.Dropout(dropout_prob)

    def forward(self, x):
        # conv layers
        x = F.relu(self.d(self.conv1(x)))
        x = F.relu(self.d(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv3(x)))
        x = F.relu(self.d(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv5(x)))
        x = F.relu(self.d(self.conv6(x)))
        x = F.relu(self.d(self.conv7(x)))
        x = F.relu(self.d(self.conv8(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.d(self.conv9(x)))
        x = F.relu(self.d(self.conv10(x)))
        x = F.relu(self.d(self.conv11(x)))
        x = F.relu(self.d(self.conv12(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.d(self.fcl1(x)))
        x = F.relu(self.d(self.fcl2(x)))
        x = self.fcl3(x)
        x = F.log_softmax(x, dim=1)
        return x


class ProbCNNet15l(nn.Module):
    """Implementation of a Probabilistic Convolutional Neural Network with 15 layers
    (used for the experiments on CIFAR-10 so it assumes a specific input size,
    number of classes, architecture, kernel size, etc.).

    Parameters
    ----------
    rho_prior : float
        prior scale hyperparmeter (to initialise the scale of
        the posterior)

    prior_dist : string
        string that indicates the type of distribution for the
        prior and posterior

    device : string
        Device the code will run in (e.g. 'cuda')

    init_net : CNNet object
        Network object used to initialise the prior

    """

    def __init__(self, rho_prior, prior_dist, device='cuda', init_net=None):
        super().__init__()
        # TODO
        self.conv1 = ProbConv2d(
            in_channels=3, out_channels=32, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv1 if init_net else None)
        self.conv2 = ProbConv2d(
            in_channels=32, out_channels=64, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv2 if init_net else None)
        self.conv3 = ProbConv2d(
            in_channels=64, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv3 if init_net else None)
        self.conv4 = ProbConv2d(
            in_channels=128, out_channels=128, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv4 if init_net else None)
        self.conv5 = ProbConv2d(
            in_channels=128, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv5 if init_net else None)
        self.conv6 = ProbConv2d(
            in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv6 if init_net else None)
        self.conv7 = ProbConv2d(
            in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv7 if init_net else None)
        self.conv8 = ProbConv2d(
            in_channels=256, out_channels=256, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv8 if init_net else None)
        self.conv9 = ProbConv2d(
            in_channels=256, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv9 if init_net else None)
        self.conv10 = ProbConv2d(
            in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv10 if init_net else None)
        self.conv11 = ProbConv2d(
            in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv11 if init_net else None)
        self.conv12 = ProbConv2d(
            in_channels=512, out_channels=512, rho_prior=rho_prior, prior_dist=prior_dist, device=device,
            kernel_size=3, padding=1, init_layer=init_net.conv12 if init_net else None)
        self.fcl1 = ProbLinear(2048, 1024, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl1 if init_net else None)
        self.fcl2 = ProbLinear(1024, 512, rho_prior=rho_prior,
                               prior_dist=prior_dist, device=device, init_layer=init_net.fcl2 if init_net else None)
        self.fcl3 = ProbLinear(
            512, 10, rho_prior=rho_prior, prior_dist=prior_dist, device=device, init_layer=init_net.fcl3 if init_net else None)

    def forward(self, x, sample=False, clamping=True, pmin=1e-4):
        # conv layers
        x = F.relu(self.conv1(x, sample))
        x = F.relu(self.conv2(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x, sample))
        x = F.relu(self.conv4(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv5(x, sample))
        x = F.relu(self.conv6(x, sample))
        x = F.relu(self.conv7(x, sample))
        x = F.relu(self.conv8(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv9(x, sample))
        x = F.relu(self.conv10(x, sample))
        x = F.relu(self.conv11(x, sample))
        x = F.relu(self.conv12(x, sample))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer
        x = F.relu(self.fcl1(x, sample))
        x = F.relu(self.fcl2(x, sample))
        x = self.fcl3(x, sample)
        x = output_transform(x, clamping, pmin)
        return x

    def compute_kl(self):
        # KL as a sum of the KL for each individual layer
        return self.conv1.kl_div + self.conv2.kl_div + self.conv3.kl_div + self.conv4.kl_div + self.conv5.kl_div
        + self.conv6.kl_div + self.conv7.kl_div + \
            self.conv8.kl_div + self.conv9.kl_div + self.conv10.kl_div
        + self.conv11.kl_div + self.conv12.kl_div + \
            self.fcl1.kl_div + self.fcl2.kl_div + self.fcl3.kl_div


def output_transform(x, clamping=True, pmin=1e-4):
    """Computes the log softmax and clamps the values using the
    min probability given by pmin.

    Parameters
    ----------
    x : tensor
        output of the network

    clamping : bool
        whether to clamp the output probabilities

    pmin : float
        threshold of probabilities to clamp.
    """
    # lower bound output prob
    output = F.log_softmax(x, dim=1)
    if clamping:
        output = torch.clamp(output, np.log(pmin))
    return output


def trainNNet(net, optimizer, epoch, train_loader, device='cuda', verbose=False,log_wandb=False, wandb_is_prior=False):
    """Train function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print training metrics

    """
    # train and report training metrics
    net.train()
    total, correct, avgloss = 0.0, 0.0, 0.0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        net.zero_grad()
        output = net(data)
        loss = F.nll_loss(output, target.long())
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        avgloss = avgloss + loss.detach()
    # show the average loss and KL during the epoch
    
    if log_wandb:
        if wandb_is_prior:
            prependString="Prior/"
        else:
            prependString="Posterior/"
        to_log = {
            prependString+"Epoch" : epoch,
            prependString+"Train obj" : avgloss/(batch_id+1),
            prependString+"Train 0-1 Error" : 1-(correct/total)
        }
        wandb.log(to_log)

    if verbose:
        print(
            f"-Epoch {epoch :.5f}, Train loss: {avgloss/(batch_id+1) :.5f}, Train err:  {1-(correct/total):.5f}")


def testNNet(net, test_loader, device='cuda', verbose=False):
    """Test function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    test_loader: DataLoader object
        Test data loader

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print test metrics

    """
    if test_loader is not None:
        net.eval()
        correct, total, avgloss = 0, 0.0, 0.0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(tqdm(test_loader)):
                data, target = data.to(device), target.to(device)
                outputs = net(data)
                avgloss += F.nll_loss(outputs, target.long()).item()
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        if verbose:
            print(
                f"-Prior: Test loss: {avgloss/(batch_id+1) :.5f}, Test err:  {1-(correct/total):.5f}")
        return avgloss/(batch_id+1), 1-(correct/total)
    else: 
        return -1, -1


def trainPNNet(net, optimizer, pbobj, epoch, train_loader, lambda_var=None, optimizer_lambda=None, verbose=True, log_wandb=False, wandb_is_prior=False):
    """Train function for a probabilistic NN (including CNN)

    Parameters
    ----------
    net : ProbNNet/ProbCNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    pbobj : pbobj object
        PAC-Bayes inspired training objective to use for training

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    optimizer_lambda : optim object
        Optimizer to use for the learning the lambda_variable

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print test metrics

    """
    net.train()
    # variables that keep information about the results of optimising the bound
    avgerr, avgbound, avgkl, avgloss, cum_loss = 0.0, 0.0, 0.0, 0.0, 0.0

    if pbobj.objective == 'flamb':
        lambda_var.train()
        # variables that keep information about the results of optimising lambda (only for flamb)
        avgerr_l, avgbound_l, avgkl_l, avgloss_l = 0.0, 0.0, 0.0, 0.0

    if pbobj.objective == 'bbb':
        clamping = False
    else:
        clamping = True

    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(pbobj.device), target.to(pbobj.device)
        net.zero_grad()
        bound, kl, _, loss, err = pbobj.train_obj(
            net, data, target.long(), lambda_var=lambda_var, clamping=clamping)
        
        bound.backward()
        optimizer.step()
        avgbound += bound.item()
        avgkl += kl
        avgloss += loss.item()
        avgerr += err

        if pbobj.objective == 'flamb':
            # for flamb we also need to optimise the lambda variable
            lambda_var.zero_grad()
            bound_l, kl_l, _, loss_l, err_l = pbobj.train_obj(
                net, data, target.long(), lambda_var=lambda_var, clamping=clamping)
            bound_l.backward()
            optimizer_lambda.step()
            avgbound_l += bound_l.item()
            avgkl_l += kl_l
            avgloss_l += loss_l.item()
            avgerr_l += err_l
    if log_wandb:
        if wandb_is_prior:
            prependString="Prior/"
        else:
            prependString="Posterior/"
        cum_loss = cum_loss + avgloss
        to_log = {
            prependString+"Epoch" : epoch,
            prependString+"Train obj" : avgbound/(batch_id+1),
            prependString+"avgKLn" : avgkl/(batch_id+1),
            prependString+"NLL loss" : avgloss/(batch_id+1),
            prependString+"Cumloss(training speed)" : cum_loss,
            prependString+"Train 0-1 Error" : avgerr/(batch_id+1)
        }
        if pbobj.objective == 'flamb':
            to_log["afterLambda Train obj"] = avgbound_l/(batch_id+1)
            to_log["afterLambda avgKLn"] = avgkl_l/(batch_id+1)
            to_log["afterLambda NLL loss"] = avgloss_l/(batch_id+1)
            to_log["afterLambda Train 0-1 Error"] = avgerr_l/(batch_id+1)
            to_log["afterLambda last lambda value"] = lambda_var.lamb_scaled.item()
        wandb.log(to_log)
    if verbose:
        # show the average of the metrics during the epoch
        print(
            f"-Batch average epoch {epoch :.0f} results, Train obj: {avgbound/(batch_id+1) :.5f}, KL/n: {avgkl/(batch_id+1) :.5f}, NLL loss: {avgloss/(batch_id+1) :.5f}, Train 0-1 Error:  {avgerr/(batch_id+1) :.5f}")
        if pbobj.objective == 'flamb':
            print(
                f"-After optimising lambda: Train obj: {avgbound_l/(batch_id+1) :.5f}, KL/n: {avgkl_l/(batch_id+1) :.5f}, NLL loss: {avgloss_l/(batch_id+1) :.5f}, Train 0-1 Error:  {avgerr_l/(batch_id+1) :.5f}, last lambda value: {lambda_var.lamb_scaled.item() :.5f}")


       

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def trainNNetmixup(net, optimizer, epoch, train_loader, device='cuda', verbose=False, alpha=1.0):
    """Train function for a standard NN (including CNN)

    Parameters
    ----------
    net : NNet/CNNet object
        Network object to train

    optimizer : optim object
        Optimizer to use (e.g. SGD/Adam)

    epoch : int
        Current training epoch

    train_loader: DataLoader object
        Train loader to use for training

    device : string
        Device the code will run in (e.g. 'cuda')

    verbose: bool
        Whether to print training metrics

    """
    # train and report training metrics
    if device == 'cuda':
        use_cuda = True
    else: 
        use_cuda = False
    criterion = nn.CrossEntropyLoss()
    #alpha = 0.5
    net.train()
    total, correct, avgloss = 0.0, 0.0, 0.0
    for batch_id, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        net.zero_grad()
        inputs, targets_a, targets_b, lam = mixup_data(data, target.long(),
                                                       alpha, use_cuda)
        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))
        output = net(data)
        loss = mixup_criterion(criterion, output, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        avgloss = avgloss + loss.detach()
    # show the average loss and KL during the epoch
    if verbose:
        print(
            f"-Epoch {epoch :.5f}, Train loss: {avgloss/(batch_id+1) :.5f}, Train err:  {1-(correct/total):.5f}")

def testStochastic(net, test_loader, pbobj, device='cuda', samples=10):
    """Test function for the stochastic predictor using a PNN

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    if test_loader is not None:
        # compute mean test accuracy
        net.eval()
        correct, cross_entropy, total = 0, 0.0, 0.0
        err_samples = np.zeros(samples)
        
        
        for j in range(samples):
            with torch.no_grad():
                for batch_id, (data, target) in enumerate(tqdm(test_loader)):
                    #import ipdb; ipdb.set_trace()
                    outputs = torch.zeros(len(target), pbobj.classes).to(device)
                    data, target = data.to(device), target.to(device)
                    #import ipdb; ipdb.set_trace()
                    for i in range(len(data)):
                        outputs[i, :] = net(data[i:i+1], sample=True,
                                            clamping=True, pmin=pbobj.pmin)
                    cross_entropy += pbobj.compute_empirical_risk(
                        outputs, target.long(), bounded=True).item()
                    pred = outputs.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
            err_samples[j] = 1-(correct/total)
        #print(err_samples)

        return cross_entropy/(batch_id+1), np.mean(err_samples), np.std(err_samples)
    else:
        return -1, -1, -1


def testPosteriorMean(net, test_loader, pbobj, device='cuda'):
    """Test function for the deterministic predictor using a PNN
    (uses the posterior mean)

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    """
    if test_loader is not None:
        net.eval()
        cross_entropy, correct, total = 0.0, 0, 0.0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(tqdm(test_loader)):
                data, target = data.to(device), target.to(device)
                outputs = net(data, sample=False, clamping=True, pmin=pbobj.pmin)
                cross_entropy += pbobj.compute_empirical_risk(
                    outputs, target.long(), bounded=True).item()
                pred = outputs.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        return cross_entropy/(batch_id+1), 1-(correct/total)
    else: 
        return -1, -1


def testEnsemble(net, test_loader, pbobj, device='cuda', samples=100):
    """Test function for the ensemble predictor using a PNN

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    test_loader: DataLoader object
        Test data loader

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    samples: int
        Number of times to sample weights (i.e. members of the ensembles)

    """
    if test_loader is not None:
        net.eval()
        correct, cross_entropy, total = 0, 0.0, 0.0
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(tqdm(test_loader)):
                data, target = data.to(device), target.to(device)
                outputs = torch.zeros(samples, len(target),
                                    pbobj.classes).to(device)
                for i in range(samples):
                    outputs[i] = net(data, sample=True,
                                    clamping=True, pmin=pbobj.pmin)
                avgoutput = outputs.mean(0)
                cross_entropy += pbobj.compute_empirical_risk(
                    avgoutput, target.long(), bounded=True).item()
                pred = avgoutput.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        return cross_entropy/(batch_id+1), 1-(correct/total)
    else:
        return -1, -1


def computeRiskCertificates(net, toolarge, pbobj, device='cuda', lambda_var=None, train_loader=None, whole_train=None):
    """Function to compute risk certificates and other statistics at the end of training

    Parameters
    ----------
    net : PNNet/PCNNet object
        Network object to test

    toolarge: bool
        Whether the dataset is too large to fit in memory (computation done in batches otherwise)

    pbobj : pbobj object
        PAC-Bayes inspired training objective used during training

    device : string
        Device the code will run in (e.g. 'cuda')

    lambda_var : Lambda_var object
        Lambda variable for training objective flamb

    train_loader: DataLoader object
        Data loader for computing the risk certificate (multiple batches, used if toolarge=True)

    whole_train: DataLoader object
        Data loader for computing the risk certificate (one unique batch, used if toolarge=False)

    """
    net.eval()
    with torch.no_grad():
        if toolarge:
            train_obj, kl, loss_ce_train, err_01_train, risk_ce, risk_01 = pbobj.compute_final_stats_risk(
                net, lambda_var=lambda_var, clamping=True, data_loader=train_loader)
        else:
            # a bit hacky, we load the whole dataset to compute the bound
            for data, target in whole_train:
                data, target = data.to(device), target.to(device)
                train_obj, kl, loss_ce_train, err_01_train, risk_ce, risk_01 = pbobj.compute_final_stats_risk(
                    net, lambda_var=lambda_var, clamping=True, input=data, target=target)

    return train_obj, risk_ce, risk_01, kl, loss_ce_train, err_01_train
