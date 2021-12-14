import gc
import time
import shutil
import warnings
import hydra
import numpy as np
import torch
import torchvision
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from flows import MAF, Glow, Ffjord, Flowpp, RealNVP, ResFlow, PlanarFlow
from flows.misc import anomaly_hook
from common.utils import image_plot, save_image, scatter_plot
from flows.dataset import FlowDataLoader
from flows.modules import Logit, Identity
from common.logging import Logging
import torch.optim.lr_scheduler


from torch.distributions.normal import Normal
from torch.distributions.distribution import Distribution
from torch.distributions import constraints
from typing import Dict
from scipy.stats import gennorm
import pandas as pd
import numpy as np
import torch
from torch.distributions import ExponentialFamily,Categorical,constraints,MultivariateNormal, Independent
from torch.distributions.utils import _standard_normal,broadcast_all
from numbers import Real, Number
import math
import copy
from torch.distributions import MultivariateNormal

from collections import defaultdict

from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import torch
import numpy as np

from nnlib.nnlib import utils


def get_sgd_noise(model, arch_type, curr_device, opt, full_loader):
    """
    :param model:
    :param arch_type:
    :param curr_device:
    :param opt:
    :param full_loader:
    :return:
    """
    gc.collect()
    # We do NOT want to be training on the full gradients, just calculating them!!!!
    model.eval()
    grads, sizes = [], []
    for batch_idx, (inputs, labels) in enumerate(full_loader):
        inputs, labels = inputs.to(curr_device), labels.to(curr_device)
        opt.zero_grad()
        if arch_type == 'mlp':
            inputs = inputs.view(inputs.size(0), -1)
        outputs = model(inputs)
        loss = model.loss(outputs, labels)
        loss.backward()
        grad = [param.grad.cpu().clone() for param in model.parameters()]
        # grad = [p.grad.clone() for p in model.parameters()]
        size = inputs.shape[0]
        grads.append(grad)
        sizes.append(size)

    flat_grads = []
    for grad in grads:
        flat_grads.append(torch.cat([g.reshape(-1) for g in grad]))
    full_grads = torch.zeros(flat_grads[-1].shape)
    # Exact_Grad = torch.zeros(Flat_Grads[-1].shape).cuda()
    for g, s in zip(flat_grads, sizes):
        full_grads += g * s
    full_grads /= np.sum(sizes)
    gc.collect()
    flat_grads = torch.stack(flat_grads)
    sgd_noise = (flat_grads-full_grads).cpu()
    # Grad_noise = Flat_Grads-Exact_Grad
    return full_grads, sgd_noise

def get_tail_index(sgd_noise):
    """
    Returns an estimate of the tail-index term of the alpha-stable distribution for the stochastic gradient noise.
    In the paper, the tail-index is denoted by $\alpha$. Simsekli et. al. use the estimator posed by Mohammadi et al. in
    2015.
    :param sgd_noise:
    :return: tail-index term ($\alpha$) for an alpha-stable distribution
    """
    X = sgd_noise.reshape(-1)
    X = X[X.nonzero()]
    K = len(X)
    if len(X.shape)>1:
        X = X.squeeze()
    K1 = int(np.floor(np.sqrt(K)))
    K2 = int(K1)
    X = X[:K1*K2].reshape((K2, K1))
    Y = X.sum(1)
    # X = X.cpu().clone(); Y = Y.cpu().clone()
    a = torch.log(torch.abs(Y)).mean()
    b = (torch.log(torch.abs(X[:int(K2/4),:])).mean()+torch.log(torch.abs(X[int(K2/4):int(K2/2),:])).mean()+torch.log(torch.abs(X[int(K2/2):3*int(K2/4),:])).mean()+torch.log(torch.abs(X[3*int(K2/4):,:])).mean())/4
    alpha_hat = np.log(K1)/(a-b).item()
    return alpha_hat

class GenNormal(ExponentialFamily):
    r"""
    Creates a normal (also called Gaussian) distribution parameterized by
    :attr:`loc` and :attr:`scale`.
    Example::
        >>> m = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        >>> m.sample()  # normally distributed with loc=0 and scale=1
        tensor([ 0.1046])
    Args:
        loc (float or Tensor): mean of the distribution (often referred to as mu)
        scale (float or Tensor): standard deviation of the distribution
            (often referred to as sigma)
    """
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'p': constraints.real}
    support = constraints.real
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.loc

    @property
    def stddev(self):
        return self.scale

    @property
    def exponent(self):
        return self.p

    @property
    def variance(self):
        return self.stddev.pow(2)

    def __init__(self, loc, scale,p, validate_args=None):
        self.loc, self.scale, self.p = broadcast_all(loc, scale, p)
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super(GenNormal, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(GenNormal, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        new.p = self.p.expand(batch_shape)
        super(GenNormal, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new
    def rsample(self, sample_shape=torch.Size()):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        shape = self._extended_shape(sample_shape)
        #print('shape',shape)
        ipower = 1.0 / self.p
        
        ipower = ipower.mean()#.cpu()
        gamma_dist = torch.distributions.Gamma(ipower, 1.0)
        
        gamma_sample = gamma_dist.rsample(shape)#.cpu()
        
        binary_sample = torch.randint(low=0, high=2, size=shape, dtype=self.loc.dtype) * 2 - 1
        
        #print('~~~~~~',binary_sample.shape,gamma_sample.shape)
              
        if len(binary_sample.shape) ==  len(gamma_sample.shape) - 1:
            gamma_sample = gamma_sample.squeeze(len(gamma_sample.shape) - 1)
            
        #print('~~~',binary_sample.shape,gamma_sample.shape)
        sampled = binary_sample.to(device) * torch.pow(torch.abs(gamma_sample).to(device), ipower)
        
        print(self.loc.detach().cpu().numpy(),':::::',self.scale.detach().cpu().numpy(),':::::',self.p.detach().cpu().numpy())
        #return self.loc.to(device) + self.scale.to(device) * sampled.to(device)
        return self.loc + self.scale * sampled

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        #print('shape',shape)
        with torch.no_grad():
            ipower = 1.0 / self.p
            ipower = ipower#.cpu()
            gamma_dist = torch.distributions.Gamma(ipower, 1.0)
            gamma_sample = gamma_dist.sample(shape)#.cpu()
            binary_sample = (torch.randint(low=0, high=2, size=shape, dtype=self.loc.dtype) * 2 - 1)
            if (len(gamma_sample.shape) == len(binary_sample.shape) + 1) and gamma_sample.shape[-1]==gamma_sample.shape[-2]:
              gamma_sample = gamma_dist.sample(shape[0:-1])#.cpu()
              
#             
#             
#             
#             
            if type(ipower) == torch.Tensor:
              sampled = binary_sample.to(gamma_sample.device).squeeze() * torch.pow(torch.abs(gamma_sample.squeeze()).to(gamma_sample.device), ipower.to(gamma_sample.device))
            else:
              sampled = binary_sample.squeeze() * torch.pow(torch.abs(gamma_sample.squeeze()), torch.FloatTensor(ipower))
            #print(self.loc.item(),':::::',self.scale.item(),':::::',self.p.item())
            return self.loc + self.scale * sampled

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        # compute the variance
        var = (self.scale ** 2)
        log_scale = math.log(self.scale) if isinstance(self.scale, Real) else self.scale.log()
        return (-((value - self.loc) ** 2) / (2 * var) - log_scale - math.log(math.sqrt(2 * math.pi)))

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return 0.5 * (1 + torch.erf((value - self.loc) * self.scale.reciprocal() / math.sqrt(2)))

    def icdf(self, value):
        return self.loc + self.scale * torch.erfinv(2 * value - 1) * math.sqrt(2)

    def entropy(self):
        return 0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.scale)

    @property
    def _natural_params(self):
        return (self.loc / self.scale.pow(2), -0.5 * self.scale.pow(2).reciprocal())

    def _log_normalizer(self, x, y):
        return -0.25 * x.pow(2) / y + 0.5 * torch.log(-math.pi / y)
                                    
                                    
class StableNormal(Normal):
    """
    Add stable cdf for implicit reparametrization, and stable _log_cdf.
    """
    
    # Override default
    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ndtr(self._standardise(value))
    
    # NOTE: This is not necessary for implicit reparam.
    def _log_cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return log_ndtr(self._standardise(value))
    
    def _standardise(self, x):
        return (x - self.loc) * self.scale.reciprocal()

#
# Below are based on the investigation in https://github.com/pytorch/pytorch/issues/52973#issuecomment-787587188
# and implementations in SciPy and Tensorflow Probability
#

def ndtr(value: torch.Tensor):
    """
    Standard Gaussian cumulative distribution function.
    Based on the SciPy implementation of ndtr
    https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c#L201-L224
    """
    sqrt_half = torch.sqrt(torch.tensor(0.5, dtype=value.dtype))
    x = value * sqrt_half
    z = torch.abs(x)
    y = 0.5 * torch.erfc(z)
    output = torch.where(z < sqrt_half,
                        0.5 + 0.5 * torch.erf(x),
                        torch.where(x > 0, 1 - y, y))
    return output


# log_ndtr uses different functions over the ranges
# (-infty, lower](lower, upper](upper, infty)
# Lower bound values were chosen by examining where the support of ndtr
# appears to be zero, relative to scipy's (which is always 64bit). They were
# then made more conservative just to be safe. (Conservative means use the
# expansion more than we probably need to.)
LOGNDTR_FLOAT64_LOWER = -20.
LOGNDTR_FLOAT32_LOWER = -10.

# Upper bound values were chosen by examining for which values of 'x'
# Log[cdf(x)] is 0, after which point we need to use the approximation
# Log[cdf(x)] = Log[1 - cdf(-x)] approx -cdf(-x). We chose a value slightly
# conservative, meaning we use the approximation earlier than needed.
LOGNDTR_FLOAT64_UPPER = 8.
LOGNDTR_FLOAT32_UPPER = 5.

def log_ndtr(value: torch.Tensor):
    """
    Standard Gaussian log-cumulative distribution function.
    This is based on the TFP and SciPy implementations.
    https://github.com/tensorflow/probability/blame/master/tensorflow_probability/python/internal/special_math.py#L156-L245
    https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c#L316-L345
    """
    dtype = value.dtype
    if dtype == torch.float64:
        lower, upper = LOGNDTR_FLOAT64_LOWER, LOGNDTR_FLOAT64_UPPER
    elif dtype == torch.float32:
        lower, upper = LOGNDTR_FLOAT32_LOWER, LOGNDTR_FLOAT32_UPPER
    else:
        raise TypeError(f'dtype={value.dtype} is not supported.')
    
    # When x < lower, then we perform a fixed series expansion (asymptotic)
    # = log(cdf(x)) = log(1 - cdf(-x)) = log(1 / 2 * erfc(-x / sqrt(2)))
    # = log(-1 / sqrt(2 * pi) * exp(-x ** 2 / 2) / x * (1 + sum))
    # When x >= lower and x <= upper, then we simply perform log(cdf(x))
    # When x > upper, then we use the approximation log(cdf(x)) = log(1 - cdf(-x)) \approx -cdf(-x)
    # The above approximation comes from Taylor expansion of log(1 - y) = -y - y^2/2 - y^3/3 - y^4/4 ...
    # So for a small y the polynomial terms are even smaller and negligible.
    # And we know that for x > upper, y = cdf(x) will be very small.
    return torch.where(value > upper,
                       -ndtr(-value),
                       torch.where(value >= lower,
                                   torch.log(ndtr(value)),
                                   log_ndtr_series(value)))

def log_ndtr_series(value: torch.Tensor, num_terms=3):
    """
    Function to compute the asymptotic series expansion of the log of normal CDF
    at value.
    This is based on the SciPy implementation.
    https://github.com/scipy/scipy/blob/master/scipy/special/cephes/ndtr.c#L316-L345
    """
    # sum = sum_{n=1}^{num_terms} (-1)^{n} (2n - 1)!! / x^{2n}))
    value_sq = value ** 2
    t1 = -0.5 * (np.log(2 * np.pi) + value_sq) - torch.log(-value)
    t2 = torch.zeros_like(value)
    value_even_power = value_sq.clone()
    double_fac = 1
    multiplier = -1
    for n in range(1, num_terms + 1):
        t2.add_(multiplier * double_fac / value_even_power)
        value_even_power.mul_(value_sq)
        double_fac *= (2 * n - 1)
        multiplier *= -1
    return t1 + torch.log1p(t2)

class MixtureSameFamily(Distribution):
    r"""
    The `MixtureSameFamily` distribution implements a (batch of) mixture
    distribution where all component are from different parameterizations of
    the same distribution type. It is parameterized by a `Categorical`
    "selecting distribution" (over `k` component) and a component
    distribution, i.e., a `Distribution` with a rightmost batch shape
    (equal to `[k]`) which indexes each (batch of) component.
    Copied from PyTorch 1.8, so that it can be used with earlier PyTorch versions.
    Examples::
        # Construct Gaussian Mixture Model in 1D consisting of 5 equally
        # weighted normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Normal(torch.randn(5,), torch.rand(5,))
        >>> gmm = MixtureSameFamily(mix, comp)
        # Construct Gaussian Mixture Modle in 2D consisting of 5 equally
        # weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.ones(5,))
        >>> comp = D.Independent(D.Normal(
                     torch.randn(5,2), torch.rand(5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)
        # Construct a batch of 3 Gaussian Mixture Models in 2D each
        # consisting of 5 random weighted bivariate normal distributions
        >>> mix = D.Categorical(torch.rand(3,5))
        >>> comp = D.Independent(D.Normal(
                    torch.randn(3,5,2), torch.rand(3,5,2)), 1)
        >>> gmm = MixtureSameFamily(mix, comp)
    Args:
        mixture_distribution: `torch.distributions.Categorical`-like
            instance. Manages the probability of selecting component.
            The number of categories must match the rightmost batch
            dimension of the `component_distribution`. Must have either
            scalar `batch_shape` or `batch_shape` matching
            `component_distribution.batch_shape[:-1]`
        component_distribution: `torch.distributions.Distribution`-like
            instance. Right-most batch dimension indexes component.
    """
    arg_constraints: Dict[str, constraints.Constraint] = {}
    has_rsample = False

    def __init__(self,
                 mixture_distribution,
                 component_distribution,
                 validate_args=None):
        self._mixture_distribution = mixture_distribution
        self._component_distribution = component_distribution

        if not isinstance(self._mixture_distribution, Categorical):
            raise ValueError(" The Mixture distribution needs to be an "
                             " instance of torch.distribtutions.Categorical")

        if not isinstance(self._component_distribution, Distribution):
            raise ValueError("The Component distribution need to be an "
                             "instance of torch.distributions.Distribution")

        # Check that batch size matches
        mdbs = self._mixture_distribution.batch_shape
        cdbs = self._component_distribution.batch_shape[:-1]
        for size1, size2 in zip(reversed(mdbs), reversed(cdbs)):
            if size1 != 1 and size2 != 1 and size1 != size2:
                raise ValueError("`mixture_distribution.batch_shape` ({0}) is not "
                                 "compatible with `component_distribution."
                                 "batch_shape`({1})".format(mdbs, cdbs))

        # Check that the number of mixture component matches
        km = self._mixture_distribution.logits.shape[-1]
        kc = self._component_distribution.batch_shape[-1]
        if km is not None and kc is not None and km != kc:
            raise ValueError("`mixture_distribution component` ({0}) does not"
                             " equal `component_distribution.batch_shape[-1]`"
                             " ({1})".format(km, kc))
        self._num_component = km

        event_shape = self._component_distribution.event_shape
        self._event_ndims = len(event_shape)
        super(MixtureSameFamily, self).__init__(batch_shape=cdbs,
                                                event_shape=event_shape,
                                                validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        batch_shape = torch.Size(batch_shape)
        batch_shape_comp = batch_shape + (self._num_component,)
        new = self._get_checked_instance(MixtureSameFamily, _instance)
        new._component_distribution = \
            self._component_distribution.expand(batch_shape_comp)
        new._mixture_distribution = \
            self._mixture_distribution.expand(batch_shape)
        new._num_component = self._num_component
        new._event_ndims = self._event_ndims
        event_shape = new._component_distribution.event_shape
        super(MixtureSameFamily, new).__init__(batch_shape=batch_shape,
                                               event_shape=event_shape,
                                               validate_args=False)
        new._validate_args = self._validate_args
        return new

    @constraints.dependent_property
    def support(self):
        # FIXME this may have the wrong shape when support contains batched
        # parameters
        return self._component_distribution.support

    @property
    def mixture_distribution(self):
        return self._mixture_distribution

    @property
    def component_distribution(self):
        return self._component_distribution

    @property
    def mean(self):
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        return torch.sum(probs * self.component_distribution.mean,
                         dim=-1 - self._event_ndims)  # [B, E]

    @property
    def variance(self):
        # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
        probs = self._pad_mixture_dimensions(self.mixture_distribution.probs)
        mean_cond_var = torch.sum(probs * self.component_distribution.variance,
                                  dim=-1 - self._event_ndims)
        var_cond_mean = torch.sum(probs * (self.component_distribution.mean -
                                           self._pad(self.mean)).pow(2.0),
                                  dim=-1 - self._event_ndims)
        return mean_cond_var + var_cond_mean

    def cdf(self, x):
        x = self._pad(x)
        cdf_x = self.component_distribution.cdf(x)
        mix_prob = self.mixture_distribution.probs

        return torch.sum(cdf_x * mix_prob, dim=-1)

    def log_prob(self, x):
        if self._validate_args:
            self._validate_sample(x)
        x = self._pad(x)
        log_prob_x = self.component_distribution.log_prob(x)  # [S, B, k]
        log_mix_prob = torch.log_softmax(self.mixture_distribution.logits,
                                         dim=-1)  # [B, k]
        return torch.logsumexp(log_prob_x + log_mix_prob, dim=-1)  # [S, B]

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            sample_len = len(sample_shape)
            batch_len = len(self.batch_shape)
            gather_dim = sample_len + batch_len
            es = self.event_shape

            # mixture samples [n, B]
            mix_sample = self.mixture_distribution.sample(sample_shape)
            mix_shape = mix_sample.shape

            # component samples [n, B, k, E]
            comp_samples = self.component_distribution.sample(sample_shape)

            # Gather along the k dimension
            mix_sample_r = mix_sample.reshape(
                mix_shape + torch.Size([1] * (len(es) + 1)))
            mix_sample_r = mix_sample_r.repeat(
                torch.Size([1] * len(mix_shape)) + torch.Size([1]) + es)

            samples = torch.gather(comp_samples, gather_dim, mix_sample_r)
            return samples.squeeze(gather_dim)

    def _pad(self, x):
        return x.unsqueeze(-1 - self._event_ndims)

    def _pad_mixture_dimensions(self, x):
        dist_batch_ndims = self.batch_shape.numel()
        cat_batch_ndims = self.mixture_distribution.batch_shape.numel()
        pad_ndims = 0 if cat_batch_ndims == 1 else \
            dist_batch_ndims - cat_batch_ndims
        xs = x.shape
        x = x.reshape(xs[:-1] + torch.Size(pad_ndims * [1]) +
                      xs[-1:] + torch.Size(self._event_ndims * [1]))
        return x

    def __repr__(self):
        args_string = '\n  {},\n  {}'.format(self.mixture_distribution,
                                             self.component_distribution)
        return 'MixtureSameFamily' + '(' + args_string + ')'


class ReparametrizedMixtureSameFamily(MixtureSameFamily):
    """
    Adds rsample method to the MixtureSameFamily method
    that implements implicit reparametrization.
    """
    has_rsample = True
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not self._component_distribution.has_rsample:
            raise ValueError('Cannot reparameterize a mixture of non-reparameterized components.')

        # NOTE: Not necessary for implicit reparametrisation.
        if not callable(getattr(self._component_distribution, '_log_cdf', None)):
            warnings.warn(message=('The component distributions do not have numerically stable '
                                   '`_log_cdf`, will use torch.log(cdf) instead, which may not '
                                   'be stable. NOTE: this will not affect implicit reparametrisation.'),
                        )

    def rsample(self, sample_shape=torch.Size()):
        """Adds reparameterization (pathwise) gradients to samples of the mixture.
        
        Based on Tensorflow Probability implementation
        https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/distributions/mixture_same_family.py#L433-L498
        Implicit reparameterization gradients are
        .. math::
            dx/dphi = -(d transform(x, phi) / dx)^-1 * d transform(x, phi) / dphi,
        where transform(x, phi) is distributional transform that removes all
        parameters from samples x.
        We implement them by replacing x with
        -stop_gradient(d transform(x, phi) / dx)^-1 * transform(x, phi)]
        for the backward pass (gradient computation).
        The derivative of this quantity w.r.t. phi is then the implicit
        reparameterization gradient.
        Note that this replaces the gradients w.r.t. both the mixture
        distribution parameters and components distributions parameters.
        Limitations:
        1. Fundamental: components must be fully reparameterized.
        2. Distributional transform is currently only implemented for
            factorized components.
        Args:
            x: Sample of mixture distribution
        Returns:
            Tensor with same value as x, but with reparameterization gradients
        """
        x = self.sample(sample_shape=sample_shape)

        event_size = prod(self.event_shape)
        if event_size != 1:
            # Multivariate case
            x_2d_shape = (-1, event_size)

            # Perform distributional transform of x in [S, B, E] shape,
            # but have Jacobian of size [S*prod(B), prod(E), prod(E)].
            def reshaped_distributional_transform(x_2d):
                return torch.reshape(
                        self._distributional_transform(x_2d.reshape(x.shape)),
                        x_2d_shape)

            # transform_2d: [S*prod(B), prod(E)]
            # jacobian: [S*prod(B), prod(E), prod(E)]
            x_2d = x.reshape(x_2d_shape)
            transform_2d = reshaped_distributional_transform(x_2d)
            # At the moment there isn't an efficient batch-Jacobian implementation
            # in PyTorch, so we have to loop over the batch.
            # TODO: Use batch-Jacobian, once one is implemented in PyTorch.
            # or vmap: https://github.com/pytorch/pytorch/issues/42368
            jac = x_2d.new_zeros(x_2d.shape + (x_2d.shape[-1],))
            for i in range(x_2d.shape[0]):
                jac[i, ...] = jacobian(self._distributional_transform, x_2d[i, ...]).detach()

            # We only provide the first derivative; the second derivative computed by
            # autodiff would be incorrect, so we raise an error if it is requested.
            # TODO: prevent 2nd derivative of transform_2d.

            # Compute [- stop_gradient(jacobian)^-1 * transform] by solving a linear
            # system. The Jacobian is lower triangular because the distributional
            # transform for i-th event dimension does not depend on the next
            # dimensions.
            surrogate_x_2d = -torch.triangular_solve(transform_2d[..., None], jac.detach(), upper=False)[0]
            surrogate_x = surrogate_x_2d.reshape(x.shape)
        else:
            # For univariate distributions the Jacobian/derivative of the transformation is the
            # density, so the computation is much cheaper.
            transform = self._distributional_transform(x)
            log_prob_x = self.log_prob(x)
            
            if self._event_ndims > 1:
                log_prob_x = log_prob_x.reshape(log_prob_x.shape + (1,)*self._event_ndims)

            surrogate_x = -transform*torch.exp(-log_prob_x.detach())

        # Replace gradients of x with gradients of surrogate_x, but keep the value.
        return x + (surrogate_x - surrogate_x.detach())

    def _distributional_transform(self, x):
        """Performs distributional transform of the mixture samples.
        Based on Tensorflow Probability implementation
        https://github.com/tensorflow/probability/blob/v0.12.2/tensorflow_probability/python/distributions/mixture_same_family.py#L500-L574
        Distributional transform removes the parameters from samples of a
        multivariate distribution by applying conditional CDFs:
        .. math::
            (F(x_1), F(x_2 | x1_), ..., F(x_d | x_1, ..., x_d-1))
        (the indexing is over the 'flattened' event dimensions).
        The result is a sample of product of Uniform[0, 1] distributions.
        We assume that the components are factorized, so the conditional CDFs become
        .. math::
          `F(x_i | x_1, ..., x_i-1) = sum_k w_i^k F_k (x_i),`
        where :math:`w_i^k` is the posterior mixture weight: for :math:`i > 0`
        :math:`w_i^k = w_k prob_k(x_1, ..., x_i-1) / sum_k' w_k' prob_k'(x_1, ..., x_i-1)`
        and :math:`w_0^k = w_k` is the mixture probability of the k-th component.
        Args:
            x: Sample of mixture distribution
        Returns:
            Result of the distributional transform
        """
        # Obtain factorized components distribution and assert that it's
        # a scalar distribution.
        if isinstance(self._component_distribution, tdistr.Independent):
            univariate_components = self._component_distribution.base_dist
        else:
            univariate_components = self._component_distribution

        # Expand input tensor and compute log-probs in each component
        x = self._pad(x)  # [S, B, 1, E]
        # NOTE: Only works with fully-factorised distributions!
        log_prob_x = univariate_components.log_prob(x)  # [S, B, K, E]
        
        event_size = prod(self.event_shape)
        if event_size != 1:
            # Multivariate case
            # Compute exclusive cumulative sum
            # log prob_k (x_1, ..., x_i-1)
            cumsum_log_prob_x = log_prob_x.reshape(-1, event_size)  # [S*prod(B)*K, prod(E)]
            cumsum_log_prob_x = torch.cumsum(cumsum_log_prob_x, dim=-1)
            cumsum_log_prob_x = cumsum_log_prob_x.roll(1, -1)
            cumsum_log_prob_x[:, 0] = 0
            cumsum_log_prob_x = cumsum_log_prob_x.reshape(log_prob_x.shape)

            logits_mix_prob = self._pad_mixture_dimensions(self._mixture_distribution.logits)

            # Logits of the posterior weights: log w_k + log prob_k (x_1, ..., x_i-1)
            log_posterior_weights_x = logits_mix_prob + cumsum_log_prob_x
            
            # Normalise posterior weights
            component_axis = -self._event_ndims-1
            posterior_weights_x = torch.softmax(log_posterior_weights_x, dim=component_axis)

            cdf_x = univariate_components.cdf(x)  # [S, B, K, E]
            return torch.sum(posterior_weights_x * cdf_x, dim=component_axis)
        else:
            # For univariate distributions logits of the posterior weights = log w_k
            log_posterior_weights_x = self._mixture_distribution.logits
            posterior_weights_x = torch.softmax(log_posterior_weights_x, dim=-1)
            posterior_weights_x = self._pad_mixture_dimensions(posterior_weights_x)

            cdf_x = univariate_components.cdf(x)  # [S, B, K, E]
            component_axis = -self._event_ndims-1
            return torch.sum(posterior_weights_x * cdf_x, dim=component_axis)


    def _log_cdf(self, x):
        x = self._pad(x)
        if callable(getattr(self._component_distribution, '_log_cdf', None)):
            log_cdf_x = self.component_distribution._log_cdf(x)
        else:
            # NOTE: This may be unstable
            log_cdf_x = torch.log(self.component_distribution.cdf(x))

        if isinstance(self.component_distribution, (tdistr.Bernoulli, tdistr.Binomial, tdistr.ContinuousBernoulli, 
                                                    tdistr.Geometric, tdistr.NegativeBinomial, tdistr.RelaxedBernoulli)):
            log_mix_prob = torch.sigmoid(self.mixture_distribution.logits)
        else:
            log_mix_prob = F.log_softmax(self.mixture_distribution.logits, dim=-1)

        return torch.logsumexp(log_cdf_x + log_mix_prob, dim=-1)



networks = {
    'planar': PlanarFlow,
    'realnvp': RealNVP,
    'glow': Glow,
    'flow++': Flowpp,
    'maf': MAF,
    'resflow': ResFlow,
    'ffjord': Ffjord,
}

# -----------------------------------------------
# logging
# -----------------------------------------------
logger = Logging(__file__)


# -----------------------------------------------
# train/eval model
# -----------------------------------------------
class Model(object):
    def __init__(self, dims=(2, ), datatype=None, cfg=None, **dist_args):
        if torch.cuda.is_available():
            self.device = torch.device('cuda', cfg.run.gpu)
        else:
            self.device = torch.device('cpu')

        self.name = cfg.network.name
        self._batch_size = cfg.train.samples
        self.dims = dims
        self.dimension = np.prod(dims)
        self._var_base_dist = dist_args['variable_bd']
        self._loss = dist_args['loss']
        self.net = networks[self.name](dims=self.dims, datatype=datatype, cfg=cfg.network)
        
        if dist_args['bd_family'] == 'mvn':

            self.mu = torch.nn.Parameter(torch.zeros(self.dimension, dtype=torch.float32).to(self.device) + dist_args['mu'])
            self.covar = torch.nn.Parameter(torch.eye(self.dimension, dtype=torch.float32).to(self.device) + dist_args['cov'])

            if self._var_base_dist == True:
                self.mu.requires_grad = True
                self.covar.requires_grad = True
            else:
                self.mu.requires_grad = False
                self.covar.requires_grad = False
                
            self.base_distribution = MultivariateNormal(self.mu, self.covar)
            self.net.dp1 = self.mu
            self.net.dp2 = self.covar
            
            
        elif dist_args['bd_family'] == 'ggd':
            self.loc = torch.nn.Parameter(torch.zeros(self.dimension, dtype=torch.float32).to(self.device) + dist_args['loc'], requires_grad = True)
            self.scale = torch.nn.Parameter(torch.zeros(self.dimension, dtype=torch.float32).to(self.device) + dist_args['scale'], requires_grad = True)
            self.p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32).to(self.device) + dist_args['p'], requires_grad = True)
            
            if self._var_base_dist == True:
                self.loc.requires_grad = True
                self.scale.requires_grad = True
                self.p.requires_grad = True

            else:
                self.loc.requires_grad = False
                self.scale.requires_grad = False
                self.p.requires_grad = False
                
            #print('pshape',self.p.shape,GenNormal(loc=self.loc,scale=self.scale,p=self.p))
            self.base_distribution = Independent(GenNormal(loc=self.loc,scale=self.scale,p=self.p),1)
            #print(self.loc,self.scale,self.p)
            self.net.dp1 = self.loc
            self.net.dp2 = self.scale
            self.net.dp3 = self.p
            
        elif dist_args['bd_family'] == 'mvggd':
            self.loc = torch.nn.Parameter(torch.zeros(self.dimension, dtype=torch.float32).to(self.device) + dist_args['loc'], requires_grad = True)
            self.scale = torch.nn.Parameter(torch.zeros(self.dimension, dtype=torch.float32).to(self.device) + dist_args['scale'], requires_grad = True)
            self.p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32).to(self.device) + dist_args['p'], requires_grad = True)
            self.dw = torch.nn.Parameter(torch.zeros(self.dimension, dtype=torch.float32).to(self.device) + dist_args['dw'], requires_grad = True)
            
            if self._var_base_dist == True:
                self.loc.requires_grad = True
                self.scale.requires_grad = True
                self.p.requires_grad = True
                self.dw.requires_grad = True

            else:
                self.loc.requires_grad = False
                self.scale.requires_grad = False
                self.p.requires_grad = False
                self.dw.requires_grad = False
                
            
            mix = torch.distributions.Categorical(self.dw)
            comp = GenNormal(self.loc, self.scale,self.p)

            self.base_distribution = Independent(ReparametrizedMixtureSameFamily(mix, comp),1)
            self.net.dp1 = self.loc
            self.net.dp2 = self.scale
            self.net.dp3 = self.p
            self.net.dp3 = self.dw

        

        
        self.net.to(self.device)

        if cfg.optimizer.name == 'rmsprop':
            self.optim = torch.optim.RMSprop(self.net.parameters(),
                                             lr=cfg.optimizer.lr,
                                             weight_decay=cfg.optimizer.weight_decay)
        elif cfg.optimizer.name == 'adam':
            self.optim = torch.optim.Adam(self.net.parameters(),
                                          lr=cfg.optimizer.lr,
                                          betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                                          weight_decay=cfg.optimizer.weight_decay)
        else:
            raise Exception('optimizer "%s" is currently not supported' % (cfg.optimizer.name))

        self.schduler = torch.optim.lr_scheduler.StepLR(self.optim,
                                                        step_size=cfg.optimizer.decay_steps,
                                                        gamma=cfg.optimizer.decay_ratio,verbose=True)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def train_on_batch(self, y):
        y = y.to(self.device)
        y = y.contiguous()

        z, log_det_jacobian = self.net(y)
        z = z.view(y.size(0), -1)
        #print('!!!',self.base_distribution.log_prob(z).shape,log_det_jacobian.shape)
        if self._loss == 'ML':
            loss = -1.0 * torch.mean(self.base_distribution.log_prob(z) + log_det_jacobian)
        elif self._loss == 'TA':
            beta = -1
            logp = self.base_distribution.log_prob(z)
            #logp = self.base_distribution.log_prob(y)
            logq = self.log_py(z)
            #logq = self.log_py(y)
            diff = logp - logq
            weights = torch.exp(diff - diff.max())
            prob = torch.sign(weights.unsqueeze(1) - weights.unsqueeze(0))
            prob = torch.greater(prob, 0.5).float()
            F = 1 - prob.sum(1) / self._batch_size
            gammas = F ** beta
            gammas /= gammas.sum()
            loss = -torch.sum(torch.unsqueeze(gammas * diff, 1))
        

        self.optim.zero_grad()
        loss.backward(retain_graph=True)
        self.optim.step()
        self.schduler.step()

        return z, loss

    def save_ckpt(self, step, filename):
        ckpt = {
            'net': self.net.state_dict(),
            'optim': self.optim.state_dict(),
            'step': step,
        }
        torch.save(ckpt, filename)

    def load_ckpt(self, filename):
        ckpt = torch.load(filename)
        self.net.load_state_dict(ckpt['net'])
        self.optim.load_state_dict(ckpt['optim'])
        epoch = ckpt['step']
        return epoch

    def sample_y(self, n):
        z = self.sample_z(n)
        z = z.to(self.device)

        y, log_det_jacobians = self.net.backward(z.view(-1, *self.dims))
        log_p = self.base_distribution.log_prob(z) - log_det_jacobians

        return y, torch.exp(log_p)

    def sample_z(self, n):
        if self._var_base_dist:
            return self.base_distribution.rsample([n])
        else:
            return self.base_distribution.sample([n])
    def log_py(self, y):
        y = y.to(self.device)
        z, log_det_jacobians = self.net(y)
        return self.log_pz(z) + log_det_jacobians

    def log_pz(self, z):
        return self.base_distribution.log_prob(z)

    def py(self, y):
        return torch.exp(self.log_py(y))

    def pz(self, z):
        return torch.exp(self.log_pz(z))

    def report(self, writer, y_data, step=0, save_files=False,prefix=''):
        # set to evaluation mode
        self.net.eval()

        # prepare
        y_data = y_data.to(self.device)
        
        n_samples = y_data.size(0)
        if y_data.dim() == 2 and y_data.size(1) == 2:
            dtype = '2d'
        elif y_data.dim() == 2 and y_data.size(1) == 3:
            dtype = '3d'
        else:
            dtype = 'image'

        title = '%s_%d_steps' % (self.name, step)

        # testing
        if dtype == '2d':
            # plot data samples
            xs = y_data[:, 0].cpu().numpy()
            ys = y_data[:, 1].cpu().numpy()
            py = self.pz(y_data).detach().cpu().numpy() 
            y_image = scatter_plot(xs, ys,colors=py, title=title)
            writer.add_image('2d/data/y', y_image, step, dataformats='HWC')

            if save_image:
                out_file = prefix + 'y_data_{:06d}.jpg'.format(step)
                save_image(out_file, y_image)
                latest_file = prefix + 'y_data_latest.jpg'
                shutil.copyfile(out_file, latest_file)

            # plot latent samples
            z, _ = self.net(y_data)
            pz = self.pz(z)
            z = z.detach().cpu().numpy()
            pz = pz.detach().cpu().numpy()
            
            
            xs = z[:, 0]
            ys = z[:, 1]
            
            z_image = scatter_plot(xs, ys, colors=pz, title=title)
            writer.add_image('2d/train/z', z_image, step, dataformats='HWC')

            if save_image:
                out_file = prefix + 'z_sample_{:06d}.jpg'.format(step)
                save_image(out_file, z_image)
                latest_file = prefix + 'z_sample_latest.jpg'
                shutil.copyfile(out_file, latest_file)

            # save plot
            y, py = self.sample_y(max(1000, n_samples))
            y = y.detach().cpu().numpy()
            py = py.detach().cpu().numpy()
            xs = y[:, 0]
            ys = y[:, 1]

            y_image = scatter_plot(xs, ys, colors=py, title=title)
            writer.add_image('2d/test/y', y_image, step, dataformats='HWC')

            if save_image:
                out_file = prefix + 'y_sample_{:06d}.jpg'.format(step)
                save_image(out_file, y_image)
                latest_file = prefix + 'y_sample_latest.jpg'
                shutil.copyfile(out_file, latest_file)

            # 2D visualization
            map_size = 256
            ix = (np.arange(map_size) + 0.5) / map_size * 2.0 - 1.0
            iy = (np.arange(map_size) + 0.5) / map_size * -2.0 + 1.0

            ix, iy = np.meshgrid(ix, iy)
            ix = ix.reshape((-1))
            iy = iy.reshape((-1))
            y = np.stack([ix, iy], axis=1)
            y = torch.tensor(y, dtype=torch.float32, requires_grad=True)

            py = self.py(y)
            py = py.detach().cpu().numpy()
            py_map = py.reshape((map_size, map_size))

            map_image = image_plot(py_map, title=title)#, extent=[-1, 1, -1, 1])
            writer.add_image('2d/test/map', map_image, step, dataformats='HWC')

            if save_image:
                out_file = prefix + 'y_dist_{:06d}.jpg'.format(step)
                save_image(out_file, map_image)
                latest_file = prefix + 'y_dist_latest.jpg'
                shutil.copyfile(out_file, latest_file)

        if dtype == '3d':
            # plot latent samples
            z, _ = self.net(y_data)
            pz = self.pz(z)
            z = z.detach().cpu().numpy()
            pz = pz.detach().cpu().numpy()
            xs = z[:, 0]
            ys = z[:, 1]
            zs = z[:, 2]

            z_image = scatter_plot(xs, ys, zs, colors=pz, title=title)
            writer.add_image('3d/train/z', z_image, step, dataformats='HWC')

            if save_image:
                out_file = prefix + 'z_sample_{:06d}.jpg'.format(step)
                save_image(out_file, z_image)
                latest_file = prefix + 'z_sample_latest.jpg'
                shutil.copyfile(out_file, latest_file)

            # save plot
            y, py = self.sample_y(max(1000, n_samples))
            y = y.detach().cpu().numpy()
            py = py.detach().cpu().numpy()
            xs = y[:, 0]
            ys = y[:, 1]
            zs = y[:, 2]

            y_image = scatter_plot(xs, ys, zs, colors=py, title=title)
            writer.add_image('3d/test/y', y_image, step, dataformats='HWC')

            if save_image:
                out_file = prefix + 'y_sample_{:06d}.jpg'.format(step)
                save_image(out_file, y_image)
                latest_file = prefix + 'y_sample_latest.jpg'
                shutil.copyfile(out_file, latest_file)

        if dtype == 'image':
            # plot data samples
            images = torch.clamp(y_data.detach().cpu(), 0.0, 1.0)
            grid_image = torchvision.utils.make_grid(images, nrow=8, pad_value=1)
            grid_image = grid_image.permute(1, 2, 0).numpy()
            writer.add_image('image/test/data', grid_image, step, dataformats='HWC')

            if save_image:
                out_file = prefix + 'y_data_{:06d}.jpg'.format(step)
                save_image(out_file, grid_image)
                latest_file = prefix + 'y_data_latest.jpg'
                shutil.copyfile(out_file, latest_file)

            # sample with generative flow
            y, _ = self.sample_y(max(64, n_samples))
            y = y.detach().cpu().numpy()
            images = torch.from_numpy(y[:64])
            images = torch.clamp(images, 0.0, 1.0)
            grid_image = torchvision.utils.make_grid(images, nrow=8, pad_value=1)
            grid_image = grid_image.permute(1, 2, 0).numpy()
            writer.add_image('image/test/sample', grid_image, step, dataformats='HWC')

            if save_image:
                out_file = prefix + 'y_image_{:06d}.jpg'.format(step)
                save_image(out_file, grid_image)
                latest_file = prefix + 'y_image_latest.jpg'
                shutil.copyfile(out_file, latest_file)



def compute_probs(data, n=10):
    h, e = np.histogram(data, n)
    p = h/data.shape[0]
    return e, p

def support_intersection(p, q):
    sup_int = (
        list(
            filter(
                lambda x: (x[0]!=0) & (x[1]!=0), zip(p, q)
            )
        )
    )
    return sup_int

def get_probs(list_of_tuples):
    p = np.array([p[0] for p in list_of_tuples])
    q = np.array([p[1] for p in list_of_tuples])
    return p, q

def kl_divergence(p, q):
    return np.sum(p*np.log(p/q))

def js_divergence(p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*kl_divergence(p, m) + (1./2.)*kl_divergence(q, m)

def compute_kl_divergence(train_sample, test_sample, n_bins=50):
    """
    Computes the KL Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p, q)
    p, q = get_probs(list_of_tuples)

    return kl_divergence(p, q)

def compute_js_divergence(train_sample, test_sample, n_bins=50):
    """
    Computes the JS Divergence using the support 
    intersection between two different samples
    """
    e, p = compute_probs(train_sample, n=n_bins)
    _, q = compute_probs(test_sample, n=e)

    list_of_tuples = support_intersection(p,q)
    p, q = get_probs(list_of_tuples)

    return js_divergence(p, q)


def KLdivergence(x, y):
  """Compute the Kullback-Leibler divergence between two multivariate samples.
  Parameters
  ----------
  x : 2D array (n,d)
    Samples from distribution P, which typically represents the true
    distribution.
  y : 2D array (m,d)
    Samples from distribution Q, which typically represents the approximate
    distribution.
  Returns
  -------
  out : float
    The estimated Kullback-Leibler divergence D(P||Q).
  References
  ----------
  Pérez-Cruz, F. Kullback-Leibler divergence estimation of
continuous distributions IEEE International Symposium on Information
Theory, 2008.
  """
  from scipy.spatial import cKDTree as KDTree

  # Check the dimensions are consistent
  x = np.atleast_2d(x)
  y = np.atleast_2d(y)

  n,d = x.shape
  m,dy = y.shape

  assert(d == dy)


  # Build a KD tree representation of the samples and find the nearest neighbour
  # of each point in x.
  xtree = KDTree(x)
  ytree = KDTree(y)

  # Get the first two nearest neighbours for x, since the closest one is the
  # sample itself.
  r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
  s = ytree.query(x, k=1, eps=.01, p=2)[0]

  # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
  # on the first term of the right hand side.
  return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))

@hydra.main(config_path='configs', config_name='default')
def main(cfg):
    # show parameters
    klds = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ddistrib = 'ggd'
    print('***** parameters ****')
    print(OmegaConf.to_yaml(cfg))
    print('*********************')
    print('')
    ddim = 2
    for loss_type in ['ML','TA']:
        for vprior in ['ggd','mvn','ggd']:
            for vvariable in [True]:
                for vnbeta in [2.]:#2.]:
                    for vdbeta in [0.4]:#,1.2, 2., 2.8,3.6]:
                        gn = []
                        gc.collect()
                        grads, sizes = [], []
                        # dataset
                        if ddistrib != 'ggd':
                            dataset = FlowDataLoader(ddistrib,
                                                     batch_size=cfg.train.samples,
                                                     total_steps=cfg.train.steps,
                                                     shuffle=True)
                        else:
                            dataset = FlowDataLoader(ddistrib,
                                                         batch_size=cfg.train.samples,
                                                         total_steps=cfg.train.steps,
                                                         shuffle=True, beta = vdbeta, dim = ddim)


                        # setup train/eval model
                        if vprior == 'mvn':
                            model = Model(dims=dataset.dims, datatype=dataset.dtype, cfg=cfg, bd_family = 'mvn', variable_bd = vvariable, mu = 0., cov = 1., loss = loss_type)
                        elif vprior == 'ggd':
                            model = Model(dims=dataset.dims, datatype=dataset.dtype, cfg=cfg, bd_family = 'ggd', variable_bd = vvariable, loc = 0., scale = 1., p = vnbeta, dim = ddim, loss = loss_type)
                        elif vprior == 'mvggd':
                            model = Model(dims=dataset.dims, datatype=dataset.dtype, cfg=cfg, bd_family = 'mvggd', variable_bd = vvariable, loc = 0., scale = 1., p = vnbeta, dw=1., dim = ddim, loss = loss_type)

                        # summary writer
                        writer = SummaryWriter('./')

                        # CuDNN backends
                        if cfg.run.debug:
                            torch.backends.cudnn.deterministic = True
                            torch.backends.cudnn.benchmark = False
                            torch.autograd.set_detect_anomaly(True)
                            for submodule in model.net.modules():
                                submodule.register_forward_hook(anomaly_hook)

                        else:
                            torch.backends.cudnn.benchmark = True

                        # resume from checkpoint
                        start_step = 0
                        if cfg.run.ckpt_path is not None:
                            start_step = model.load_ckpt(cfg.run.ckpt_path)

                        # training
                        step = start_step
                        for data in dataset:
                            # training
                            model.train()
                            start_time = time.perf_counter()
                            y = data
                            z, loss = model.train_on_batch(y)

                            #print('------------------------------------------------------------------')                            
                            #grads, sizes = [], []
                            grad = [param.grad.cpu().clone() for param in model.net.parameters() if param.grad is not None]
                            size = 1024
                            grads.append(grad)
                            sizes.append(size)
                            flat_grads = []
                            for grad in grads:
                                flat_grads.append(torch.cat([g.reshape(-1) for g in grad]))
                            full_grads = torch.zeros(flat_grads[-1].shape)
                            # Exact_Grad = torch.zeros(Flat_Grads[-1].shape).cuda()
                            for g, s in zip(flat_grads, sizes):
                                full_grads += g * s
                            full_grads /= np.sum(sizes)
                            gc.collect()
                            flat_grads = torch.stack(flat_grads)
                            sgd_noise = (flat_grads-full_grads).cpu()
                            # Grad_noise = Flat_Grads-Exact_Grad
                            #print('------------------------------------------FG,SGN',flat_grads.shape, sgd_noise.shape)
                            if sgd_noise.sum().item()>0.:
                                gn.append(get_tail_index(sgd_noise))
                                print('*****************',sgd_noise.shape,get_tail_index(sgd_noise))
                                
                            



                            
                            
                            elapsed_time = time.perf_counter() - start_time
                            prefix = 'ddim_' + str(ddim) + '_dbeta_' + str(vdbeta) + '_prior_' + vprior + '_vnoise_' + str(vvariable) + '_nbeta_' + str(vnbeta) + '_loss_' + loss_type + '_'
                            # update for the next step
                            step += 1

                            # reports
                            if step == start_step + 1 or step % (cfg.run.display * 10) == 0:
                                # logging
                                logger.info('[%d/%d] loss=%.5f [%.3f s/it]' %
                                            (step, cfg.train.steps, loss.item(), elapsed_time))

                            if step == start_step + 1 or step % (cfg.run.display * 100) == 0:
                                writer.add_scalar('{:s}/train/loss'.format(dataset.dtype), loss.item(), step)
                                save_files = step % (cfg.run.display * 1000) == 0
                                model.report(writer, torch.FloatTensor(dataset.sample(10000)), step=step, save_files=save_files, prefix=prefix)
                                writer.flush()
                                print(model.net.dp1.detach().cpu().numpy(),model.net.dp2.detach().cpu().numpy(),model.net.dp3.detach().cpu().numpy() if model.net.dp3 is not None else 0,model.net.dp4.detach().cpu().numpy() if model.net.dp4 is not None else 0)

                            if step == start_step + 1 or step % (cfg.run.display * 1000) == 0:
                                # save ckpt

                                ckpt_file = prefix + 'latest.pth'
                                model.save_ckpt(step, ckpt_file)
                        x = torch.FloatTensor(gennorm(beta=vdbeta).rvs(size=[20000,2])).to(device)
                        
                        
                        px = np.mean(np.exp(gennorm(beta=vdbeta).logpdf(x.detach().cpu().numpy())),axis=1)
                        qx = model.log_py(x)
                        #print('PX',px)
                        #print('QX',qx)
                        #print(model.sample_y(20000).cpu().detach())
                        y,_ = model.sample_y(20000)
                        print(get_tail_index(x),get_tail_index(y))
                        klds.append([get_tail_index(x),get_tail_index(y),F.kl_div(qx,torch.FloatTensor(px).to(device)),compute_kl_divergence(x.detach().cpu().numpy(),y.cpu().detach().numpy()),compute_kl_divergence(y.detach().cpu().numpy(),x.cpu().detach().numpy()),compute_js_divergence(x.detach().cpu().numpy(),y.cpu().detach().numpy()),compute_kl_divergence(y.detach().cpu().numpy(),x.cpu().detach().numpy()),KLdivergence(x.detach().cpu().numpy(),y.cpu().detach().numpy()),KLdivergence(y.detach().cpu().numpy(),x.cpu().detach().numpy()),loss_type,vprior,vvariable,vnbeta,vdbeta,gn])
                        #print(klds)
                        pd.DataFrame(klds).to_csv('./kld.csv')



        for vprior in ['ggd','mvn']:#,'mvggd']:
            for vvariable in [False]:
                for vnbeta in [0.4]:#,1.2, 2., 2.8,3.6]:
                    for vdbeta in [0.4,1.2, 2., 2.8,3.6]:
                    # dataset
                        if ddistrib != 'ggd':
                            dataset = FlowDataLoader(ddistrib,
                                                     batch_size=cfg.train.samples,
                                                     total_steps=cfg.train.steps,
                                                     shuffle=True)
                        else:
                            dataset = FlowDataLoader(ddistrib,
                                                         batch_size=cfg.train.samples,
                                                         total_steps=cfg.train.steps,
                                                         shuffle=True, beta = vdbeta, dim = ddim)


                        # setup train/eval model
                        if vprior == 'mvn':
                            model = Model(dims=dataset.dims, datatype=dataset.dtype, cfg=cfg, bd_family = 'mvn', variable_bd = vvariable, mu = 0., cov = 1., loss = loss_type)
                        elif vprior == 'ggd':
                            model = Model(dims=dataset.dims, datatype=dataset.dtype, cfg=cfg, bd_family = 'ggd', variable_bd = vvariable, loc = 0., scale = 1., p = vnbeta, dim = ddim, loss = loss_type)
                        elif vprior == 'mvggd':
                            model = Model(dims=dataset.dims, datatype=dataset.dtype, cfg=cfg, bd_family = 'mvggd', variable_bd = vvariable, loc = 0., scale = 1., p = vnbeta, dw=1., dim = ddim, loss = loss_type)
                        print('ddim_' + str(ddim) + '_dbeta_' + str(vdbeta) + '_prior_' + vprior + '_vnoise_' + str(vvariable) + '_nbeta_' + str(vnbeta) + '_loss_' + loss_type + '_')
                        # summary writer
                        writer = SummaryWriter('./')

                        # CuDNN backends
                        if cfg.run.debug:
                            torch.backends.cudnn.deterministic = True
                            torch.backends.cudnn.benchmark = False
                            torch.autograd.set_detect_anomaly(True)
                            for submodule in model.net.modules():
                                submodule.register_forward_hook(anomaly_hook)

                        else:
                            torch.backends.cudnn.benchmark = True

                        # resume from checkpoint
                        start_step = 0
                        if cfg.run.ckpt_path is not None:
                            start_step = model.load_ckpt(cfg.run.ckpt_path)

                        # training
                        step = start_step
                        for data in dataset:
                            prefix = 'ddim_' + str(ddim) + '_dbeta_' + str(vdbeta) + '_prior_' + vprior + '_vnoise_' + str(vvariable) + '_nbeta_' + str(vnbeta) + '_loss_' + loss_type + '_'
                            # training
                            model.train()
                            start_time = time.perf_counter()
                            y = data
                            z, loss = model.train_on_batch(y)
                            elapsed_time = time.perf_counter() - start_time

                            # update for the next step
                            step += 1

                            # reports
                            if step == start_step + 1 or step % (cfg.run.display * 10) == 0:
                                # logging
                                logger.info('[%d/%d] loss=%.5f [%.3f s/it]' %
                                            (step, cfg.train.steps, loss.item(), elapsed_time))

                            if step == start_step + 1 or step % (cfg.run.display * 100) == 0:
                                writer.add_scalar('{:s}/train/loss'.format(dataset.dtype), loss.item(), step)
                                save_files = step % (cfg.run.display * 1000) == 0
                                model.report(writer, torch.FloatTensor(dataset.sample(10000)), step=step, save_files=save_files, prefix = prefix)
                                writer.flush()
                                print(model.net.dp1.detach().cpu().numpy(),model.net.dp2.detach().cpu().numpy(),model.net.dp3.detach().cpu().numpy() if model.net.dp3 is not None else 0,model.net.dp4.detach().cpu().numpy() if model.net.dp4 is not None else 0)

                            if step == start_step + 1 or step % (cfg.run.display * 1000) == 0:
                                # save ckpt

                                ckpt_file = prefix + 'latest.pth'
                                model.save_ckpt(step, ckpt_file)
                        x = torch.FloatTensor(gennorm(beta=vdbeta).rvs(size=[20000,2])).to(device)
                        px = np.mean(np.exp(gennorm(beta=vdbeta).logpdf(x.detach().cpu().numpy())),axis=1)
                        qx = model.log_py(x)
                        print('PX',px)
                        print('QX',qx)
                        #print(model.sample_y(20000).cpu().detach())
                        y,_ = model.sample_y(20000)
                        klds.append([F.kl_div(qx,torch.FloatTensor(px).to(device)),compute_kl_divergence(x.detach().cpu().numpy(),y.cpu().detach().numpy()),compute_kl_divergence(y.detach().cpu().numpy(),x.cpu().detach().numpy()),compute_js_divergence(x.detach().cpu().numpy(),y.cpu().detach().numpy()),compute_kl_divergence(y.detach().cpu().numpy(),x.cpu().detach().numpy()),KLdivergence(x.detach().cpu().numpy(),y.cpu().detach().numpy()),KLdivergence(y.detach().cpu().numpy(),x.cpu().detach().numpy()),loss_type,vprior,vvariable,vnbeta,vdbeta])
                        print(klds)
                        pd.DataFrame(klds).to_csv('./kld.csv')
                        



    pd.DataFrame(klds).to_csv('./kld.csv')


if __name__ == '__main__':
    main()
