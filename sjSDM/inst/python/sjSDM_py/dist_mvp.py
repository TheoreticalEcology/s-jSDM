import torch
import pyro
from torch.distributions import constraints
from torch.distributions.utils import lazy_property
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape

class MultivariateProbit(TorchDistribution):
    arg_constraints = {'loc': constraints.real_vector}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale=None, link="logit", validate_args=None):
        if loc.dim() < 1:
            raise ValueError("loc must be at least one-dimensional.")

        loc_ = loc.unsqueeze(-1)  # temporarily add dim on right

        if scale.dim() < 2:
            raise ValueError("scale must be at least two-dimensional, "
                             "with optional leading batch dimensions")
        self.scale, loc_ = torch.broadcast_tensors(scale, loc_)
        self.scale = scale
        self._covariance_matrix = scale
        self.loc = loc_[..., 0]  # drop rightmost dim
        self.df = scale.shape[-1]
        
        if link == "logit":
            self.link = lambda value: torch.sigmoid(value)
        elif link == "probit":
            self.link = lambda value: torch.distributions.Normal(0.0, 1.0).cdf(value)
        elif link == "linear":
            self.link = lambda value: torch.clamp(value, 0.0, 1.0)

        batch_shape, event_shape = self.loc.shape[:-1], self.loc.shape[-1:]
        super(MultivariateProbit, self).__init__(batch_shape, event_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(MultivariateProbit, _instance)
        batch_shape = torch.Size(batch_shape)
        loc_shape = batch_shape + self.event_shape
        cov_shape = batch_shape + self.event_shape + torch.Size([self.df])
        new.loc = self.loc.expand(loc_shape)
        if 'scale' in self.__dict__:
            new.scale = self.scale #.expand(cov_shape)
            new.df = self.df
            new._covariance_matrix = self.scale
            new.link = self.link
        super(MultivariateProbit, new).__init__(batch_shape,
                                                self.event_shape,
                                                validate_args=False)
        new._validate_args = self._validate_args
        return new

    @lazy_property
    def covariance_matrix(self):
        return (torch.matmul(self._covariance_matrix,
                             self._covariance_matrix.transpose(-1, -2))
                .expand(self._batch_shape + self._event_shape + self._event_shape))

    @property
    def mean(self):
        return self.loc

    def rsample(self, sample_shape=torch.Size()):
        # shape = self._extended_shape(sample_shape)
        # eps = _standard_normal(shape, dtype=self.loc.dtype, device=self.loc.device)
        # return self.loc + _batch_mv(self._unbroadcasted_scale_tril, eps)
        if sample_shape == torch.Size():
            sample_shape = [1]
        shape = self._extended_shape(sample_shape)
        shape2 = torch.tensor(shape[:-1]).prod().view([1])
        
        eps = torch.tensor(0.00001)
        one = torch.tensor(1.0)
        alpha = torch.tensor(1.7012)
        half = torch.tensor(0.5)
        noise = torch.randn(size = torch.Size([100, shape2, self.df]))
        E = self.link(torch.tensordot(noise, self.scale.t(), dims = 1).add(self.loc).mul(alpha)).mul(one.sub(eps)).add(eps.mul(half)).mean(dim = 0)
        return E.view(shape)
    
    def log_prob(self, value):
        #if self._validate_args:
        #    self._validate_sample(value)
        
        #print(self.scale)
        eps = torch.tensor(0.00001)
        zero = torch.tensor(0.0)
        one = torch.tensor(1.0)
        alpha = torch.tensor(1.7012)
        half = torch.tensor(0.5)
        
        shape = value.shape
        shape2 = torch.tensor(shape[:-1]).prod().view([1])
        value = value.view(torch.Size([shape2, shape[-1]]))
        
        #print(self.scale.shape)
        
        noise = torch.randn(size = [100, value.shape[0], self.df])
        E = self.link(torch.tensordot(noise, self.scale.t(), dims = 1).add(self.loc).mul(alpha)).mul(one.sub(eps)).add(eps.mul(half))
        logprob = E.log().mul(value).add(one.sub(E).log().mul(one.sub(value))).neg().sum(dim = 2).neg()
        maxlogprob = logprob.max(dim = 0).values
        Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
        loss = Eprob.log().neg().sub(maxlogprob).neg()
        return loss.view(shape[:-1])