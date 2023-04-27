import torch
import pyro
from torch.distributions import constraints
from torch.distributions.utils import lazy_property
from pyro.distributions.torch_distribution import TorchDistribution
from pyro.distributions.util import broadcast_shape
import warnings 

warnings.filterwarnings("ignore")

class MultivariateProbit(TorchDistribution):
    arg_constraints = {'loc': constraints.real_vector}
    support = constraints.real
    has_rsample = True

    def __init__(self, loc, scale=None, link="probit",sampling=5000, validate_args=None):
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
        self.sampling = sampling
        self.alpha = 1.0
        
        if link == "logit":
            self.link = lambda value: torch.sigmoid(value)
            self.alpha = 1.7012
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
            new.sampling = self.sampling
            new.alpha = self.alpha
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
        alpha = torch.tensor(self.alpha)
        half = torch.tensor(0.5)
        noise = torch.randn(size = torch.Size([100, shape2, self.df]))
        E = self.link(torch.tensordot(noise, self.scale.t(), dims = 1).add(self.loc).mul(alpha)).mul(one.sub(eps)).add(eps.mul(half)).mean(dim = 0)
        return E.view(shape)
    
    def log_prob(self, value):
        #if self._validate_args:
        #    self._validate_sample(value)
        
        #print(self.scale)
        eps = torch.tensor(0.00001, device=value.device)
        zero = torch.tensor(0.0, device=value.device)
        one = torch.tensor(1.0, device=value.device)
        alpha = torch.tensor(self.alpha, device=value.device)
        half = torch.tensor(0.5, device=value.device)
        
        shape = value.shape
        shape2 = torch.tensor(shape[:-1]).prod().view([1])
        value = value.view(torch.Size([shape2, shape[-1]]))
        
        #print(self.scale.shape)
        
        noise = torch.randn(size = [self.sampling, value.shape[0], self.df], device=value.device)
        E = self.link(torch.tensordot(noise, self.scale.t(), dims = 1).add(self.loc).mul(alpha)).mul(one.sub(eps)).add(eps.mul(half))
        logprob = E.log().mul(value).add(one.sub(E).log().mul(one.sub(value))).neg().sum(dim = 2).neg()
        maxlogprob = logprob.max(dim = 0).values
        Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
        loss = Eprob.log().neg().sub(maxlogprob).neg()
        return loss.view(shape[:-1])


def MVP_logLik(Y, X, sigma, device, dtype, batch_size=25, alpha = 1.0, sampling=1000, link="probit", individual=False, theta = None):

    torch.set_default_tensor_type('torch.FloatTensor')
    
    if dtype is not None:
        dtype = torch.float32

    if device.type == 'cuda' and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        pin_memory = False
        device = device.type+ ":" + str(device.index)
    else:
        pin_memory = True

    if link=="probit":
        link_func = lambda value: torch.sigmoid(value)
    elif link=="linear":
        link_func = lambda value: torch.clamp(value, 0.0, 1.0)
    elif link=="logit":
        link_func = lambda value: torch.sigmoid(value)
    elif link=="count":
        link_func = lambda value: torch.exp(value)
    elif link=="nbinom":
        link_func = lambda value: torch.exp(value)
    
    if theta is not None:
        theta = torch.tensor(theta, dtype=dtype, device=torch.device(device))

    data = torch.utils.data.TensorDataset(torch.tensor(Y, dtype=dtype, device=torch.device('cpu')), torch.tensor(X, dtype=dtype, device=torch.device('cpu')))
    DataLoader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin_memory, drop_last=False)
    torch.cuda.empty_cache()
    sigma=torch.tensor(sigma, dtype=dtype, device=torch.device(device))
    logLik = []
    for step, (y, pred) in enumerate(DataLoader):
        y = y.to(device, non_blocking=True)
        pred = pred.to(device, non_blocking=True)
        noise = torch.randn(size = [sampling, pred.shape[0], sigma.shape[1]], device=torch.device(device), dtype=dtype)
        E = link_func( torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(pred).mul(alpha) ).mul(0.999999).add(0.0000005)
        if link in ["probit", "linear", "logit"] : 
            logprob = E.log().mul(y).add((1.0 - E).log().mul(1.0 - y)).neg().sum(dim = 2).neg()
        elif link == "count":
            logprob = torch.distributions.Poisson(rate=E).log_prob(y).sum(2)
        elif link == "nbinom":
            eps = 0.0001
            theta_tmp = 1.0/(torch.nn.functional.softplus(theta)+eps)
            probs = torch.clamp((1.0 - theta_tmp/(theta_tmp+E)) + eps, 0.0, 1.0-eps)
            logprob = torch.distributions.NegativeBinomial(total_count=theta_tmp, probs=probs).log_prob(y).sum(2)
        maxlogprob = logprob.max(dim = 0).values
        Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
        loss = Eprob.log().neg().sub(maxlogprob)
        logLik.append(loss.reshape([pred.shape[0], 1]).data)
    if individual is not True:
        logLik = torch.cat(logLik).sum().data.cpu().numpy()
    else:
        logLik = torch.cat(logLik).data.cpu().numpy()    
    return logLik