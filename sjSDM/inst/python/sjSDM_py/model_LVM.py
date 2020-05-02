import torch
import numpy as np
import pyro
import sys

class Model_LVM():
    def __init__(self, device="cpu", dtype="float32"):
        device, dtype = self._device_and_dtype(device, dtype)
        self.device = device
        self.dtype = dtype
        self.pyro = pyro
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
    
    @staticmethod
    def _device_and_dtype(device, dtype):
        """define device and dtype

        parse device and dtype into torch.dtype and torch.device

        # Arguments
            device: str or torch.device, "cpu" or "gpu"
            dtype: str or torch.dtype, "float32" and "float64" are supported

        # Example

        ```python
            device, dtype = _device_and_dtype(torch.device("cpu"), "float32")
            device, dtype = _device_and_dtype("gpu", "float32")
        ````

        """
        if type(device) is int:
            device = torch.device('cuda:'+str(device))

        if type(device) is str:
            device = torch.device(device)

        if type(dtype) is not str:
            return device, dtype
        else:
            if dtype == "float32":
                return device, torch.float32
            if dtype == "float64":
                return device, torch.float64
        return device, torch.float32
    
    def _get_DataLoader(self, X, Y=None,SP=None,RE=None, batch_size=25, shuffle=True, parallel=0, drop_last=True):
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
            pin_memory = False
        else:
            pin_memory = True
        #init_func = lambda: torch.multiprocessing.set_start_method('spawn', True)
        if type(RE) is np.ndarray:
            if type(Y) is np.ndarray:
                if type(SP) is np.ndarray:
                    data = torch.utils.data.TensorDataset(torch.tensor(X.copy(), dtype=torch.float32, device=torch.device('cpu')), 
                                                          torch.tensor(Y.copy(), dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(SP, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.asarray(RE).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')),
                                                          torch.tensor(np.arange(0, X.shape[0]).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))
                else:
                    data = torch.utils.data.TensorDataset(torch.tensor(X.copy(), dtype=torch.float32, device=torch.device('cpu')), 
                                                          torch.tensor(Y.copy(), dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.asarray(RE).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')),
                                                          torch.tensor(np.arange(0, X.shape[0]).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))

            else:
                if type(SP) is np.ndarray:
                    data = torch.utils.data.TensorDataset(torch.tensor(X.copy(), dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(SP, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.asarray(RE).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')),
                                                          torch.tensor(np.arange(0, X.shape[0]).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))
                else: 
                    data = torch.utils.data.TensorDataset(torch.tensor(X.copy(), dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.asarray(RE).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')),
                                                          torch.tensor(np.arange(0, X.shape[0]).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))
        
        else: 
            if type(Y) is np.ndarray:
                if type(SP) is np.ndarray:
                    data = torch.utils.data.TensorDataset(torch.tensor(X.copy(), dtype=torch.float32, device=torch.device('cpu')), 
                                                        torch.tensor(Y.copy(), dtype=torch.float32, device=torch.device('cpu')),
                                                        torch.tensor(SP, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.arange(0, X.shape[0]).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))
                else:
                    data = torch.utils.data.TensorDataset(torch.tensor(X.copy(), dtype=torch.float32, device=torch.device('cpu')), 
                                                        torch.tensor(Y.copy(), dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.arange(0, X.shape[0]).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))

            else:
                if type(SP) is np.ndarray:
                    data = torch.utils.data.TensorDataset(torch.tensor(X.copy(), dtype=torch.float32, device=torch.device('cpu')),
                                                        torch.tensor(SP, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.arange(0, X.shape[0]).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))
                else: 
                    data = torch.utils.data.TensorDataset(torch.tensor(X.copy(), dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.arange(0, X.shape[0]).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))          

        DataLoader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=int(parallel), pin_memory=pin_memory, drop_last=drop_last)
        torch.cuda.empty_cache()
        return DataLoader
    
    def build(self, input_shape, out, scale_mu = 5.0, scale_lf = 1.0,scale_lv=1.0,  df=2,
              guide='LaplaceApproximation', 
              family = "binomial", link = "logit"):

        self.create_link(family, link)
            
        #self.link = link
        torch.cuda.empty_cache()
        pyro.enable_validation(True)
        pyro.clear_param_store()
        #XX = torch.tensor(X, dtype = torch.float32)
        #YY = torch.tensor(Y, dtype = torch.float32)
        self.e = input_shape[1]
        self.sp = out
        self.n = input_shape[0]
        self.df = df
        e = self.e
        sp = self.sp
        n = self.n

        def model(XX, YY=None, indices=None, posterior = self.posterior):
            mu_scale = torch.ones([e, sp])*scale_mu
            mu_loc = torch.zeros([e, sp])
            lv_scale2 = torch.ones([n, df])*scale_lv
            lv_loc = torch.zeros([n, df])
            lf_scale2 = torch.ones([df, sp])*scale_lf
            lf_loc = torch.zeros([df, sp])
            mu = pyro.sample("mu", pyro.distributions.Normal(mu_loc, mu_scale).to_event())
            lf = pyro.sample("lf", pyro.distributions.Normal(lf_loc, lf_scale2).to_event())
            lv = pyro.sample("lv", pyro.distributions.Normal(lv_loc, lv_scale2).to_event())
            with pyro.plate('data', size = XX.shape[0]):
                loc_tmp = XX.matmul(mu).add( lv.index_select(0, indices).matmul(lf) )
                posterior(loc_tmp, YY)
        self.model = model

        guide2 = guide
        self.elbo = pyro.infer.JitTrace_ELBO(ignore_jit_warnings=True)
        if guide2 == 'LaplaceApproximation':
            self.guide = pyro.infer.autoguide.AutoLaplaceApproximation(self.model)

        if guide2 == 'LowRankMultivariateNormal':
            self.guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(self.model, init_loc_fn=pyro.infer.autoguide.init_to_mean)
            self.elbo = pyro.infer.Trace_ELBO(ignore_jit_warnings=True)

        if guide2 == "Delta":
            self.guide = pyro.infer.autoguide.AutoDelta(self.model)

        if guide2 == 'DiagonalNormal':
            self.guide = pyro.infer.autoguide.AutoDiagonalNormal(self.model)
    


    def create_link(self, family="binomial", link="logit"):
        if family == "binomial":
            if link == "logit":
                self.link = lambda value: torch.sigmoid(value)
                self.posterior = lambda loc, YY: pyro.sample("obs", pyro.distributions.Binomial(1, probs = torch.sigmoid(loc)).to_event(), obs = YY)
                self.logLik = lambda loc, YY: pyro.distributions.Binomial(1, probs = torch.sigmoid(loc)).log_prob(YY)
            if link == "probit":
                normal = torch.distributions.Normal(0.0, 1.0)
                self.link = lambda value: normal.cdf(value)
                self.posterior = lambda loc, YY: pyro.sample("obs", pyro.distributions.Binomial(1, probs = normal.cdf(loc)).to_event(), obs = YY)
                self.logLik = lambda loc, YY:  pyro.distributions.Binomial(1, probs = normal.cdf(loc)).log_prob(YY)
        if family == "poisson":
            if link == "log":
                self.link = lambda value: value.exp()
                self.posterior = lambda loc, YY: pyro.sample("obs", pyro.distributions.Poisson(rate= loc.exp()).to_event(), obs = YY)
                self.logLik = lambda loc, YY: pyro.distributions.Poisson(rate= loc.exp()).log_prob(YY)
            if link == "linear":
                self.link = lambda value: torch.clamp(value, min=0.00000001)
                self.posterior = lambda loc, YY: pyro.sample("obs", pyro.distributions.Poisson(rate= torch.clamp(loc, min=0.00000001)).to_event(), obs = YY)
                self.logLik = lambda loc, YY: pyro.distributions.Poisson(rate= torch.clamp(loc, min=0.00000001)).log_prob(YY)

    def fit(self, X, Y,
            lr = [0.1], epochs = 200,
            batch_size = 25,
            num_samples=100,
            parallel = 0):
        
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

 
        
        if len(lr) is 1:
             adam = pyro.optim.Adam({'lr' : lr[0]})
        else:
            def per_param_callable(module_name, param_name):
                if param_name == 'mu':
                    return {"lr": lr[0]}
                if param_name == 'lf':
                    return {"lr": lr[1]}
                else:
                    return {"lr": lr[2]}
            adam = pyro.optim.Adam(per_param_callable)

        
        self.svi = pyro.infer.SVI(self.model, self.guide, adam, loss = self.elbo)
        stepSize = np.floor(X.shape[0] / batch_size).astype(int)
        dataLoader = self._get_DataLoader(X, Y, batch_size = batch_size, shuffle=True, parallel=parallel)
        batch_loss = np.zeros(stepSize)
        self.history = np.zeros(epochs)
        for epoch in range(epochs):
            for step, (x, y, ind) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                ind = ind.to(self.device, non_blocking=True).view([-1])
                loss = self.svi.step(x, y, ind)
                batch_loss[step] = loss
            bl = np.mean(batch_loss)
            _ = sys.stdout.write("\rEpoch: {}/{} loss: {} ".format(epoch+1,epochs, np.round(bl, 3).astype(str)))
            sys.stdout.flush()
        self.posterior_samples = pyro.infer.Predictive(self.model, guide=self.guide, num_samples=num_samples)(torch.tensor(X.copy(), dtype=torch.float32), 
                                                                                                              torch.tensor(Y.copy(), dtype=torch.float32),
                                                                                                              torch.tensor(np.arange(0, X.shape[0]), dtype=torch.long))

        
        self.lf = self.posterior_samples["lf"].data.squeeze(dim=1).mean(dim=0)
        self.lv = self.posterior_samples["lv"].data.squeeze(dim=1).mean(dim=0)
        self.mu = self.posterior_samples["mu"].data.squeeze(dim=1).mean(dim=0)

    def getLogLik(self, X, Y, batch_size = 25, parallel=0):
        dataLoader = self._get_DataLoader(X = X, Y = Y, batch_size = batch_size, shuffle = False, parallel = parallel, drop_last = False)
        ll = 0
        for step, (x, y, ind) in enumerate(dataLoader):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            ind = ind.to(self.device, non_blocking=True).view([-1])
            lin = self.link(x.matmul(self.mu).add( self.lv.index_select(0, ind).matmul(self.lf) ))
            ll += self.logLik(lin, y).sum().data.cpu().numpy()
        return ll
    
    def predict(self, newdata, batch_size = 25, parallel=0, mean_field=True ):
        dataLoader = self._get_DataLoader(X = newdata, Y = None, batch_size = batch_size, shuffle = False, parallel = parallel, drop_last = False)
        pred = []

        if mean_field:
            for step, (x, ind) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                lin = self.link(x.matmul(self.mu))
                pred.append(lin)
            return torch.cat(pred, dim=0).data.cpu().numpy()
        else:
            mu = self.posterior_samples["mu"].squeeze(dim=1)
            for step, (x, ind) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                lin = self.link( x.expand([mu.shape[0],batch_size,newdata.shape[1]]).matmul(mu) )
                pred.append(lin)
            return torch.cat(pred,dim=1).data.cpu().numpy()

    def predictPosterior(self, X, batch_size = 25, parallel=0, num_samples=10):
        dataLoader = self._get_DataLoader(X = X, batch_size = batch_size, shuffle = False, parallel = parallel, drop_last = False)
        pred = []
        mu = self.posterior_samples["mu"].squeeze(dim=1)
        lf = self.posterior_samples["lf"].squeeze(dim=1)
        lv = self.posterior_samples["lv"].squeeze(dim=1)
        for step, (x, ind) in enumerate(dataLoader):
            x = x.to(self.device, non_blocking=True)
            ind = ind.to(self.device, non_blocking=True).view([-1])
            lin = self.link( x.expand([mu.shape[0],batch_size,X.shape[1]]).matmul(mu).add( lv.index_select(1, ind).matmul(lf) ) )
            pred.append(lin)
        return torch.cat(pred,dim=1).data.cpu().numpy()

    @property
    def covariance(self):
        return self.lf.t().matmul(self.lf).data.cpu().numpy()

    @property
    def weights(self):
        return self.mu.data.cpu().numpy()

    @property
    def lfs(self):
        return self.lf.data.cpu().numpy()

    @property
    def lvs(self):
        return self.lv.data.cpu().numpy()

    def set_posterior_samples(self, ps):
        self.posterior_samples = ps