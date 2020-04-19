from .dist_mvp import MultivariateProbit
import torch
import numpy as np
import pyro
import logging
import sys
logging.basicConfig(format='%(message)s', level=logging.INFO)

class Model_SVI():
    def __init__(self, device="cpu", dtype="float32"):
        device, dtype = self._device_and_dtype(device, dtype)
        self.device = device
        self.dtype = dtype
    
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
                    data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=torch.device('cpu')), 
                                                          torch.tensor(Y, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(SP, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.asarray(RE).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))
                else:
                    data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=torch.device('cpu')), 
                                                          torch.tensor(Y, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.asarray(RE).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))

            else:
                if type(SP) is np.ndarray:
                    data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(SP, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.asarray(RE).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))
                else: 
                    data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=torch.device('cpu')),
                                                          torch.tensor(np.asarray(RE).reshape([-1,1]), dtype=torch.long, device=torch.device('cpu')))
        
        else: 
            if type(Y) is np.ndarray:
                if type(SP) is np.ndarray:
                    data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=torch.device('cpu')), 
                                                        torch.tensor(Y, dtype=torch.float32, device=torch.device('cpu')),
                                                        torch.tensor(SP, dtype=torch.float32, device=torch.device('cpu')))
                else:
                    data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=torch.device('cpu')), 
                                                        torch.tensor(Y, dtype=torch.float32, device=torch.device('cpu')))

            else:
                if type(SP) is np.ndarray:
                    data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=torch.device('cpu')),
                                                        torch.tensor(SP, dtype=torch.float32, device=torch.device('cpu')))
                else: 
                    data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32, device=torch.device('cpu')))            

        DataLoader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=int(parallel), pin_memory=pin_memory, drop_last=drop_last)
        torch.cuda.empty_cache()
        return DataLoader
    
    def build_model(self, X, Y,Re,SP, df, scale_mu = 5.0, scale_scale = 1.0, batch_size = 20):
        pyro.enable_validation(True)
        pyro.clear_param_store()
        #XX = torch.tensor(X, dtype = torch.float32)
        #YY = torch.tensor(Y, dtype = torch.float32)
        self.e = X.shape[1]
        self.sp = Y.shape[1]
        self.df = df
        e = self.e
        sp = self.sp

        if Re is None:
            def model(XX, YY=None, link = self.link):
                mu_scale = torch.ones([e, sp])
                mu_loc = torch.zeros([e, sp])
                scale_scale2 = torch.ones([sp, df])
                scale_loc = torch.zeros([sp, df])
                mu = pyro.sample("mu", pyro.distributions.Normal(mu_loc, mu_scale).to_event())
                scale = pyro.sample("scale", pyro.distributions.Normal(scale_loc, scale_scale2).to_event())
                with pyro.plate('data', size = XX.shape[0]):
                    loc_tmp = XX.matmul(mu)
                    #loc_tmp = mu(XX[ind,:])
                    pyro.sample("obs", MultivariateProbit(loc_tmp, scale), obs = YY)
        else:
            len_unique = np.unique(Re[:,0]).shape[0]
            #indices = torch.tensor(Re, dtype=torch.long)
            def model(XX, YY=None, Re=None, link = self.link):
                mu_scale = torch.ones([e, sp])*scale_mu
                mu_loc = torch.zeros([e, sp])
                scale_scale2 = torch.ones([sp, df])*scale_scale
                scale_loc = torch.zeros([sp, df])
                mu = pyro.sample("mu", pyro.distributions.Normal(mu_loc, mu_scale).to_event())
                scale = pyro.sample("scale", pyro.distributions.Normal(scale_loc, scale_scale2).to_event())
                re = pyro.sample("re", pyro.distributions.Normal(0.0, torch.ones([len_unique,1])).to_event())
                with pyro.plate('data', size = XX.shape[0]):
                    if Re is None:
                        loc_tmp = XX.matmul(mu)
                    else:
                        loc_tmp = XX.matmul(mu) + re.gather(0, Re)
                    pyro.sample("obs", MultivariateProbit(loc_tmp, scale), obs = YY)
        
        return model

    def get_guide(self):
        df = self.df
        e = self.e
        sp = self.sp
        if Re is None:
            def guide(XX, YY):
                mu_loc_0 = pyro.param("mu_loc_0", torch.zeros([e, sp]))
                mu_scale_0 = pyro.param("mu_scale_0", torch.ones([e, sp])*7.0, constraint=pyro.distributions.constraints.positive)
                scale_loc_0 = pyro.param("scale_loc_0", torch.zeros([sp, df]))
                scale_scale_0 = pyro.param("scale_scale_0", torch.ones([sp, df])*1.0, constraint=pyro.distributions.constraints.positive)
                mu = pyro.sample("mu", pyro.distributions.Normal(mu_loc_0, mu_scale_0).to_event()) #, infer = baseline_dict)
                scale = pyro.sample("scale", pyro.distributions.Normal(scale_loc_0, scale_scale_0).to_event()) #, infer = baseline_dict)

            return guide
        else:
            def guide(XX, YY, Re):
                mu_loc_0 = pyro.param("mu_loc_0", torch.zeros([e, sp]))
                mu_scale_0 = pyro.param("mu_scale_0", torch.ones([e, sp])*7.0, constraint=pyro.distributions.constraints.positive)
                scale_loc_0 = pyro.param("scale_loc_0", torch.zeros([sp, df]))
                scale_scale_0 = pyro.param("scale_scale_0", torch.ones([sp, df])*1.0, constraint=pyro.distributions.constraints.positive)
                mu = pyro.sample("mu", pyro.distributions.Normal(mu_loc_0, mu_scale_0).to_event()) #, infer = baseline_dict)
                scale = pyro.sample("scale", pyro.distributions.Normal(scale_loc_0, scale_scale_0).to_event()) #, infer = baseline_dict)

            return guide

    def fit(self, X, Y, df,RE=None, SP=None,
            guide='LaplaceApproximation', 
            scale_mu = 5.0, scale_scale = 0.01, 
            lr = 0.001, epochs = 200,
            batch_size = 25,
            num_samples=100,
            parallel = 0,
            link = "logit"):
        
        if self.device.type == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
            

        self.link = link
        torch.cuda.empty_cache()
        guide2 = guide
        self.model = self.build_model(X, Y,RE,SP, df, scale_mu, scale_scale, batch_size)
        elbo = pyro.infer.JitTrace_ELBO(ignore_jit_warnings=True)
        if guide2 == 'LaplaceApproximation':
            self.guide = pyro.infer.autoguide.AutoLaplaceApproximation(self.model)

        if guide2 == 'LowRankMultivariateNormal':
            #self.guide = pyro.infer.autoguide.AutoLowRankMultivariateNormal(self.model, init_loc_fn=pyro.infer.autoguide.init_to_mean)
            self.guide = pyro.infer.autoguide.AutoGuideList(self.model)
            self.guide.add(pyro.infer.autoguide.AutoLowRankMultivariateNormal(pyro.poutine.block(self.model, expose=["mu"]), init_loc_fn=pyro.infer.autoguide.init_to_mean, init_scale=scale_mu))
            self.guide.add(pyro.infer.autoguide.AutoLowRankMultivariateNormal(pyro.poutine.block(self.model, expose=["scale"]),  init_loc_fn=pyro.infer.autoguide.init_to_mean, init_scale=scale_scale))
            elbo = pyro.infer.Trace_ELBO(ignore_jit_warnings=True)
            if type(RE) is np.ndarray:
                self.guide.add(pyro.infer.autoguide.AutoLowRankMultivariateNormal(pyro.poutine.block(self.model, expose=["re"]), init_loc_fn=pyro.infer.autoguide.init_to_mean, init_scale=1.0))

        if guide2 == "Delta":
            self.guide = pyro.infer.autoguide.AutoDelta(self.model, init_loc_fn=pyro.infer.autoguide.init_to_mean)

        if guide2 == 'DiagonalNormal':
            self.guide = pyro.infer.autoguide.AutoGuideList(self.model)
            self.guide.add(pyro.infer.autoguide.AutoDiagonalNormal(pyro.poutine.block(self.model, expose=["mu"]), init_loc_fn=pyro.infer.autoguide.init_to_mean, init_scale=scale_mu ))
            self.guide.add(pyro.infer.autoguide.AutoDiagonalNormal(pyro.poutine.block(self.model, expose=["scale"]),  init_loc_fn=pyro.infer.autoguide.init_to_mean, init_scale=scale_scale))
            if type(RE) is np.ndarray:
                self.guide.add(pyro.infer.autoguide.AutoDiagonalNormal(pyro.poutine.block(self.model, expose=["re"]), init_loc_fn=pyro.infer.autoguide.init_to_mean, init_scale=1.0))


        if guide2 == 'manually':
            self.guide = self.get_guide()
        
        if len(lr) is 1:
             adam = pyro.optim.Adam({'lr' : lr[0]})
        else:
            def per_param_callable(module_name, param_name):
                if param_name == 'mu':
                    return {"lr": lr[0]}
                else:
                    return {"lr": lr[1]}
            adam = pyro.optim.Adam(per_param_callable)
        #elbo = pyro.infer.Trace_ELBO(ignore_jit_warnings=True)
        #guide()
        #XX = torch.tensor(X, dtype=torch.float32)
        #YY = torch.tensor(Y, dtype=torch.float32)
        
        self.svi = pyro.infer.SVI(self.model, self.guide, adam, loss = elbo)
        stepSize = np.floor(X.shape[0] / batch_size).astype(int)
        dataLoader = self._get_DataLoader(X, Y, SP, RE, batch_size, True, parallel)
        batch_loss = np.zeros(stepSize)
        self.history = np.zeros(epochs)
        if RE is None:          
            for epoch in range(epochs):
                for step, (x, y) in enumerate(dataLoader):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    loss = self.svi.step(x, y)
                    batch_loss[step] = loss
                bl = np.mean(batch_loss)
                _ = sys.stdout.write("\rEpoch: {}/{} loss: {} ".format(epoch+1,epochs, np.round(bl, 3).astype(str)))
                sys.stdout.flush()
            self.posterior_samples = pyro.infer.Predictive(self.model, guide=self.guide, num_samples=num_samples)(torch.tensor(X, dtype=torch.float32), 
                                                                                                                  torch.tensor(Y, dtype=torch.float32))

        else:
            for epoch in range(epochs):
                for step, (x, y, re) in enumerate(dataLoader):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    re=re.to(self.device, non_blocking=True)
                    #print(re.dtype)
                    loss = self.svi.step(x, y, re)
                    batch_loss[step] = loss
                bl = np.mean(batch_loss)
                _ = sys.stdout.write("\rEpoch: {}/{} loss: {} ".format(epoch+1,epochs, np.round(bl, 3).astype(str)))
                sys.stdout.flush()
            self.posterior_samples = pyro.infer.Predictive(self.model, guide=self.guide, num_samples=num_samples)(torch.tensor(X, dtype=torch.float32), 
                                                                                                                  torch.tensor(Y, dtype=torch.float32),
                                                                                                                  torch.tensor(np.asarray(RE).reshape([-1,1]), dtype=torch.long))

        
        self.scale = self.posterior_samples["scale"].data.squeeze().mean(dim=0).cpu()
        self.mu = self.posterior_samples["mu"].data.squeeze().mean(dim=0).cpu()
        
        if RE is None:
            self.Re = None
        else:
            self.Re = self.posterior_samples["re"].data.squeeze().mean(dim=0).cpu()
    
    def predict(self, newdata, SP=None, RE=None, batch_size = 25, parallel=0, num_samples=10 ):
        dataLoader = self._get_DataLoader(X = newdata, Y = None, SP=SP,RE=RE, batch_size = batch_size, shuffle = False, parallel = parallel, drop_last = False)
        pred = []
        if RE is None:                          
            for step, (x) in enumerate(dataLoader):
                x = x[0].to(self.device, non_blocking=True)
                pred.append(pyro.infer.Predictive(self.model, guide=self.guide, num_samples=num_samples)(XX = x,YY= None)["obs"].squeeze())
        else:
            for step, (x, re) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                re = re.to(self.device, non_blocking=True)
                pred.append(pyro.infer.Predictive(self.model, guide=self.guide, num_samples=num_samples)(XX = x,YY= None, RE=re)["obs"].squeeze())  
        return torch.cat(pred, dim=1).data.cpu().numpy()
    
    def logLik(self, X, Y, RE=None, SP=None, batch_size = 25, parallel=0):
        if self.device.type == 'cuda':
            device = self.device.type+ ":" + str(self.device.index)
        else:
            device = 'cpu'

        ww = torch.tensor(self.weights)
        logLik = 0
        dataLoader = self._get_DataLoader(X = X, Y = Y, SP=SP,RE=RE, batch_size = batch_size, shuffle = False, parallel = parallel, drop_last = False)
        pred = []
        if RE is None:                          
            for step, (x, y) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                mu = torch.nn.functional.linear(x, ww.t())
                logLik += MultivariateProbit(mu, self.scale).log_prob(y).sum().data.cpu().numpy()
                
        else:
            for step, (x, y, re) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                re = re.to(self.device, non_blocking=True)
                spatial_re = self.re.gather(0, re.to(self.device, non_blocking=True))
                y = y.to(self.device, non_blocking=True)
                mu = torch.nn.functional.linear(x, ww.t()) + spatial_re
                logLik += MultivariateProbit(mu, self.scale).log_prob(y).sum().data.cpu().numpy()
        return logLik  

    @property
    def covariance(self):
        return self.scale.matmul(self.scale.t()).data.cpu().numpy()

    @property
    def weights(self):
        return self.mu.data.cpu().numpy()

    @property
    def re(self):
        if self.Re is None:
            return None
        else:
            return self.Re.data.cpu().numpy()
