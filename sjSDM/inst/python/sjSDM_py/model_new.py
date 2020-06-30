import numpy as np
import torch
import itertools
from tqdm import tqdm
from torch import nn, optim
import sys


class Model_sjSDM:
    def __init__(self,alpha = 1.70169, device="cpu", dtype="float32"):
        self.params = []
        self.losses = []
        self.env = None
        self.spatial = None
        device, dtype = self._device_and_dtype(device, dtype)
        self.device = device
        self.dtype = dtype
        self.alpha = alpha
        self.re = None
        
        torch.set_default_tensor_type('torch.FloatTensor')

        @torch.jit.script
        def l1_loss(tensor: torch.Tensor, l1: float):
            return tensor.abs().sum().mul(l1)

        @torch.jit.script
        def l2_loss(tensor: torch.Tensor, l2: float):
            return tensor.pow(2.0).sum().mul(l2)
        
        self.l1_l2 = [l1_loss, l2_loss]

    def __call__(self):
        pass

    
    def __repr__(self):
        return "Model_base: \n  \n"

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


    def add_env(self, input_shape, output_shape, hidden=[], activation=['linear'], l1=-99, l2=-99):
        self.env = (self._build_NN(input_shape, output_shape, hidden, activation))
        self.params.append(self.env.parameters())
        self.input_shape = input_shape
        self.output_shape = output_shape
        for p in self.env.parameters():
            if l1 > 0.0: 
                self.losses.append(lambda: self.l1_l2[0](p, l1))
            if l2 > 0.0:
                self.losses.append(lambda: self.l1_l2[1](p, l2))

    def add_spatial(self, input_shape, output_shape, hidden = [], activation = ['linear'], l1=-99, l2=-99):
        #hidden.append(1)
        self.spatial = (self._build_NN(input_shape, output_shape, hidden, activation))
        self.params.append(self.spatial.parameters())
        for p in self.spatial.parameters():
            if l1 > 0.0: 
                self.losses.append(lambda: self.l1_l2[0](p, l1))
            if l2 > 0.0:
                self.losses.append(lambda: self.l1_l2[1](p, l2))
                
    def _build_NN(self, input_shape, output_shape, hidden, activation):
        model_list = nn.ModuleList()
        if len(hidden) != len(activation):
            activation = [activation[0] for _ in range(len(hidden))]

        if len(hidden) > 0:
            for i in range(len(hidden)):
                if i == 0:
                    model_list.append(nn.Linear(input_shape, hidden[i], bias=False))
                else:
                    model_list.append(nn.Linear(hidden[i-1], hidden[i], bias=False))

                if activation[i] == "relu":
                     model_list.append(nn.ReLU())
                if activation[i] == "tanh": 
                    model_list.append(nn.Tanh())
                if activation[i] == "sigmoid":
                    model_list.append(nn.Sigmoid())
        
        if len(hidden) > 0:
            model_list.append(nn.Linear(hidden[-1], output_shape, bias=False))
        else:
            model_list.append(nn.Linear(input_shape, output_shape, bias=False))
        model = nn.Sequential(*model_list)
        return model
    
    def _get_DataLoader(self, X, Y=None,SP=None,RE=None, batch_size=25, shuffle=True, parallel=0, drop_last=True):
        # reticulate creates non writeable arrays
        X = X.copy()
        if type(Y) is np.ndarray:
            Y = Y.copy()
        if type(SP) is np.ndarray:
            SP = SP.copy()

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

    def build(self, df=None,Re=None, optimizer=None, l1=0.0, l2=0.0,
              reg_on_Cov=True, reg_on_Diag=True, inverse=False, link="probit", diag=False, scheduler=True,patience=2, factor = 0.05):
        
        if self.device.type == 'cuda' and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.env.cuda(self.device)
            if self.spatial is not None:
                self.spatial.cuda(self.device)
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.link = link
        self.df = df
        r_dim = self.output_shape
        if not diag:
            low = -np.sqrt(6.0/(r_dim+df))
            high = np.sqrt(6.0/(r_dim+df))               
            self.sigma = torch.tensor(np.random.uniform(low, high, [r_dim, df]), requires_grad = True, dtype = self.dtype, device = self.device).to(self.device)
            self._loss_function = self._build_loss_function()
            self._build_cov_constrain_function(l1 = l1, l2 = l2, reg_on_Cov = reg_on_Cov, reg_on_Diag = reg_on_Diag, inverse = inverse)
            self.params.append([self.sigma])
        else:
            self.sigma = torch.zeros([r_dim, r_dim], dtype = self.dtype, device = self.device).to(self.device)
            self.df = r_dim
            self._loss_function = self._build_loss_function()
            self._build_cov_constrain_function(l1 = l1, l2 = l2, reg_on_Cov = reg_on_Cov, reg_on_Diag = reg_on_Diag, inverse = inverse)
        
        if Re != None:
            self.re = torch.tensor(np.random.normal(0.0, 0.0001, [Re, 1]), requires_grad = True, dtype = self.dtype, device = self.device).to(self.device)
            self.params.append([self.re])

        if optimizer != None:
            self.optimizer = optimizer(params = itertools.chain(*self.params))

        if scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=patience, factor=factor, verbose=True)
            self.useSched = True
        else:
            self.useSched = False

    def fit(self, X, Y, SP=None, RE=None, batch_size=25, epochs=100, sampling=100, parallel=0):
        stepSize = np.floor(X.shape[0] / batch_size).astype(int)
        dataLoader = self._get_DataLoader(X, Y, SP, RE, batch_size, True, parallel)
        any_losses = len(self.losses) > 0
        #torch.set_default_tensor_type('torch.cuda.FloatTensor')
        batch_loss = np.zeros(stepSize)
        self.history = np.zeros(epochs)

        df = self.df
        alpha = self.alpha
        
        if self.device.type == 'cuda':
            device = self.device.type+ ":" + str(self.device.index)
        else:
            device = 'cpu'

        re_loss = lambda value: -torch.distributions.Normal(0.0, 1.0).log_prob(value)
        desc='loss: Inf'
        ep_bar = tqdm(range(epochs),bar_format= "Iter: {n_fmt}/{total_fmt} {l_bar}{bar}| [{elapsed}, {rate_fmt}{postfix}]", file=sys.stdout)
        if type(SP) is np.ndarray:
            for epoch in ep_bar:
                for step, (x, y, sp) in enumerate(dataLoader):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    sp = sp.to(self.device, non_blocking=True)
                    mu = self.env(x) + self.spatial(sp)
                    #tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float
                    loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device)
                    loss = loss.mean()
                    if any_losses:
                        for k in range(len(self.losses)):
                            loss+= self.losses[k]()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss[step] = loss.item()
                #torch.cuda.empty_cache()
                bl = np.mean(batch_loss)
                bl = np.round(bl, 3)
                #_ = sys.stdout.write("\rEpoch: {}/{} loss: {} ".format(epoch+1,epochs, np.round(bl, 3).astype(str)))
                ep_bar.set_postfix(loss=f'{bl}')
                #sys.stdout.flush()
                self.history[epoch] = bl
                if self.useSched:
                    self.scheduler.step(bl)                    
        else:
            for epoch in ep_bar:  
                for step, (x, y) in enumerate(dataLoader):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    mu = self.env(x)
                    #tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float
                    loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device)
                    loss = loss.mean()
                    if any_losses:
                        for k in range(len(self.losses)):
                            loss += self.losses[k]()
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    batch_loss[step] = loss.item()
                #torch.cuda.empty_cache()
                bl = np.mean(batch_loss)
                bl = np.round(bl, 3)
                #_ = sys.stdout.write("\rEpoch: {}/{} loss: {} ".format(epoch+1,epochs, np.round(bl, 3).astype(str)))
                ep_bar.set_postfix(loss=f'{bl}')
                #sys.stdout.flush()
                self.history[epoch] = bl
                if self.useSched:
                    self.scheduler.step(bl) 
        torch.cuda.empty_cache()
        
    def logLik(self,X, Y,SP=None,RE=None, batch_size=25, parallel=0, sampling=100,individual=False,train=True):
        """Returns log-likelihood of model

        :param X: 2D-numpy array, environemntal predictors
        :param Y: 2D-numpy array, species responses
        :param batch_size: int of 1, newdata will be split into batches
        :param sampling: int of 1, sampling parameter for the Monte-Carlo Integreation
        :param parallel: int of 1, number of workers for the dataLoader

        """
        dataLoader = self._get_DataLoader(X = X, Y = Y, SP=SP, RE=RE, batch_size = batch_size, shuffle = False, parallel = parallel, drop_last = False)
        loss_function = self._build_loss_function(train=train)
        torch.cuda.empty_cache()
        any_losses = len(self.losses) > 0
        
        if self.device.type == 'cuda':
            device = self.device.type+ ":" + str(self.device.index)
        else:
            device = 'cpu'

        logLik = []
        logLikReg = 0
                
        if type(SP) is np.ndarray:
            for step, (x, y, sp) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                sp = sp.to(self.device, non_blocking=True)
                mu = self.env(x) + self.spatial(sp)
                # loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device)
                loss = loss_function(mu, y, self.sigma, x.shape[0], sampling, self.df, self.alpha, device)
                #logLik += loss.data.cpu().numpy()
                logLik.append(loss.data)
        else:
            for step, (x, y) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                mu = self.env(x)
                # loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device)
                loss = loss_function(mu, y, self.sigma, x.shape[0], sampling, self.df, self.alpha, device)
                #logLik += loss.data.cpu().numpy()
                logLik.append(loss.data)
        
        #if any_losses:
        loss_reg = torch.tensor(0.0, device=self.device, dtype=self.dtype).to(self.device)
        for k in range(len(self.losses)):
                loss_reg = loss_reg.add(self.losses[k]())
            
        logLikReg = loss_reg.data.cpu().numpy()
        #print(loss_reg)
            
        torch.cuda.empty_cache()
        if individual is not True:
            logLik = torch.cat(logLik).sum().data.cpu().numpy()
        else:
            logLik = torch.cat(logLik).data.cpu().numpy()
        #print(logLikReg)
        return logLik, logLikReg

    def predict(self, newdata=None,SP=None,RE=None, train=False, batch_size=25, parallel=0, sampling=100, link=True):
        """predict for newdata
        
        Predict on newdata in batches

        :param newdata: 2D-numpy array, environmental data
        :param train: logical of 1, in case of dropout layer -> train state
        :param batch_size: int of 1, newdata will be split into batches
        :param sampling: int of 1, sampling parameter for the Monte-Carlo Integreation
        :param parallel: int of 1, number of workers for the dataLoader

        """
        dataLoader = self._get_DataLoader(X = newdata, Y = None, SP=SP,RE=RE, batch_size = batch_size, shuffle = False, parallel = parallel, drop_last = False)
        loss_function = self._build_loss_function(train = False, raw=not link)

        pred = []
        if self.device.type == 'cuda':
            device = self.device.type+ ":" + str(self.device.index)
        else:
            device = 'cpu'
        
        if type(SP) is np.ndarray:
            for step, (x, sp) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                sp = sp.to(self.device, non_blocking=True)
                mu = self.env(x) + self.spatial(sp)
                # loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device)
                loss = loss_function(mu, self.sigma, x.shape[0], sampling, self.df, self.alpha, device)
                pred.append(loss)
        else:
            for step, (x) in enumerate(dataLoader):
                x = x[0].to(self.device, non_blocking=True)
                mu = self.env(x)
                # loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device)
                loss = loss_function(mu, self.sigma, x.shape[0], sampling, self.df, self.alpha, device)
                pred.append(loss)
        predictions = torch.cat(pred, dim = 0).data.cpu().numpy()
        return predictions

    def se(self, X, Y, SP=None, RE=None, batch_size=25, parallel=0, sampling=100):
        dataLoader = self._get_DataLoader(X, Y, SP, RE, batch_size=batch_size, shuffle=False)
        loss_func = self._build_loss_function(train=True)
        se = []
        weights_base = np.transpose(self.env_weights[0])
        y_dim = Y.shape[1]

        if self.device.type == 'cuda':
            device = self.device.type+ ":" + str(self.device.index)
        else:
            device = 'cpu'
        
        _ = sys.stdout.write("\nCalculating standard errors...\n")
        #(mu: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str):
        if type(SP) is np.ndarray:
            for i in range(Y.shape[1]):
                _ = sys.stdout.write("\rSpecies: {}/{} ".format(i+1, y_dim))
                sys.stdout.flush()
                weights = torch.tensor(weights_base[:,i].reshape([-1,1]), device=self.device, dtype=self.dtype, requires_grad=True).to(self.device)
                if i == 0:
                    constants = torch.tensor(weights_base[:,(i+1):], device=self.device, dtype=self.dtype).to(self.device)
                    w = torch.cat([weights, constants], dim=1)
                elif i < y_dim:
                    w = torch.cat([torch.tensor(weights_base[:,0:i], device=self.device, dtype=self.dtype).to(self.device), 
                                   weights, 
                                   torch.tensor(weights_base[:,(i+1):], device=self.device, dtype=self.dtype).to(self.device)],dim=1)
                else:
                    constants = torch.tensor(weights_base[:,0:i], device=self.device, dtype=self.dtype).to(self.device)
                    w = torch.cat([constants, weights], dim=1)
                for step, (x, y, sp) in enumerate(dataLoader):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    sp = sp.to(self.device, non_blocking=True)
                    mu = torch.nn.functional.linear(x, w.t()) + self.spatial(sp)
                    loss = loss_func(mu, y,self.sigma, x.shape[0], sampling, self.df, self.alpha,device).sum()
                    first_gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True, allow_unused=True)
                    second = []
                    for j in range(self.input_shape):
                        second.append(torch.autograd.grad(first_gradients[0][j,0], inputs = weights, retain_graph = True, create_graph = False, allow_unused = False)[0])
                        hessian = torch.cat(second,dim=1)
                    if step < 1:
                        hessian_out = hessian
                    else:
                        hessian_out += hessian
                se.append(torch.sqrt(torch.diag(torch.inverse(hessian_out))).data.cpu().numpy())
            return se                
        else:
            for i in range(Y.shape[1]):
                _ = sys.stdout.write("\rSpecies: {}/{} ".format(i+1, y_dim))
                sys.stdout.flush()
                weights = torch.tensor(weights_base[:,i].reshape([-1,1]), device=self.device, dtype=self.dtype, requires_grad=True).to(self.device)
                if i == 0:
                    constants = torch.tensor(weights_base[:,(i+1):], device=self.device, dtype=self.dtype).to(self.device)
                    w = torch.cat([weights, constants], dim=1)
                elif i < y_dim:
                    w = torch.cat([torch.tensor(weights_base[:,0:i], device=self.device, dtype=self.dtype).to(self.device), 
                                   weights, 
                                   torch.tensor(weights_base[:,(i+1):], device=self.device, dtype=self.dtype).to(self.device)],dim=1)
                else:
                    constants = torch.tensor(weights_base[:,0:i], device=self.device, dtype=self.dtype).to(self.device)
                    w = torch.cat([constants, weights], dim=1)
                for step, (x, y) in enumerate(dataLoader):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    mu = torch.nn.functional.linear(x, w.t())
                    loss = loss_func(mu, y,self.sigma, x.shape[0], sampling, self.df, self.alpha,device).sum()
                    first_gradients = torch.autograd.grad(loss, weights, retain_graph=True, create_graph=True, allow_unused=True)
                    second = []
                    for j in range(self.input_shape):
                        second.append(torch.autograd.grad(first_gradients[0][j,0], inputs = weights, retain_graph = True, create_graph = False, allow_unused = False)[0])
                        hessian = torch.cat(second,dim=1)
                    if step < 1:
                        hessian_out = hessian
                    else:
                        hessian_out += hessian
                se.append(torch.sqrt(torch.diag(torch.inverse(hessian_out))).data.cpu().numpy())
            return se

    def _build_cov_constrain_function(self, l1=None, l2=None, reg_on_Cov=None, reg_on_Diag=None, inverse=None):
        if reg_on_Cov:
            if reg_on_Diag:
                diag = int(0)
            else:
                diag = int(1)
            if l1 > 0.0:
                @torch.jit.script
                def l1_ll(sigma: torch.Tensor, l1: float, diag: int, inverse: bool):
                    ss = sigma.matmul(sigma.t())
                    if inverse:
                        ss = ss.inverse()
                    return ss.triu(diag).abs().sum().mul(l1)
                self.losses.append(lambda: l1_ll(self.sigma, l1, diag, inverse))
            
            if l2 > 0.0 :
                @torch.jit.script
                def l2_ll(sigma: torch.Tensor, l2: float, diag: int, inverse: bool):
                    ss = sigma.matmul(sigma.t())
                    if inverse:
                        ss = ss.inverse()
                    return ss.triu(diag).pow(2.0).sum().mul(l2)
                self.losses.append(lambda: l2_ll(self.sigma, l2, diag, inverse))
        else:
            if l1 > 0.0:
                self.losses.append( lambda: self.l1_l2[0](self.sigma, l1) )
            if l2 > 0.0:
                self.losses.append( lambda: self.l1_l2[1](self.sigma, l2) )
        return None

    def _build_loss_function(self, train=True, raw=False):

        if train:
            if self.link == "logit":
                @torch.jit.script
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device))
                    E = torch.sigmoid(   torch.tensordot(noise, sigma.t(), [2], [0]).add(mu).mul(alpha)   ).mul(0.999999).add(0.0000005)
                    logprob = E.log().mul(Ys).add((1.0 - E).log().mul(1.0 - Ys)).neg().sum(dim = 2).neg()
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    loss = Eprob.log().neg().sub(maxlogprob)
                    return loss
            elif self.link == "linear":
                @torch.jit.script
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device))
                    E = torch.clamp(torch.tensordot(noise, sigma.t(), [2], [0]).add(mu).mul(alpha), 0.0, 1.0).mul(0.999999).add(0.0000005)
                    logprob = E.log().mul(Ys).add((1.0 - E).log().mul(1.0 - Ys)).neg().sum(dim = 2).neg()
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    loss = Eprob.log().neg().sub(maxlogprob)
                    return loss
            elif self.link == "probit":

                link_func = lambda value: torch.distributions.Normal(0.0, 1.0).cdf(value)
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device))
                    E = link_func(torch.tensordot(noise, sigma.t(), dims = 1).add(mu)).mul(0.999999).add(0.0000005)
                    logprob = E.log().mul(Ys).add((1.0 - E).log().mul(1.0 - Ys)).neg().sum(dim = 2).neg()
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    loss = Eprob.log().neg().sub(maxlogprob)
                    return loss               
        else:
            if self.link == "probit": 
                link_func = lambda value: torch.distributions.Normal(0.0, 1.0).cdf(value)
            elif self.link == "logit":
                link_func = lambda value: torch.sigmoid(value)
            elif self.link == "linear":
                link_func = lambda value: torch.clamp(value, 0.0, 1.0)

            if raw:
                link_func = lambda value: value

            def tmp(mu: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str):
                noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device))
                if self.link == "logit": 
                    E = link_func(torch.tensordot(noise, sigma.t(), 1).add(mu).mul(alpha)).mul(0.999999).add(0.0000005)
                else:
                    E = link_func(torch.tensordot(noise, sigma.t(), 1).add(mu)).mul(0.999999).add(0.0000005)
                return E.mean(dim = 0)

        return tmp

    @property
    def weights(self):
        return [(lambda p: p.data.cpu().numpy())(p) for p in self.params()] 

    @property
    def get_sigma(self):
        return self.sigma.data.cpu().numpy()

    def set_sigma(self, w):
        with torch.no_grad():
            self.sigma.data = torch.tensor(w, device=self.device, dtype=self.dtype).data
            
    @property
    def covariance(self):
        return (self.sigma.matmul(self.sigma.t()) + torch.eye(self.sigma.shape[0], dtype=self.sigma.dtype, device=self.device)).data.cpu().numpy()

    @property
    def env_weights(self):
        return [(lambda p: p.data.cpu().numpy())(p) for p in self.env.parameters()]

    def set_env_weights(self, w):
        counter = 0
        with torch.no_grad():
            for i in range(len(self.env)):
                if type(self.env[i]) is torch.nn.modules.linear.Linear:
                    self.env[i].weight = torch.nn.Parameter(torch.tensor(w[counter], dtype=self.env[i].weight.dtype, device=self.env[i].weight.device))
                    counter+=1

    @property
    def spatial_weights(self):
        if self.spatial is not None:
            return [(lambda p: p.data.cpu().numpy())(p) for p in self.spatial.parameters()]
        else:
            return None

    def set_spatial_weights(self, w):
        if self.spatial is not None:
            counter = 0
            with torch.no_grad():
                for i in range(len(self.spatial)):
                    if type(self.spatial[i]) is torch.nn.modules.linear.Linear:
                        self.spatial[i].weight = torch.nn.Parameter(torch.tensor(w[counter], dtype=self.spatial[i].weight.dtype, device=self.spatial[i].weight.device))
                        counter+=1
        else:
            return None



  # def fill_lower_tril(self, sigma):
   #     xc = torch.cat([sigma[self.r_dim:], sigma.flip(dims=[0])])
   #     y = xc.view(self.r_dim, self.r_dim)
   #     return torch.tril(y)