import numpy as np
import torch
import itertools
import sys
from .utils_fa import covariance
from typing import Union, Tuple, List, Optional, Callable
from tqdm import tqdm
from torch import nn, optim
import warnings 

warnings.filterwarnings("ignore")

class Model_sjSDM:
    def __init__(self, device: str = "cpu", dtype: str = "float32"):
        """sjSDM constructor

        Args:
            device (str, optional): Device type. Defaults to "cpu".
            dtype (str, optional): Dtype. Defaults to "float32".

        """
        self.params = []
        self.losses = []
        self.env = None
        self.spatial = None
        device, dtype = self._device_and_dtype(device, dtype) # type: ignore
        self.device = device
        self.dtype = dtype
        torch.set_default_dtype(self.dtype)
        torch.set_default_tensor_type('torch.FloatTensor')

        @torch.jit.script
        def l1_loss(tensor: torch.Tensor, l1: float):
            return tensor.abs().sum().mul(l1)

        @torch.jit.script
        def l2_loss(tensor: torch.Tensor, l2: float):
            return tensor.pow(2.0).sum().mul(l2)
        
        @torch.jit.script
        def l1_l2_loss(tensor: torch.Tensor, l1: float, l2: float):
            return tensor.pow(2.0).sum().mul(l2)+tensor.abs().sum().mul(l1)
        
        self.l1_l2 = [l1_loss, l2_loss, l1_l2_loss]

    def __call__(self):
        pass

    
    def __repr__(self):
        return "Model_base: \n  \n"

    @staticmethod
    def _device_and_dtype(device: Union[int, str, torch.device], dtype: Union[str, torch.dtype]) -> Tuple[torch.device, torch.dtype]:
        """Device and dtype parster

        Args:
            device (Union[int, str, torch.device]): device
            dtype (Union[str, torch.dtype]): dtype

        Returns:
            Tuple[torch.device, torch.dtype]: Tuple of device and dtype
        """        

        if type(device) is int:
            device = torch.device('cuda:'+str(device))

        if type(device) is str:
            device = torch.device(device)

        if type(dtype) is not str:
            return device, dtype # type: ignore

        if dtype == "float32":
            return device, torch.float32 # type: ignore
        if dtype == "float64":
            return device, torch.float64 # type: ignore
        return device, torch.float32 # type: ignore


    def add_env(self, 
                input_shape: int, 
                output_shape: int, 
                hidden: List = [], 
                activation: List[str] =['linear'], 
                bias: List[bool] = [False], 
                l1: float = -99, 
                l2: float = -99, 
                dropout: float = -99,
                intercept=False) -> None:
        """Add environmental model

        Args:
            input_shape (int): number of predictors
            output_shape (int): number of species
            hidden (List, optional): List of units for hidden layers (each entry corresponds to one hidden layer). Defaults to [].
            activation (List[str], optional): List of activation functions. Defaults to ['linear'].
            bias (List[bool], optional): Bias in each hidden layer. Defaults to [False].
            l1 (float, optional): LASSO regularization. Defaults to -99.
            l2 (float, optional): Ridge regularization. Defaults to -99.
            dropout (float, optional): [Dropoutrate. Defaults to -99.
        """                
        
        self.env = (self._build_NN(input_shape, output_shape, hidden,bias, activation, dropout))
        self.params.append(self.env.parameters())
        self.input_shape = input_shape
        self.output_shape = output_shape

        individual_losses = []
        for index, p in enumerate(self.env.parameters()):
            if index == 0:
                if (l1 > 0.0) & (l2 > 0.0): 
                    if intercept is False:
                        individual_losses.append(lambda p: self.l1_l2[2](p, l1, l2))
                    else:
                        individual_losses.append(lambda p: self.l1_l2[2](p[:,1:], l1, l2))
                    next
                elif (l1 <= 0.0) & (l2 > 0.0):
                    if intercept is False:
                        individual_losses.append(lambda p: self.l1_l2[1](p, l2))
                    else:
                        individual_losses.append(lambda p: self.l1_l2[1](p[:,1:], l2))
                elif (l1 > 0.0) & (l2 <= 0.0):
                    if intercept is False:
                        individual_losses.append(lambda p: self.l1_l2[0](p, l1))
                    else:
                        individual_losses.append(lambda p: self.l1_l2[0](p[:,1:], l1))
            else:
                if (l1 > 0.0) & (l2 > 0.0): 
                    individual_losses.append(lambda p: self.l1_l2[2](p, l1, l2))
                elif (l1 <= 0.0) & (l2 > 0.0):
                    individual_losses.append(lambda p: self.l1_l2[1](p, l2))
                elif (l1 > 0.0) & (l2 <= 0.0):
                    individual_losses.append(lambda p: self.l1_l2[0](p, l1))
        if len(individual_losses) > 0:
            def loss():
                tmp_loss = [
                    individual_losses[index](p).reshape([1])
                    for index, p in enumerate(self.env.parameters())
                    if len(p.shape) > 1
                ]
                return torch.cat(tmp_loss).sum()
            self.losses.append( loss )
                    


    def add_spatial(self, 
                    input_shape: int, 
                    output_shape: int, 
                    hidden: List = [], 
                    activation: List[str] = ['linear'], 
                    bias: List[bool] = [False], 
                    l1: float = -99, 
                    l2: float = -99, 
                    dropout: float = -99,
                    intercept=False) -> None:
        """Add spatial model

        Args:
            input_shape (int): number of predictors
            output_shape (int): number of species
            hidden (List, optional): List of units for hidden layers (each entry corresponds to one hidden layer). Defaults to [].
            activation (List[str], optional): List of activation functions. Defaults to ['linear'].
            bias (List[bool], optional): Bias in each hidden layer. Defaults to [False].
            l1 (float, optional): LASSO regularization. Defaults to -99.
            l2 (float, optional): Ridge regularization. Defaults to -99.
            dropout (float, optional): [Dropoutrate. Defaults to -99.
        """                    
        self.spatial = (self._build_NN(input_shape, output_shape, hidden, bias, activation, dropout))
        self.params.append(self.spatial.parameters())
        
        individual_losses = []
        for p in self.spatial.parameters():
            if (l1 > 0.0) & (l2 > 0.0): 
                individual_losses.append(lambda p: self.l1_l2[2](p, l1, l2))
            elif (l1 <= 0.0) & (l2 > 0.0):
                individual_losses.append(lambda p: self.l1_l2[1](p, l2))
            elif (l1 > 0.0) & (l2 <= 0.0):
                individual_losses.append(lambda p: self.l1_l2[0](p, l1))
        if len(individual_losses) > 0:
            def loss():
                tmp_loss = [
                    individual_losses[index](p).reshape([1])
                    for index, p in enumerate(self.spatial.parameters())
                    if len(p.shape) > 1
                ]
                return torch.cat(tmp_loss).sum()
            self.losses.append( loss )
        
                
    def _build_NN(self, 
                  input_shape: int, 
                  output_shape: int, 
                  hidden: List, 
                  bias: List[bool], 
                  activation: List[str], 
                  dropout: float) -> torch.nn.modules.container.Sequential:
        """Build neural network

        Args:
            input_shape (int): Number of predictors
            output_shape (int): Number of species
            hidden (List): List of hidden layers
            bias (List[bool]): Biases in hidden layers
            activation (List[str]): List of activation functions
            dropout (float): Dropout rate

        Returns:
            torch.nn.modules.container.Sequential: Sequential neural network object
        """                  
        model_list = nn.ModuleList()
        if len(hidden) != len(activation):
            activation = [activation[0] for _ in range(len(hidden))]

        if len(bias) == 1:
            bias = [bias[0] for _ in range(len(hidden))]
            
        bias.insert(0, False)

        if len(hidden) > 0:
            for i in range(len(hidden)):
                if i == 0:
                    model_list.append(nn.Linear(input_shape, hidden[i], bias=bias[i]).type(self.dtype))
                else:
                    model_list.append(nn.Linear(hidden[i-1], hidden[i], bias=bias[i]).type(self.dtype))

                if activation[i] == "relu":
                     model_list.append(nn.ReLU())
                if activation[i] == "selu":
                     model_list.append(nn.SELU())
                if activation[i] == "leakyrelu":
                     model_list.append(nn.LeakyReLU())
                if activation[i] == "tanh": 
                    model_list.append(nn.Tanh())
                if activation[i] == "sigmoid":
                    model_list.append(nn.Sigmoid())
                if dropout > 0.0:
                    model_list.append(nn.Dropout(p=dropout))

        if len(hidden) > 0:
            model_list.append(nn.Linear(hidden[-1], output_shape, bias=bias[-1]).type(self.dtype))
        else:
            model_list.append(nn.Linear(input_shape, output_shape, bias=False).type(self.dtype))
        return nn.Sequential(*model_list)
    
    def _get_DataLoader(self, 
                        X: np.ndarray, 
                        Y: Optional[np.ndarray] = None,
                        SP: Optional[np.ndarray] = None,
                        batch_size: int = 25, 
                        shuffle: bool = True, 
                        parallel: int = 0, 
                        drop_last: bool = True) -> torch.utils.data.DataLoader:
        """Create dataloader

        Args:
            X (np.ndarray): Environment (n*p)
            Y (Optional[np.ndarray], optional): Species occurence matrix (n*s). Defaults to None.
            SP (Optional[np.ndarray], optional): Spatial matrix (n*sp). Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 25.
            shuffle (bool, optional): Shuffle data or not. Defaults to True.
            parallel (int, optional): Parallelize data fetching or not. Defaults to 0.
            drop_last (bool, optional): Drop last rows in left-over batch or not. Defaults to True.

        Returns:
            torch.utils.data.DataLoader: DataLoader object
        """                        
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

        if type(Y) is np.ndarray:
            if type(SP) is np.ndarray:
                data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=self.dtype, device=torch.device('cpu')), 
                                                      torch.tensor(Y, dtype=self.dtype, device=torch.device('cpu')),
                                                      torch.tensor(SP, dtype=self.dtype, device=torch.device('cpu')))
            else:
                data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=self.dtype, device=torch.device('cpu')), 
                                                    torch.tensor(Y, dtype=self.dtype, device=torch.device('cpu')))
        else:
            if type(SP) is np.ndarray:
                data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=self.dtype, device=torch.device('cpu')),
                                                    torch.tensor(SP, dtype=self.dtype, device=torch.device('cpu')))
            else: 
                data = torch.utils.data.TensorDataset(torch.tensor(X, dtype=self.dtype, device=torch.device('cpu')))            

        DataLoader = torch.utils.data.DataLoader(data, batch_size=int(batch_size), shuffle=shuffle, num_workers=int(parallel), pin_memory=pin_memory, drop_last=drop_last)
        torch.cuda.empty_cache()
                
        return DataLoader

    def build(self, 
              df: Optional[int] = None,
              optimizer: Optional[Callable] = None, 
              l1: float = 0.0, 
              l2: float = 0.0,
              reg_on_Cov: bool = True, 
              reg_on_Diag: bool = True, 
              inverse: bool = False, 
              link: str = "probit",
              alpha: float = 1.0, 
              diag: bool = False, 
              scheduler: bool = True,
              patience: int = 2, 
              factor: float = 0.95,
              mixed: bool = False) -> None:
        """Build and specify sjSDM Model

        Args:
            df (Optional[int], optional): Sigma is parametrized by a [sp, df] matrix. Defaults to None.
            optimizer (Optional[Callable], optional): Optimizer used for stochastic gradient descent. Defaults to None.
            l1 (float, optional): LASSO regularization on sigma. Defaults to 0.0.
            l2 (float, optional): Ridge regularization on sigma. Defaults to 0.0.
            reg_on_Cov (bool, optional): Regularization on sigma or on it's square-root matrix (used to parametrize sigma). Defaults to True.
            reg_on_Diag (bool, optional): Regularization on diagonals or not. Defaults to True.
            inverse (bool, optional): Regularization on the precision matrix or not. Defaults to False.
            link (str, optional): which link/family to use. Defaults to "probit".
            alpha (float, optional): 1.0 corresponds to logit and 1.70169 to probit. Defaults to 1.0.
            diag (bool, optional): Use only diagonal matrix or not. Defaults to False.
            scheduler (bool, optional): Use learning rate scheduler or not. Defaults to True.
            patience (int, optional): Number of epochs for lr scheduler. Defaults to 2.
            factor (float, optional): Rate to decrease learning rate. Defaults to 0.95.
            mixed (bool, optional): use mixed half-precision training or not. Defaults to False.

        """              
        
        if self.device.type == 'cuda' and torch.cuda.is_available():
            #torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.env.cuda(self.device) # type: ignore
            if self.spatial is not None:
                self.spatial.cuda(self.device) # type: ignore
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        
        self.link = link
        self.df = df
        self.l1 = l1 #test
        self.l2 = l2 #test
        self.alpha = alpha
        self.mixed = mixed
        r_dim = self.output_shape
        
        if link == "nbinom":
            self.theta = torch.ones([r_dim], requires_grad = True, dtype = self.dtype, device = self.device).to(self.device)
            self.params.append([self.theta])
            
        if not diag:
            low = -np.sqrt(6.0/(r_dim+df)) # type: ignore
            high = np.sqrt(6.0/(r_dim+df)) # type: ignore      
            self.sigma = torch.tensor(np.random.uniform(low, high, [r_dim, df]), requires_grad = True, dtype = self.dtype, device = self.device).to(self.device) # type: ignore
            self._loss_function = self._build_loss_function()
            self._build_cov_constrain_function(l1 = l1, l2 = l2, reg_on_Cov = reg_on_Cov, reg_on_Diag = reg_on_Diag, inverse = inverse)
            self.params.append([self.sigma])
        else:
            self.sigma = torch.eye(r_dim, dtype = self.dtype, device = self.device).to(self.device) # type: ignore
            self.df = r_dim
            self._loss_function = self._build_loss_function()
            self._build_cov_constrain_function(l1 = l1, l2 = l2, reg_on_Cov = reg_on_Cov, reg_on_Diag = reg_on_Diag, inverse = inverse)

        if optimizer != None:
            self.optimizer = optimizer(params = itertools.chain(*self.params))

        if scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", patience=patience, factor=factor, verbose=True)
            self.useSched = True
        else:
            self.useSched = False

    def fit(self, 
            X: np.ndarray, 
            Y: np.ndarray, 
            SP: Optional[np.ndarray] = None, 
            batch_size: int = 25, 
            epochs: int = 100, 
            sampling: int = 100, 
            parallel: int = 0,
            early_stopping_training: int = -1) -> None:
        """Fit sjSDM model

        Args:
            X (np.ndarray): Environment, n*p matrix
            Y (np.ndarray): Species occurrence matrix, n*s matrix
            SP (Optional[np.ndarray], optional): Spatial predictors, n*sp matrix. Defaults to None.
            batch_size (int, optional): Batch size for stochastic gradient descent. Defaults to 25.
            epochs (int, optional): Number of epochs. Defaults to 100.
            sampling (int, optional): Number of MC samples for each species. Defaults to 100.
            parallel (int, optional): Use parallelization for DataLoader. Defaults to 0.
            early_stopping_training (int, optional): Use early stopping or not. Defaults to -1.
        """            
        stepSize = np.floor(X.shape[0] / batch_size).astype(int) # type: ignore
        dataLoader = self._get_DataLoader(X, Y, SP, batch_size, True, parallel)
        any_losses = len(self.losses) > 0
        batch_loss = np.zeros(stepSize)
        self.history = np.zeros(epochs)

        df = self.df
        alpha = self.alpha
        
        if self.device.type == 'cuda': # type: ignore
            device = self.device.type+ ":" + str(self.device.index) # type: ignore
        else:
            device = 'cpu'

        early_stopping_training_loss = np.inf
        counter_early_stopping_training = 0

        if early_stopping_training > 0:
            early_stopping_training_boolean = True
        else:
            early_stopping_training_boolean = False

        mixed = self.mixed

        if mixed:
            scaler = torch.cuda.amp.GradScaler()
            def update_func(loss):
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
        else:
            def update_func(loss):
                loss.backward()
                self.optimizer.step()

        desc='loss: Inf'
        ep_bar = tqdm(range(epochs),bar_format= "Iter: {n_fmt}/{total_fmt} {l_bar}{bar}| [{elapsed}, {rate_fmt}{postfix}]", file=sys.stdout)
        if type(SP) is np.ndarray:
            for epoch in ep_bar:
                for step, (x, y, sp) in enumerate(dataLoader):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    sp = sp.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=mixed):
                        mu = self.env(x) + self.spatial(sp) # type: ignore
                        #tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float
                        loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device, self.dtype)
                        loss = loss.mean()
                        if any_losses:
                            for k in range(len(self.losses)):
                                loss+= self.losses[k]()
                    update_func(loss)
                    batch_loss[step] = loss.item()
                bl = np.mean(batch_loss)
                bl = np.round(bl, 3)
                ep_bar.set_postfix(loss=f'{bl}')

                self.history[epoch] = bl
                if self.useSched:
                    self.scheduler.step(bl)
                
                if early_stopping_training_boolean:
                    if bl < early_stopping_training_loss:
                        early_stopping_training_loss = bl
                        counter_early_stopping_training = 0
                    else:
                        counter_early_stopping_training+=1
                    if counter_early_stopping_training == early_stopping_training:
                        _ = sys.stdout.write("\nEarly stopping...")
                        break
            self.spatial.eval() # type: ignore      
        else:
            for epoch in ep_bar:  
                for step, (x, y) in enumerate(dataLoader):
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    self.optimizer.zero_grad()
                    with torch.cuda.amp.autocast(enabled=mixed):
                        mu = self.env(x) # type: ignore
                        loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device, self.dtype)
                        loss = loss.mean()
                        if any_losses:
                            for k in range(len(self.losses)):
                                loss += self.losses[k]()
                    update_func(loss)
                    batch_loss[step] = loss.item()
                bl = np.mean(batch_loss)
                bl = np.round(bl, 3)
                ep_bar.set_postfix(loss=f'{bl}')
                self.history[epoch] = bl
                if self.useSched:
                    self.scheduler.step(bl)
                
                if early_stopping_training_boolean:
                    if bl < early_stopping_training_loss:
                        early_stopping_training_loss = bl
                        counter_early_stopping_training = 0
                    else:
                        counter_early_stopping_training+=1
                    if counter_early_stopping_training == early_stopping_training:
                        _ = sys.stdout.write("\nEarly stopping...")
                        break
        torch.cuda.empty_cache()
        self.env.eval()

        
    def logLik(self,
               X: np.ndarray,  
               Y: np.ndarray,
               SP: Optional[np.ndarray] = None, 
               batch_size: int = 25, 
               parallel: int= 0, 
               sampling: int = 100,
               individual: bool = False,
               train: bool = True) -> Tuple[float, float]:
        """Calculate log-Likelihood

        Args:
            X (np.ndarray): Environment, n*p matrix
            Y (np.ndarray): Species occurrence matrix, n*s matrix
            SP (Optional[np.ndarray], optional): Spatial predictor matrix, n*sp matrix. Defaults to None.
            batch_size (int, optional): Batch size (to prevent memory leakage). Defaults to 25.
            parallel (int, optional): Use parallelization for DataLoader. Defaults to 0.
            sampling (int, optional): Number of MC samples for each species. Defaults to 100.
            individual (bool, optional): Calculate log-LL for each species or not. Defaults to False.
            train (bool, optional): Train or evaluation mode. Defaults to True.

        Returns:
            Tuple[float, float]: Tuple of log-Likelihood and regularization loss
        """               
        dataLoader = self._get_DataLoader(X = X, Y = Y, SP=SP, batch_size = batch_size, shuffle = False, parallel = parallel, drop_last = False)
        loss_function = self._build_loss_function(train=train, individual=individual)
        torch.cuda.empty_cache()
        any_losses = len(self.losses) > 0

        if self.device.type == 'cuda': # type: ignore
            device = self.device.type+ ":" + str(self.device.index) # type: ignore
        else:
            device = 'cpu'

        logLik = []
        logLikReg = 0
        if type(SP) is np.ndarray:
            for step, (x, y, sp) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                sp = sp.to(self.device, non_blocking=True)
                mu = self.env(x) + self.spatial(sp) # type: ignore
                loss = loss_function(mu, y, self.sigma, x.shape[0], sampling, self.df, self.alpha, device, self.dtype)
                logLik.append(loss.data)
        else:
            for step, (x, y) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                mu = self.env(x) # type: ignore
                loss = loss_function(mu, y, self.sigma, x.shape[0], sampling, self.df, self.alpha, device, self.dtype)
                logLik.append(loss.data)

        loss_reg = torch.tensor(0.0, device=self.device, dtype=self.dtype).to(self.device) # type: ignore
        for k in range(len(self.losses)):
                loss_reg = loss_reg.add(self.losses[k]())

        logLikReg = loss_reg.data.cpu().numpy()
        torch.cuda.empty_cache()
        if not individual:
            logLik = torch.cat(logLik).sum().data.cpu().numpy()
        else:
            logLik = torch.cat(logLik).data.cpu().numpy()
        return logLik, logLikReg

    def predict(self, 
                newdata: Optional[np.ndarray] = None,
                SP: Optional[np.ndarray] = None,
                train: bool = False, 
                batch_size: int = 25, 
                parallel: int = 0, 
                sampling: int = 100, 
                link: bool = True,
                dropout: bool = False,
                simulate: bool = False) -> np.ndarray:
        """Predict with sjSDM

        Args:
            newdata (Optional[np.ndarray], optional): (New) environment, n*p matrix. Defaults to None.
            SP (Optional[np.ndarray], optional): Spatial predictor matrix, n*sp matrix. Defaults to None.
            train (bool, optional): train or evaluation mode. Defaults to False.
            batch_size (int, optional): Batch size. Defaults to 25.
            parallel (int, optional): Parallelization of DataLoader. Defaults to 0.
            sampling (int, optional): Number of MC-samples for each species. Defaults to 100.
            link (bool, optional): Linear or response scale. Defaults to True.
            dropout (bool, optional): Use dropout during predictions or not. Defaults to False.
            simulate (bool, optional): Return simulated values on linear scale. Defaults to False.

        Returns:
            np.ndarray: Predictions
        """                

        dataLoader = self._get_DataLoader(X = newdata, Y = None, SP=SP, batch_size = batch_size, shuffle = False, parallel = parallel, drop_last = False)
        loss_function = self._build_loss_function(train = False, raw=not link, simulate=simulate)

        pred = []
        if self.device.type == 'cuda': # type: ignore
            device = self.device.type+ ":" + str(self.device.index) # type: ignore
        else:
            device = 'cpu'

        self.env.eval()
        if type(SP) is np.ndarray:
            self.spatial.eval()
        
        # TODO: export to method
        if dropout:
            self.env.train()
            if type(SP) is np.ndarray:
                self.spatial.train()
        
        if type(SP) is np.ndarray:
            for step, (x, sp) in enumerate(dataLoader):
                x = x.to(self.device, non_blocking=True)
                sp = sp.to(self.device, non_blocking=True)
                mu = self.env(x) + self.spatial(sp) # type: ignore
                # loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device)
                loss = loss_function(mu, self.sigma, x.shape[0], sampling, self.df, self.alpha, device, self.dtype)
                pred.append(loss)
            self.spatial.eval()
        else:
            for step, (x) in enumerate(dataLoader):
                x = x[0].to(self.device, non_blocking=True)
                mu = self.env(x) # type: ignore
                # loss = self._loss_function(mu, y, self.sigma, batch_size, sampling, df, alpha, device)
                loss = loss_function(mu, self.sigma, x.shape[0], sampling, self.df, self.alpha, device, self.dtype)
                pred.append(loss)

        self.env.eval()
        cat_dim = 0
        if simulate:
            cat_dim = 1
        return torch.cat(pred, dim = cat_dim).data.cpu().numpy()

    def se(self, 
           X: np.ndarray, 
           Y: np.ndarray, 
           SP: Optional[np.ndarray] = None, 
           batch_size: int = 25, 
           parallel: int = 0, 
           sampling: int = 100) -> List[np.ndarray]:
        """Calculate standard errors for environmental coefficients

        Args:
            X (np.ndarray): Environment, n*p matrix
            Y (np.ndarray): Species occurences, n*s matrix
            SP (Optional[np.ndarray], optional): Spatial predictor matrix, n*sp matrix. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to 25.
            parallel (int, optional): Parallelization of DataLoader. Defaults to 0.
            sampling (int, optional): Number of MC samples for each species. Defaults to 100.

        Returns:
            List[np.ndarray]: Standard errors for environmental coefficients
        """           
        dataLoader = self._get_DataLoader(X, Y, SP, batch_size=batch_size, shuffle=False)
        loss_func = self._build_loss_function(train=True)
        se = []
        weights_base = np.transpose(self.env_weights[0])
        y_dim = Y.shape[1]

        if self.device.type == 'cuda':
            device = self.device.type+ ":" + str(self.device.index)
        else:
            device = 'cpu'
        
        _ = sys.stdout.write("\nCalculating standard errors...\n")

        desc='loss: Inf'
        sp_bar = tqdm(range(Y.shape[1]),bar_format= "Species: {n_fmt}/{total_fmt} {l_bar}{bar}| [{elapsed}, {rate_fmt}]", file=sys.stdout)

        if type(SP) is np.ndarray:
            for i in sp_bar:
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
                    loss = loss_func(mu, y,self.sigma, x.shape[0], sampling, self.df, self.alpha, device, self.dtype).sum()
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
            for i in sp_bar:
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
                    loss = loss_func(mu, y, self.sigma, x.shape[0], sampling, self.df, self.alpha, device, self.dtype).sum()
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

    def _build_cov_constrain_function(self, 
                                      l1: float = None, 
                                      l2: float = None, 
                                      reg_on_Cov: Optional[bool] = None, 
                                      reg_on_Diag: Optional[bool] = None, 
                                      inverse: Optional[bool] = None) -> None:
        """Build regularization losses for species associations

        Args:
            l1 (float, optional): LASSO regularization. Defaults to None.
            l2 (float, optional): Ridge regularization. Defaults to None.
            reg_on_Cov (Optional[bool], optional): Regularization on covariance matrix. Defaults to None.
            reg_on_Diag (Optional[bool], optional): Regularization on diagonals. Defaults to None.
            inverse (Optional[bool], optional): Regularization on precision matrix. Defaults to None.

        """                                      
        if reg_on_Cov:
            if reg_on_Diag:
                diag = int(0)
            else:
                diag = int(1)
            
            if l1 > 0.0 and l2 > 0.0:
                if inverse:
                    identity = torch.eye(self.sigma.shape[0], dtype=self.sigma.dtype, device=self.sigma.device).to(self.sigma.device)
                    @torch.jit.script
                    def l1_l2_ll(sigma: torch.Tensor, l1: float, l2: float, diag: int, identity: torch.Tensor):
                        sigma1 = sigma.matmul(sigma.t()).add(identity)
                        #ss = sigma1.add(identity).inverse()
                        ss = identity.cholesky_solve(sigma1.cholesky())
                        return ss.triu(diag).abs().sum().mul(l1) + ss.tril(-1).abs().sum().mul(l1) + ss.triu(diag).pow(2.0).sum().mul(l2) + ss.tril(-1).pow(2.0).sum().mul(l2) #+ sigma1.pow(2.0).sum().mul(0.0001)
                    self.losses.append(lambda: l1_l2_ll(self.sigma, l1,l2, diag, identity))
                else:
                    @torch.jit.script
                    def l1_l2_ll(sigma: torch.Tensor, l1: float, l2: float, diag: int):
                        ss = sigma.matmul(sigma.t())
                        #ss = ss.add(identity).inverse()
                        return ss.triu(diag).abs().sum().mul(l1) + ss.tril(-1).abs().sum().mul(l1) + ss.triu(diag).pow(2.0).sum().mul(l2) + ss.tril(-1).pow(2.0).sum().mul(l2)
                    self.losses.append(lambda: l1_l2_ll(self.sigma, l1,l2, diag))
            else:    
                if l1 > 0.0:
                    if inverse:
                        identity = torch.eye(self.sigma.shape[0], dtype=self.sigma.dtype, device=self.sigma.device).to(self.sigma.device)
                        @torch.jit.script
                        def l1_ll(sigma: torch.Tensor, l1: float, diag: int, identity: torch.Tensor):
                            sigma2= sigma.matmul(sigma.t()).add(identity)
                            #ss = sigma2.add(identity).inverse()
                            ss = identity.cholesky_solve(sigma2.cholesky())
                            return ss.triu(diag).abs().sum().mul(l1) + ss.tril(-1).abs().sum().mul(l1) #  + sigma2.pow(2.0).sum().mul(0.0001)
                        self.losses.append(lambda: l1_ll(self.sigma, l1, diag, identity))
                    else:
                        @torch.jit.script
                        def l1_ll(sigma: torch.Tensor, l1: float, diag: int):
                            ss = sigma.matmul(sigma.t())
                            return ss.triu(diag).abs().sum().mul(l1) + ss.tril(-1).abs().sum().mul(l1)
                        self.losses.append(lambda: l1_ll(self.sigma, l1, diag))                    

                if l2 > 0.0 :
                    if inverse:
                        identity = torch.eye(self.sigma.shape[0], dtype=self.sigma.dtype, device=self.sigma.device).to(self.sigma.device)
                        @torch.jit.script
                        def l2_ll(sigma: torch.Tensor, l2: float, diag: int, identity: torch.Tensor):
                            sigma2 = sigma.matmul(sigma.t()).add(identity)
                            #ss = sigma2.inverse()
                            ss = identity.cholesky_solve(sigma2.cholesky())
                            return ss.triu(diag).pow(2.0).sum().mul(l2) + ss.tril(-1).pow(2.0).sum().mul(l2)
                        self.losses.append(lambda: l2_ll(self.sigma, l2, diag, identity))
                    else:
                        identity = torch.eye(self.sigma.shape[0], dtype=self.sigma.dtype, device=self.sigma.device).to(self.sigma.device)
                        @torch.jit.script
                        def l2_ll(sigma: torch.Tensor, l2: float, diag: int):
                            ss = sigma.matmul(sigma.t())
                            return ss.triu(diag).pow(2.0).sum().mul(l2) + ss.tril(-1).pow(2.0).sum().mul(l2)
                        self.losses.append(lambda: l2_ll(self.sigma, l2, diag))                    
        else:
            if l1 > 0.0:
                self.losses.append( lambda: self.l1_l2[0](self.sigma, l1) )
            if l2 > 0.0:
                self.losses.append( lambda: self.l1_l2[1](self.sigma, l2) )

    def _build_loss_function(self, train: bool = True, raw: bool = False, individual:bool = False, simulate: bool = False) -> Callable:
        """Build loss (likelihood) function

        Args:
            train (bool, optional): Train or evaulation mode Defaults to True.
            raw (bool, optional): Linear or response scale. Defaults to False.
            individual(bool, optional): Return individual logLL values. Defaults to False.
            simulate(bool, optional): Return simulated values. Defaults to False.

        Returns:
            Callable: loss function
        """
        
        if simulate:
            #tmp(mu: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str):
            @torch.jit.script
            def tmp(mu: torch.Tensor,  sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device), dtype=dtype)
                return torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu)
            return tmp

        if train:
            if self.link == "logit" or self.link == "probit":
                @torch.jit.script
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device), dtype=dtype)
                    E = torch.sigmoid(   torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu).mul(alpha)   ).mul(0.999999).add(0.0000005)
                    logprob = E.log().mul(Ys).add((1.0 - E).log().mul(1.0 - Ys)).neg().sum(dim = 2).neg()
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    loss = Eprob.log().neg().sub(maxlogprob)
                    return loss
            elif self.link == "linear":
                @torch.jit.script
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device), dtype=dtype)
                    E = torch.clamp(torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu).mul(alpha), 0.0, 1.0).mul(0.999999).add(0.0000005)
                    logprob = E.log().mul(Ys).add((1.0 - E).log().mul(1.0 - Ys)).neg().sum(dim = 2).neg()
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    loss = Eprob.log().neg().sub(maxlogprob)
                    return loss
            elif self.link == "count":
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device),dtype=dtype)
                    E = torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu).exp()
                    logprob = torch.distributions.Poisson(rate=E).log_prob(Ys).sum(2)
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    return Eprob.log().neg().sub(maxlogprob)
            elif self.link == "nbinom":
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device),dtype=dtype)
                    E = torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu).exp()
                    eps = 0.0001
                    theta = 1.0/(torch.nn.functional.softplus(self.theta)+eps)
                    probs = torch.clamp((1.0 - theta/(theta+E)) + eps, 0.0, 1.0-eps)
                    logprob = torch.distributions.NegativeBinomial(total_count=theta, probs=probs).log_prob(Ys).sum(2)
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    return Eprob.log().neg().sub(maxlogprob)          

            elif self.link == "normal":
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma.matmul(sigma.t()).add(torch.eye(sigma.shape[0]))).log_prob(Ys).neg()
        else:
            if self.link == "probit": 
                link_func = lambda value: torch.distributions.Normal(0.0, 1.0).cdf(value)
            elif self.link == "logit":
                link_func = lambda value: torch.sigmoid(value)
            elif self.link == "linear":
                link_func = lambda value: torch.clamp(value, 0.0, 1.0)
            elif self.link == "count":
                link_func = lambda value: value.exp()
            elif self.link == "nbinom":
                link_func = lambda value: value.exp()
            elif self.link == "normal":
                link_func = lambda value: value

            if raw:
                link_func = lambda value: value

            def tmp(mu: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                # noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device))
                if self.link == "logit": 
                    E = link_func(mu.mul(alpha)).mul(0.999999).add(0.0000005)
                else:
                    E = link_func(mu).mul(0.999999).add(0.0000005)
                return E
    
    
        if individual:
            if self.link == "logit" or self.link == "probit":
                #@torch.jit.script
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device), dtype=dtype)
                    E = torch.sigmoid(   torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu).mul(alpha)   ).mul(0.999999).add(0.0000005)
                    logprob = E.log().mul(Ys).add((1.0 - E).log().mul(1.0 - Ys)).neg()#.sum(dim = 2).neg()
                    Prop = logprob/logprob.sum(dim = 2).reshape([sampling, batch_size, 1])
                    logprob = logprob.sum(dim = 2).neg()
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    return Eprob.log().neg().sub(maxlogprob).reshape([batch_size, 1])
            elif self.link == "linear":
                @torch.jit.script
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device), dtype=dtype)
                    E = torch.clamp(torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu).mul(alpha), 0.0, 1.0).mul(0.999999).add(0.0000005)
                    logprob = E.log().mul(Ys).add((1.0 - E).log().mul(1.0 - Ys)).neg() #.sum(dim = 2).neg()
                    logprob = logprob.sum(dim = 2).neg()
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    return Eprob.log().neg().sub(maxlogprob).reshape([batch_size, 1])
            elif self.link == "count":
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device),dtype=dtype)
                    E = torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu).exp()
                    logprob = torch.distributions.Poisson(rate=E).log_prob(Ys)#.sum(2)
                    logprob = logprob.sum(dim = 2)# .neg()
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    return Eprob.log().neg().sub(maxlogprob).reshape([batch_size, 1])
            elif self.link == "nbinom":
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    noise = torch.randn(size = [sampling, batch_size, df], device=torch.device(device),dtype=dtype)
                    E = torch.einsum("ijk, lk -> ijl", [noise, sigma]).add(mu).exp()
                    eps = 0.0001
                    theta = 1.0/(torch.nn.functional.softplus(self.theta)+eps)
                    probs = torch.clamp((1.0 - theta/(theta+E)) + eps, 0.0, 1.0-eps)
                    logprob = torch.distributions.NegativeBinomial(total_count=theta, probs=probs).log_prob(Ys).sum(2)
                    maxlogprob = logprob.max(dim = 0).values
                    Eprob = logprob.sub(maxlogprob).exp().mean(dim = 0)
                    return Eprob.log().neg().sub(maxlogprob).reshape([batch_size, 1])

            elif self.link == "normal":
                def tmp(mu: torch.Tensor, Ys: torch.Tensor, sigma: torch.Tensor, batch_size: int, sampling: int, df: int, alpha: float, device: str, dtype: torch.dtype):
                    return torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma.matmul(sigma.t()).add(torch.eye(sigma.shape[0]))).log_prob(Ys).neg()

        return tmp

    @property
    def weights(self):
        return [(lambda p: p.data.cpu().numpy())(p) for p in self.params()] 

    @property
    def get_sigma(self):
        return self.sigma.data.cpu().numpy()

    def set_sigma(self, w: np.ndarray):
        """Set sigma

        Args:
            w (np.ndarray): Square-root sigma, s*df matrix 
        """        
        with torch.no_grad():
            self.sigma.data = torch.tensor(w, device=self.device, dtype=self.dtype).data
            
    @property
    def covariance(self):
        return (self.sigma.matmul(self.sigma.t()) + torch.eye(self.sigma.shape[0], dtype=self.sigma.dtype, device=self.device)).data.cpu().numpy()

    @property
    def env_weights(self):
        return [(lambda p: p.data.cpu().numpy())(p) for p in self.env.parameters()]

    def set_env_weights(self, w: np.ndarray):
        """Set environmental coefficients

        Args:
            w (np.ndarray): environmental coefficients, p*s matrix
        """        
        with torch.no_grad():
            counter = 0
            for i in range(len(self.env)):
                if type(self.env[i]) is torch.nn.modules.linear.Linear:
                    self.env[i].weight = torch.nn.Parameter(torch.tensor(w[counter], dtype=self.env[i].weight.dtype, device=self.env[i].weight.device))
                    counter+=1
                    if self.env[i].bias is not None:
                        self.env[i].bias = torch.nn.Parameter(torch.tensor(w[counter], dtype=self.env[i].bias.dtype, device=self.env[i].bias.device))
                        counter+=1

    @property
    def spatial_weights(self):
        if self.spatial is not None:
            return [(lambda p: p.data.cpu().numpy())(p) for p in self.spatial.parameters()]
        else:
            return None
        
    @property
    def get_theta(self):
        if self.theta is not None:
            return self.theta.data.cpu().numpy()
        else:
            return None
        
    def set_theta(self, w: np.ndarray):
        if self.theta is None:
            return None
        else:
            with torch.no_grad():
                self.theta.data = torch.tensor(w, device=self.device, dtype=self.dtype).data          

    def set_spatial_weights(self, w: np.ndarray):
        """Set spatial coefficients

        Args:
            w (np.ndarray): spatial coefficients, sp*ps matrix

        """        
        if self.spatial is None:
            return None
        with torch.no_grad():
            counter = 0
            for i in range(len(self.spatial)):
                if type(self.spatial[i]) is torch.nn.modules.linear.Linear:
                    self.spatial[i].weight = torch.nn.Parameter(torch.tensor(w[counter], dtype=self.spatial[i].weight.dtype, device=self.spatial[i].weight.device))
                    counter+=1
                    if self.spatial[i].bias is not None:
                        self.spatial[i].bias = torch.nn.Parameter(torch.tensor(w[counter], dtype=self.spatial[i].bias.dtype, device=self.spatial[i].bias.device))
                        counter+=1
