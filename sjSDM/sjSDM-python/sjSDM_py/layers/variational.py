import numpy as np
import torch
from ..utils_fa import _device_and_dtype

class Layer_dense_variational:
    """Layer_dense_veriational object

    creates dense (fully connected) layer object.

    :param hidden: int of 1, input shape == output shape if last layer
    :param activation: str of 1, None/tanh/sigmoid/relu are currently supported
    :param l1: float of 1, lasso penality on weights
    :param l2: float of 1, ridge penality on weights
    :param device: str of 1, "cpu" or "gpu"
    :param dtype: str of 1, "float32" or "float64"

    # Example

        >>> # if 10 species:
        >>> layer = Layer_dense(10, activation=None)

    """
    def __init__(self, hidden=None, activation=None, bias=False, 
                 prior = 0.5, prior_trainable = False, 
                 kl_weight = 0.01, device="cpu", dtype="float32"):
        self.hidden = hidden
        self.activation = activation
        self.__w = None
        self.__w_sd = None
        self.__b = None
        self.__b_sd = None
        self.bias = bias
        self.prior = prior
        self.__prior_w = None
        self.__prior_b = None
        self.prior_trainable = prior_trainable
        self.kl_weight = kl_weight
        self._kernel = None
        self._bias_kernel = None
        self.run = None
        self.loss = None
        self.shape = [-1, hidden]
        device, dtype = self._device_and_dtype(device, dtype)
        self.device = None
        self.dtype = None
        self.__run = None

    def __repr__(self):
        return ("Layer_dense_variational: hidden -> {}"
                "activation -> {}"
                "bias -> {}").format(self.hidden, self.activation, self.bias)

    def __call__(self, x_input):
        return self.__run(x_input)

    _device_and_dtype = _device_and_dtype

    def build(self, device=None, dtype=None):
        """Build object

        Build, and initialize layer's weights. Normally, it is done by a Model_base object

        :param device: str of 1, "cpu" or "gpu"
        :param dtype: str of 1, "float32" or "float64"

        """
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        
        self.__w = torch.tensor(np.random.normal(0.0, 0.001, self.shape),
                                dtype=self.dtype,
                                requires_grad=True,
                                device=self.device).to(self.device)
        self.__w_sd = torch.tensor(np.random.normal(0.0, 0.001, self.shape),
                                   dtype=self.dtype,
                                   requires_grad=True,
                                   device=self.device).to(self.device)
        if self.prior_trainable:
            self.__prior_w = torch.tensor(np.random.normal(0.0, 0.001, self.shape[0] * self.shape[1]),
                                          dtype=self.dtype,
                                          requires_grad=True,
                                          device=self.device).to(self.device)

        else:
            prior = torch.distributions.Normal(torch.tensor(0.0, device = self.device, dtype = self.dtype).to(self.device),
                                               torch.tensor(self.prior, device=self.device, dtype=self.dtype).to(self.device))
        if self.bias:
            self.__b = torch.tensor(np.random.normal(0.0, 0.001, [self.shape[1], 1]),
                                  dtype=self.dtype,
                                  requires_grad=True,
                                  device=self.device).to(self.device)
            self.__b_sd = torch.tensor(np.random.normal(0.0, 0.001, [self.shape[1], 1]),
                                       dtype=self.dtype,
                                       requires_grad=True,
                                       device=self.device).to(self.device)
            if self.prior_trainable:
                self.__prior_b = torch.tensor(np.random.normal(0.0, 0.001, [self.shape[1], 1]),
                                                               dtype=self.dtype,
                                                               requires_grad=True,
                                                               device=self.device).to(self.device)
        
        kl_weight = torch.tensor(self.kl_weight, device = self.device, dtype=self.dtype).to(self.device)
        eps = torch.tensor(1e-3, device = self.device, dtype=self.dtype).to(self.device)

        self._kernel = lambda: torch.distributions.Normal(self.__w.reshape([-1, 1]), eps + torch.nn.functional.softplus(self.__w_sd.reshape([-1, 1])))

        if self.bias:
            self._bias_kernel = lambda: torch.distributions.Normal(self.__b.reshape([-1,1]), eps + torch.nn.functional.softplus(self.__b_sd.reshape([-1, 1])))

        if self.activation is None:
            activation = lambda input: input
        if self.activation == "tanh":
            activation = lambda input: torch.tanh(input)
        if self.activation == "relu":
            activation = lambda input: torch.nn.functional.relu(input)
        if self.activation == "sigmoid":
            activation = lambda input: torch.nn.functional.sigmoid(input)

        if self.bias:
            def loss():
                kernel = self._kernel()
                bias_kernel = self._bias_kernel()
                if self.prior_trainable:
                    prior = torch.distributions.Normal(self.__prior_w, torch.tensor(self.prior, device=self.device, dtype=self.dtype).to(self.device))
                    prior_b = torch.distributions.Normal(self.__prior_b, torch.tensor(self.prior, device=self.device, dtype=self.dtype).to(self.device))
                else:
                    prior = torch.distributions.Normal(torch.tensor(0.0, device = self.device, dtype = self.dtype).to(self.device),
                                                       torch.tensor(self.prior, device=self.device, dtype=self.dtype).to(self.device))
                    prior_b = torch.distributions.Normal(torch.tensor(0.0, device = self.device, dtype = self.dtype).to(self.device),
                                                         torch.tensor(self.prior, device=self.device, dtype=self.dtype).to(self.device))
                return kl_weight * (torch.sum(torch.distributions.kl_divergence(kernel, prior)) + torch.sum(torch.distributions.kl_divergence(bias_kernel, prior_b)))
            self.loss = loss

            def run(input):
                kernel = self._kernel()
                bias_kernel = self._bias_kernel()
                return activation(torch.nn.functional.linear(input, kernel.rsample().reshape(self.shape).t(), bias_kernel.rsample().t()))
            self.__run = run
        
        else:
            def loss():
                kernel = self._kernel()
                if self.prior_trainable:
                    prior = torch.distributions.Normal(self.__prior_w, torch.tensor(self.prior, device=self.device, dtype=self.dtype).to(self.device))
                else:
                    prior = torch.distributions.Normal(torch.tensor(0.0, device = self.device, dtype = self.dtype).to(self.device),
                                                       torch.tensor(self.prior, device=self.device, dtype=self.dtype).to(self.device))
           
                return kl_weight * torch.sum(torch.distributions.kl_divergence(kernel, prior))
            self.loss = loss

            def run(input):
                kernel = self._kernel()
                return activation(torch.nn.functional.linear(input, kernel.rsample().reshape(self.shape).t()))
            self.__run = run


    def _set_shape(self, shape):
        """set shape 

        internal: set shape of layer

        """
        self.shape[0] = shape
    
    def get_shape(self):
        """get_shape

        get shape of layer (dimension of weight kernel)

        """
        return self.shape
    
    def set_weights(self, w):
        """set weights

        set layer's weights

        :param w: list of numpy weights, must be same shape as kernel weights

        """
        if self.bias:
            self.__w.data = torch.tensor(w[0], dtype=self.dtype, device=self.device).to(self.device).data
            self.__w_sd.data = torch.tensor(w[1], dtype=self.dtype, device=self.device).to(self.device).data
            self.__b.data = torch.tensor(w[2], dtype=self.dtype, device=self.device).to(self.device).data
            self.__b_sd.data = torch.tensor(w[3], dtype=self.dtype, device=self.device).to(self.device).data
        else:
            self.__w.data = torch.tensor(w[0], dtype=self.dtype, device=self.device).to(self.device).data
            self.__w_sd.data = torch.tensor(w[1], dtype=self.dtype, device=self.device).to(self.device).data
        
        if self.prior_trainable:
            if self.bias:
                self.__prior_w.data = torch.tensor(w[4], dtype=self.dtype, device=self.device).to(self.device).data
                self.__prior_b.data = torch.tensor(w[5], dtype=self.dtype, device=self.device).to(self.device).data
            else:
                self.__prior_w.data = torch.tensor(w[2], dtype=self.dtype, device=self.device).to(self.device).data
    
    def get_weights_numpy(self):
        """get weights

        return layer's weights as list of numpy arrays

        """
        if self.bias:
            tmp = [self.__w.data.cpu().numpy(), torch.nn.functional.softplus(self.__w_sd).data.cpu().numpy(),self.__b.data.cpu().numpy(), torch.nn.functional.softplus(self.__b_sd).data.cpu().numpy()]
        else:
            tmp = [self.__w.data.cpu().numpy(), torch.nn.functional.softplus(self.__w_sd).data.cpu().numpy()]
        
        if self.prior_trainable:
            if self.bias: 
                tmp.extend([self.__prior_w.data.cpu().numpy(), self.__prior_b.data.cpu().numpy()])
            else: 
                tmp.append(self.__prior_w.data.cpu().numpy())

        return tmp
    
    def get_weights(self):
        """get weights (torch)

        return weights as torch tensors

        """
        if self.bias:
            tmp = [self.__w, self.__w_sd, self.__b, self.__b_sd]
        else:
            tmp = [self.__w, self.__w_sd]

        if self.prior_trainable:
            if self.bias: 
                tmp.extend([self.__prior_w, self.__prior_b])
            else:
                tmp.append(self.__prior_w)

        return tmp
    
    def get_loss(self):
        """ get loss

        return layer's losses

        """
        return self.loss