import numpy as np
import torch
from ..utils_fa import _device_and_dtype


class Layer_dense:
    """Layer_dense object

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
    def __init__(self, hidden=None, activation=None, bias=False, l1=0.0, l2=0.0, device="cpu", dtype="float32"):
        self.hidden = hidden
        self.activation = activation
        self.w = None
        self.b = None
        self.bias = bias
        self.run = None
        self.l1 = l1
        self.l2 = l2
        self.loss = None
        self.shape = [-1, hidden]
        device, dtype = self._device_and_dtype(device, dtype)
        self.device = device
        self.dtype = dtype
        self.__run = None

    def __repr__(self):
        return ("Layer_dense: hidden -> {}"
                "activation -> {}"
                "bias -> {}"
                "l1 -> {}"
                "l2 -> {}").format(self.hidden, self.activation, self.bias, self.l1, self.l2)

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
        
        self.w = torch.tensor(np.random.normal(0.0, 0.001, self.shape),
                              dtype=self.dtype,
                              requires_grad=True,
                              device=self.device).to(self.device)
        if self.bias:
            self.b = torch.tensor(np.random.normal(0.0, 0.001, [self.shape[1], 1]),
                                  dtype=self.dtype,
                                  requires_grad=True,
                                  device=self.device).to(self.device)
        
        if self.l1 > 0 or self.l2 > 0:
            l1_t = torch.tensor(self.l1, dtype=self.dtype, device=self.device).to(self.device)
            l2_t = torch.tensor(self.l2, dtype=self.dtype, device=self.device).to(self.device)
            #self.loss = lambda: torch.add(torch.sum(l2_t * self.w * self.w), 
            #                              torch.sum(l1_t * torch.abs(self.w)))
            self.loss = lambda: self.w.pow(2.0).sum().mul(l2_t).add(self.w.abs().sum().mul(l1_t))
        
        if self.activation is None:
            activation = lambda input: input
        if self.activation == "tanh":
            activation = lambda input: torch.tanh(input)
        if self.activation == "relu":
            activation = lambda input: torch.nn.functional.relu(input)
        if self.activation == "sigmoid":
            activation = lambda input: torch.nn.functional.sigmoid(input)

        if self.bias:
            self.__run = lambda input: activation(torch.nn.functional.linear(input, self.w.t(), self.b.t()))
        else:
            self.__run = lambda input: activation(torch.nn.functional.linear(input, self.w.t()))
    
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
        self.w.data = torch.tensor(w[0], dtype=self.dtype, device=self.device).to(self.device).data
        if self.bias:
            self.b.data = torch.tensor(w[1], dtype=self.dtype, device=self.device).to(self.device).data
    
    def get_weights_numpy(self):
        """get weights

        return layer's weights as list of numpy arrays

        """
        if self.bias:
            return [self.w.data.cpu().numpy(), self.b.data.cpu().numpy()]
        return [self.w.data.cpu().numpy()]
    
    def get_weights(self):
        """get weights (torch)

        return weights as torch tensors

        """
        if self.bias:
            return [self.w, self.b]
        return [self.w]
    
    def get_loss(self):
        """ get loss

        return layer's losses

        """
        return self.loss
