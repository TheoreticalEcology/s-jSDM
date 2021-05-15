import torch
import torch_optimizer as optim
import madgrad

def optimizer_adamax(lr = 0.002,betas = [0.9, 0.999], eps = 1e-08 , weight_decay = 0.0):
    """
    Adamax Optimizer

    # Arguments
    :param lr: float of 1 > 0.0, learning rate
    :param betas: tuple of floats
    :param eps: float of 1 > 0.0
    :param weight_decay: float of 1 > 0.0

    """
    return lambda params: torch.optim.Adamax(params = params, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)

def optimizer_RMSprop(lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
    """
    RMSprop Optimizer

    # Arguments
    :param lr: float of 1 > 0.0, learning rate
    :param alpha: float of 1 > 0.0
    :param eps: float of 1 > 0.0
    :param weight_decay: float of 1 > 0.0
    :param momentum: float of 1 > 0.0
    :param centered: logical of 1

    """
    return lambda params: torch.optim.RMSprop(params = params, lr = lr, alpha=alpha, eps = eps, weight_decay = weight_decay, momentum=momentum, centered=centered)

def optimizer_SGD(lr=1e-2, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    """
    SGD Optimizer

    # Arguments
    :param lr: float of 1 > 0.0, learning rate
    :param momentum: float of 1 > 0.0
    :param dampening: float of 1 > 0.0
    :param eps: float of 1 > 0.0
    :param weight_decay: float of 1 > 0.0
    :param momentum: float of 1 > 0.0
    :param centered: logical of 1

    """
    return lambda params: torch.optim.SGD(params = params, lr = lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)


def optimizer_AccSGD(lr=1e-3,
    kappa=1000.0,
    xi=10.0,
    small_const=0.7,
    weight_decay=0):
    return lambda params: optim.AccSGD(params=params,lr=lr,kappa=kappa,xi=xi,small_const=small_const,weight_decay=weight_decay)

def optimizer_AdaBound(lr= 1e-3,
    betas= (0.9, 0.999),
    final_lr = 0.1,
    gamma=1e-3,
    eps= 1e-8,
    weight_decay=0,
    amsbound=True):
    return lambda params: optim.AdaBound(params=params,lr=lr, betas=betas,final_lr=final_lr,gamma=gamma,eps= eps,weight_decay=weight_decay,amsbound=amsbound)

def optimizer_DiffGrad(lr= 1e-3,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0):
    return lambda params: optim.DiffGrad(params=params,lr=lr, betas=betas,eps=eps,weight_decay=weight_decay)

def optimizer_madgrad(lr=1e-2, momentum=0, weight_decay=0, eps=1e-6):
    return lambda params: madgrad.MADGRAD(params = params, lr = lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
