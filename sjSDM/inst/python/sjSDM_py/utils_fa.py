import torch
import numpy as np
from typing import Optional, List


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

def covariance(X):
    # taken from https://github.com/pytorch/pytorch/issues/19037#issuecomment-739002393 
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1/(D-1) * X @ X.transpose(-1, -2)  

def importance(beta: np.ndarray, 
               covX: np.ndarray, 
               sigma: np.ndarray, 
               precision: torch.dtype = torch.float32, 
               covSP: Optional[np.ndarray] = None, 
               betaSP: Optional[np.ndarray] = None) -> dict:
    """Variation partitioning

    Args:
        beta (np.ndarray): environmental coefficients
        covX (np.ndarray): covariance matrix of predictors
        sigma (np.ndarray): square-root matrix of species associations
        precision (torch.dtype, optional): Float or doubles. Defaults to torch.float32.
        covSP (Optional[np.ndarray], optional): covariance matrix of spatial predictors. Defaults to None.
        betaSP (Optional[np.ndarray], optional): spatial coefficients. Defaults to None.

    Returns:
        dict: importances for each functional group
    """               
    beta = torch.tensor(beta.copy(), dtype=precision, device=torch.device("cpu"))
    sigma = torch.tensor(sigma.copy(), dtype=precision, device=torch.device("cpu"))
    sigmaBeta = torch.tensor(covX.copy(), dtype=precision, device=torch.device("cpu"))
    association = sigma.matmul(sigma.t()).add( torch.diag(torch.ones(sigma.shape[0],dtype=precision, device=torch.device("cpu") )) )

    betaCorrected = sigmaBeta.t().matmul(beta)
    Xtotal = torch.einsum("ej, ej -> j", beta, betaCorrected)
    Xsplit = torch.einsum("ej, ej -> je", beta, betaCorrected)

    PredRandom = (association.sum(dim=0) - association.diag()).abs() / association.shape[0]

    if type(betaSP) is np.ndarray:
        betaSP = torch.tensor(betaSP.copy(), dtype=precision, device=torch.device("cpu"))
        sigmaBetaSP = torch.tensor(covSP.copy(), dtype=precision, device=torch.device("cpu"))
        betaSPCorrected = sigmaBetaSP.t().matmul(betaSP)
        SPtotal = torch.einsum("ej, ej -> j", betaSP, betaSPCorrected)
        SPsplit = torch.einsum("ej, ej -> je", betaSP, betaSPCorrected)

        variTotal = Xtotal + PredRandom + SPtotal
        variPartX = Xtotal/variTotal
        variPartSP = SPtotal/variTotal
        variPartRandom = PredRandom/variTotal
        nGroups = Xsplit.shape[1]
        nSP = SPsplit.shape[1]
        variPartXSplit = Xsplit/torch.repeat_interleave(Xsplit.sum(1), nGroups).reshape((Xsplit.shape[0], -1))
        variPartSPSplit = SPsplit/torch.repeat_interleave(SPsplit.sum(1), nSP).reshape((SPsplit.shape[0], -1))
        res = {
            "env" : (variPartX.repeat_interleave(nGroups).reshape([-1, nGroups]) * variPartXSplit).data.cpu().numpy(),
            "spatial" : (variPartSP.repeat_interleave(nSP).reshape([-1, nSP]) * variPartSPSplit).data.cpu().numpy(),
            "biotic" : (variPartRandom).data.cpu().numpy()
        }
    else:
        variTotal = Xtotal + PredRandom
        variPartX = Xtotal/variTotal
        variPartRandom = PredRandom/variTotal
        nGroups = Xsplit.shape[1]
        variPartXSplit = Xsplit/torch.repeat_interleave(Xsplit.sum(1), nGroups).reshape((Xsplit.shape[0], -1))
        res = {
            "env" : (variPartX.repeat_interleave(nGroups).reshape([-1, nGroups]) * variPartXSplit).data.cpu().numpy(),
            "biotic" : (variPartRandom).data.cpu().numpy()
        }
    return res






