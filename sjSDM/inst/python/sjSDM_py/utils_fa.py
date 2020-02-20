import torch

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
