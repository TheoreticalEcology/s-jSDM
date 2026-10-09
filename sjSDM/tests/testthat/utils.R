skip_if_no_torch = function(required_version = NULL) {
  if (!is_torch_available())
    skip("torch is not available for testing")
}

force_r = function(x) x

is_gpu_available = function() {
  if( torch::cuda_is_available() ) return("gpu")
  else return("cpu")
}
