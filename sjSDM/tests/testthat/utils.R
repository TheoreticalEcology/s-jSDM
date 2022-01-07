skip_if_no_torch = function(required_version = NULL) {
  if (!is_torch_available())
    skip("required torch version not available for testing")
}

force_r = function(x) {
  if(inherits(x, "python.builtin.object")) return(reticulate::py_to_r( x ))
  else return(x)
}

is_gpu_available = function() {
  if( force_r(pkg.env$torch$cuda$is_available()) ) return("gpu")
  else return("cpu")
}