#' is_torch_available
#' check whetcher torch is available
#' @export
is_torch_available = function() {
  #implementation_module <- resolve_implementation_module()
  if (reticulate::py_module_available("torch")) {
    TRUE
  } else {
    FALSE
  }
}

#' is_sjSDM_py_available
#' check whetcher torch is available
is_sjSDM_py_available = function() {
  #implementation_module <- resolve_implementation_module()
  if (reticulate::py_module_available("sjSDM_py")) {
    TRUE
  } else {
    FALSE
  }
}


#' useGPU
#' use a specific gpu
#' @param device number
#' @export
useGPU = function(device = 0) {
  if(!torch$cuda$is_available()) stop("Cuda/gpu not available...")
  device <<- torch$device(paste0("cuda:",device))
}

#' useCPU
#' use CPU
#' @export
useCPU = function(){
  device <<- torch$device("cpu")
}

#' gpuInfo
#' list gpu infos
#' @export
gpuInfo = function(){
  print(torch$cuda$get_device_properties(device))
}


#' check model
#' check model and rebuild if necessary
#' @param object of class sjSDM
checkModel = function(object) {
  if(!inherits(object, "sjSDM")) stop("model not of class sjSDM")
  
  if(!reticulate::py_is_null_xptr(object$model)) return(object)
  
  object$model = object$get_model()
  
  object$model$set_weights(object$weights)
  object$model$set_sigma(object$sigma)
  return(object)
}


