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
  if(!inherits(object, c("sjSDM", "sjSDM_DNN"))) stop("model not of class sjSDM")
  
  if(!reticulate::py_is_null_xptr(object$model)) return(object)
  
  object$model = object$get_model()
  
  object$model$set_weights(object$weights)
  object$model$set_sigma(object$sigma)
  return(object)
}


is_windows = function() {
  identical(.Platform$OS.type, "windows")
}

is_unix = function() {
  identical(.Platform$OS.type, "unix")
}

is_osx = function() {
  Sys.info()["sysname"] == "Darwin"
}

is_linux = function() {
  identical(tolower(Sys.info()[["sysname"]]), "linux")
}

#' check modul
#' check if modul is loaded
check_module = function(){
  if(reticulate::py_is_null_xptr(fa)) .onLoad()
}



parse_nn = function(nn) {
  slices = reticulate::iterate(nn)
  
  layers = sapply(slices, function(s) {sl = strsplit(class(s)[1], ".", fixed=TRUE)[[1]]; return(sl[length(sl)])})
  txt = paste0("===================================\n")
  
  wM = matrix(NA, nrow = length(layers), ncol= 2L)
  
  for(i in 1:length(layers)) {
    if(layers[i] == "Linear") {
      wM[i, 1] = slices[[i]]$in_features
      wM[i, 2] = slices[[i]]$out_features
      txt = paste0(txt, 
                   "Dense:\t\t (", slices[[i]]$in_features, ", ",slices[[i]]$out_features, ")\n"
                   )
    } else {
      txt = paste0(txt,
                   "Activation:\t ", layers[i], "\n"
                   )
    }
  }
  txt = paste0(txt, "===================================\n")
  
  txt = paste0(txt, "Weights :\t ", sum(apply(wM, 1,cumprod)[2,], na.rm = TRUE), "\n")
  return(txt)
}

#' @importFrom magrittr %>%
#' @export
magrittr::`%>%`