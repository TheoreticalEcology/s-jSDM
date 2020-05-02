#' is_torch_available
#' check whetcher torch is available
#' @export
is_torch_available = function() {
  #implementation_module <- resolve_implementation_module()
  if (reticulate::py_module_available("torch")) {
    return(TRUE)
  } else {
    return(FALSE)
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


#' check model
#' check model and rebuild if necessary
#' @param object of class sjSDM
checkModel = function(object) {
  if(!inherits(object, c("sjSDM", "sjSDM_DNN", "sLVM"))) stop("model not of class sjSDM")
  
  if(!reticulate::py_is_null_xptr(object$model)) return(object)
  
  object$model = object$get_model()
  
  if(inherits(object, c("sjSDM", "sjSDM_DNN"))){
    object$model$set_env_weights(lapply(object$weights, function(w) reticulate::r_to_py(w)$copy()))
    if(!is.null(object$spatial)) object$model$set_spatial_weights(lapply(object$spatial_weights, function(w) reticulate::r_to_py(w)$copy()))
    object$model$set_sigma(reticulate::r_to_py(object$sigma)$copy())
  }
  
  if(inherits(object, "sLVM")) {
    unserialize_state(object, object$state)
    object$model$set_posterior_samples(lapply(object$posterior_samples, function(p) torch$tensor(p, dtype=object$model$dtype, device=object$model$device)))
  }
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
  if(!exists("fa")){
    .onLoad()
  }

  if(!exists("fa")){
    stop("PyTorch not installed", call. = FALSE)
  }

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


serialize_state = function(model) {
  tmp = tempfile(pattern = "svi state")
  on.exit(unlink(tmp), add = TRUE)
  model$pyro$get_param_store()$save(tmp)
  return(readBin(tmp, what = "raw", n = file.size(tmp), size=1))
}

unserialize_state = function(model, state) {
  tmp = tempfile(pattern = "svi state")
  on.exit(unlink(tmp), add = TRUE)
  writeBin(state, tmp)
  model$model$pyro$get_param_store()$load(tmp)
}