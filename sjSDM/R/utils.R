#' is_torch_available
#' check whetcher torch is available
#' @export
is_torch_available = function() {
  #implementation_module <- resolve_implementation_module()
  conda = try({ reticulate::conda_binary() }, silent=TRUE)

  if(inherits(conda, "try-error")) return(FALSE)

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

createSplit = function(n=NULL,CV=5) {
  set = cut(sample.int(n), breaks = CV, labels = FALSE)
  test_indices = lapply(unique(set), function(s) which(set == s, arr.ind = TRUE))
  return(test_indices)
}


copyRP = function(w) reticulate::r_to_py(w)$copy()

addA = function(col, alpha = 0.25) apply(sapply(col, grDevices::col2rgb)/255, 2, function(x) grDevices::rgb(x[1], x[2], x[3], alpha=alpha))

#' check model
#' check model and rebuild if necessary
#' @param object of class sjSDM
checkModel = function(object) {
  check_module()
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
    type = strsplit(class(slices[[i]]), ".", fixed = TRUE)[[1]]
    
    if(layers[i] == "Linear") {
      wM[i, 1] = slices[[i]]$in_features
      wM[i, 2] = slices[[i]]$out_features
      txt = paste0(txt, paste0("Layer_", i),":",
                   "\t (", slices[[i]]$in_features, ", ",slices[[i]]$out_features, ")\n"
                   )
    } else {
      txt = paste0(txt, paste0("Layer_", i),":",
                   "\t ", layers[i], "\n"
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


#' Generate spatial eigenvectors
#' 
#' function to generate spatial eigenvectors to account for spatial autocorrelation
#' @param coords matrix or data.frame of coordinates
#' @param threshold ignore distances greater than threshold
#' 
#' @export

generateSpatialEV = function(coords = NULL, threshold = 0.0) {
  ## create dist ##
  dist = as.matrix(stats::dist(coords))
  zero = diag(0.0, ncol(dist))
  
  ## create weights ##
  if (threshold > 0) dist[dist < distance.threshold] = 0
  
  distW = 1/dist
  distW[is.infinite(distW)] = 1
  diag(distW) <- 0
  rowSW =  rowSums(distW)
  rowSW[rowSW == 0] = 1
  distW <- distW/rowSW
  
  ## scale ##
  rowM = zero + rowMeans(distW)
  colM = t(zero + colMeans(distW))
  distC = distW - rowM - colM + mean(distW)
  
  eigV = eigen(distC, symmetric = TRUE)
  values = eigV$values / max(abs(eigV$values))
  SV = eigV$vectors[, values>0]
  colnames(SV) = paste0("SE_", 1:ncol(SV))
  return(SV)
}