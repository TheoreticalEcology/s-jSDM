#' is_torch_available
#' @details check whether torch is available
#' 
#' @return Logical, is torch module available or not.
#' 
#' @export
is_torch_available = function() {

  if (reticulate::py_module_available("torch")) {
    return(TRUE)
  } else {
    return(FALSE)
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

#' check module
#' 
#' check if module is loaded
check_module = function(){
  if(is.null(pkg.env$fa)){
    .onLoad()
  }

  if(is.null(pkg.env$fa)) {
    stop("PyTorch not installed", call. = FALSE)
  }

  if(reticulate::py_is_null_xptr(pkg.env$fa)) .onLoad()
}



parse_nn = function(nn) {
  slices = reticulate::iterate(nn)
  
  layers = sapply(slices, function(s) {sl = strsplit(class(s)[1], ".", fixed=TRUE)[[1]]; return(sl[length(sl)])})
  txt = paste0("===================================\n")
  
  wM = matrix(NA, nrow = length(layers), ncol= 2L)
  
  for(i in 1:length(layers)) {
    type = strsplit(class(slices[[i]]), ".", fixed = TRUE)[[1]]
    
    if(layers[i] %in% "Linear") {
      wM[i, 1] = force_r( slices[[i]]$in_features )
      wM[i, 2] = force_r( slices[[i]]$out_features )
      txt = paste0(txt, paste0("Layer_", i),":",
                   "\t (", force_r( slices[[i]]$in_features ), ", ",force_r( slices[[i]]$out_features ), ")\n"
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



#' Generate spatial eigenvectors
#' 
#' function to generate spatial eigenvectors to account for spatial autocorrelation
#' @param coords matrix or data.frame of coordinates
#' @param threshold ignore distances greater than threshold
#' 
#' @return
#' Matrix of spatial eigenvectors. 
#' 
#' @export

generateSpatialEV = function(coords = NULL, threshold = 0.0) {
  ## create dist ##
  dist = as.matrix(stats::dist(coords))
  zero = diag(0.0, ncol(dist))
  
  ## create weights ##
  if (threshold > 0) dist[dist < threshold] = 0
  
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

force_r = function(x) {
  if(inherits(x, "python.builtin.object")) return(reticulate::py_to_r( x ))
  else return(x)
}

check_installation = function() {
  # check if dependencies are installed
  torch_ = pyro_ = torch_optimizer_ = madgrad_ = c(crayon::red(cli::symbol$cross), 0)
  if(reticulate::py_module_available("torch")) torch_ =  c(crayon::green(cli::symbol$tick), 1)
  if(reticulate::py_module_available("pyro")) pyro_ =  c(crayon::green(cli::symbol$tick), 1)
  if(reticulate::py_module_available("torch_optimizer")) torch_optimizer_ =  c(crayon::green(cli::symbol$tick), 1)
  if(reticulate::py_module_available("madgrad")) madgrad_ =  c(crayon::green(cli::symbol$tick), 1)
  return(rbind("torch" = torch_,  "torch_optimizer" = torch_optimizer_, "pyro" = pyro_, "madgrad" = madgrad_))
}
