#' is_torch_available
#' @details check whether torch is available
#' 
#' @return Logical, is torch module available or not.
#' 
#' @export
is_torch_available = function() {
  isTRUE(try(torch::torch_is_installed(), silent = TRUE))
}

createSplit = function(n=NULL,CV=5) {
  set = cut(sample.int(n), breaks = CV, labels = FALSE)
  test_indices = lapply(unique(set), function(s) which(set == s, arr.ind = TRUE))
  return(test_indices)
}


copyRP = function(w) w

# kept as a no-op from the reticulate backend so that existing call sites stay valid
force_r = function(x) x

# a model restored from disk carries dead external pointers; detect that cheaply
model_is_alive = function(m) {
  if (is.null(m)) return(FALSE)
  !inherits(try(m$state_dict(), silent = TRUE), "try-error")
}

# normalises what gets stored in model_properties, so a fit that fell back to the CPU does not
# re-warn on every rebuild. The validation itself lives in sjsdm_device().
check_device = function(device) {
  if (sjsdm_device(device)$type == "cpu") "cpu" else device
}

# predict() has to reuse the fitted design: model.matrix(formula, newdata) re-derives poly(),
# ns() or scale() from the new rows. The terms object carries predvars, which does not.
# One design per block, in the order the net adds them.
sjsdm_newdata = function(config, newdata) {
  if (!is.data.frame(newdata)) newdata = data.frame(newdata)
  lapply(c(list(config), config$nn), function(b)
    stats::model.matrix(stats::delete.response(b$terms), newdata, xlev = b$xlevels))
}

addA = function(col, alpha = 0.25) apply(sapply(col, grDevices::col2rgb)/255, 2, function(x) grDevices::rgb(x[1], x[2], x[3], alpha=alpha))

#' check model
#' check model and rebuild if necessary
#' @param object of class sjSDM
checkModel = function(object) {
  check_module()
  if(!inherits(object, c("sjSDM", "sjSDM_DNN", "sLVM"))) stop("model not of class sjSDM")
  
  if(model_is_alive(object$model)) return(object)
  
  if(!identical(object$version, sjsdm_backend_version))
    stop("this object was fitted by sjSDM backend ",
         if(is.null(object$version)) "< 1.1.0" else object$version,
         " and cannot be restored by backend ", sjsdm_backend_version, call. = FALSE)
  
  object$model = object$get_model()
  object$model$load_state_dict(torch::torch_load(object$state$raw))
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
  if(!is_torch_available())
    stop("'torch' is not available. Run torch::install_torch(), see ?installation_help",
         call. = FALSE)
  invisible(TRUE)
}



parse_nn = function(nn) {
  slices = nn$children
  layers = sapply(slices, function(s) sub("^nn_", "", class(s)[1]))
  txt = paste0("===================================\n")

  wM = matrix(NA, nrow = length(layers), ncol = 2L)

  for(i in seq_along(layers)) {
    if(layers[i] %in% "linear") {
      wM[i, 1] = slices[[i]]$in_features
      wM[i, 2] = slices[[i]]$out_features
      txt = paste0(txt, paste0("Layer_", i), ":",
                   "\t (", slices[[i]]$in_features, ", ", slices[[i]]$out_features, ")\n"
                   )
    } else {
      txt = paste0(txt, paste0("Layer_", i), ":",
                   "\t ", layers[i], "\n"
                   )
    }
  }
  txt = paste0(txt, "===================================\n")

  txt = paste0(txt, "Weights :\t ", sum(apply(wM, 1, cumprod)[2,], na.rm = TRUE), "\n")
  return(txt)
}



#' Generate spatial eigenvectors
#' 
#' Generates a Moran's eigenvector map of the distance matrix. See Dray, Legendre, and Peres-Neto, 2006 for more information.
#' 
#' @param coords matrix or data.frame of coordinates
#' @param threshold ignore distances greater than threshold
#' 
#' @return
#' Matrix of spatial eigenvectors. 
#' 
#' @references Dray, S., Legendre, P., & Peres-Neto, P. R. (2006). Spatial modelling: a comprehensive framework for principal coordinate analysis of neighbour matrices (PCNM). Ecological modelling, 196(3-4), 483-493.
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

softplus = function(x) log(1+exp(x))

check_installation = function() {
  torch_ = c(crayon::red(cli::symbol$cross), 0, "torch")
  if(is_torch_available())
    torch_ = c(crayon::green(cli::symbol$tick), 1,
               paste0("torch ", as.character(utils::packageVersion("torch"))))
  return(rbind("torch" = torch_))
}


zero_like = function(M) {
  return(matrix(0.0, nrow(M), ncol(M)))
}

