#' sjSDM_model
#' 
#' Create a new sjSDM_model
#' @param input_shape number of predictors
#' @param device which device to be used, "cpu" or "gpu"
#' @param dtype which data type, most GPU support only 32 bit floats.
#' 
#' @example /inst/examples/sjSDM_model-example.R
#' @seealso \code{\link{layer_dense}},  \code{\link{compile}},  \code{\link{fit}},  \code{\link{predict.sjSDM_model}},  \code{\link{summary.sjSDM_model}}, \code{\link{getWeights}}, \code{\link{setWeights}}, , \code{\link{plot.sjSDM_model}}
#' @export
sjSDM_model = function(input_shape,  device = NULL, dtype = "float32") {
  
  if(reticulate::py_is_null_xptr(fa)) .onLoad()
  if(is.null(device)) {
    if(torch$cuda$is_available()) device = 0L
    else device = "cpu"
  }
  if(is.numeric(device)) device = as.integer(device)
  if(device == "gpu") device = 0L
  
  object = fa$Model_base(as.integer(input_shape), device = device, dtype = dtype)
  class(object) = c("sjSDM_model",class(object))
  return(object)
}

#' Return summary of a fitted sjSDM model
#' 
#' @param object a model fitted by \code{\link{sjSDM_model}}
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' @export
summary.sjSDM_model = function(object, ...) {
  shapes = lapply(object$layers, function(l) l$shape)
  cat("Model architecture\n")
  cat("=======================\n")
  for(i in 1:length(shapes)) {
    cat("Layer_dense_", i,":\t (", shapes[[i]][1],", ", shapes[[i]][2], ")\n")
  }
  cat("=======================\n")
  wN = sum(sapply(shapes, function(s) cumprod(s)[2]))
  cat("Weights (w/o sigma):\t ", wN, "\n")
  cat("Weights (w sigma):\t ", wN + object$df*shapes[[length(shapes)]][2], "\n")
}

#' Print a fitted sjSDM model
#' 
#' @param x a moel fitted by \code{\link{sjSDM_model}}
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.sjSDM_model = function(x, ...) {
  cat("sjSDM_model, see summary(model) for details \n")
}



#' Predict from a fitted sjSDM_model 
#' 
#' @param object a model fitted by \code{\link{sjSDM_model}}
#' @param newdata newdata for predictions
#' @param ... optional arguments such as batch_size
#' @export
predict.sjSDM_model = function(object, newdata = NULL, ...) {
  pred = object$predict(newdata = newdata, ...)
  return(pred)
}

#' Plot training history
#' 
#' @param x
#' @param y unused argument
#' @param ... Additional arguments to pass to \code{\link{plot()}}
#' @export
plot.sjSDM_model = function(x, y, ...) {
  hist = x$history
  plot(y = hist, x = 1:length(hist), xlab = "Epochs", ylab = "Loss", main = "Training history", type = "o",...)
  return(invisible(hist))
}

#' Get weights
#' 
#' return weights of each layer
#' @param object object of class \code{\link{sjSDM_model}}
#' @return 
#' \itemize{
#'  \item layers - list of layer weights
#'  \item sigma - weight to construct covariance matrix
#' }
#' @export
getWeights = function(object) {
  stopifnot(inherits(object, "sjSDM_model"))
  return(list(layers = object$weights_numpy, sigma = object$sigma_numpy))
}


#' Set weights
#' 
#' set layer weights and sigma in \code{\link{sjSDM_model}} object
#' @param object object of class \code{\link{sjSDM_model}}
#' @param weights list of layer weights and sigma, see \code{\link{getWeights}}
#' @export
setWeights = function(object, weights) {
  stopifnot(inherits(object, "sjSDM_model"))
  object$set_weights(weights$layers)
  object$set_sigma(weights$sigma)
  return(invisible(object))
}

