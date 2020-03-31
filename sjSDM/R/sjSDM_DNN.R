#' Return summary of a fitted sjSDM DNN
#' 
#' @param object a model fitted by \code{\link{sjSDM}} with \code{\link{envDNN}} object
#' @param ... optional arguments for compatibility with the generic function, no functionality implemented
#' @export
summary.sjSDM_DNN = function(object, ...) {
  summary(object$model)
}


#' Print a fitted sjSDM DNN
#' 
#' @param x a moel fitted by \code{\link{sjSDM}} with \code{\link{envDNN}} object
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
print.sjSDM_DNN = function(x, ...) {
  print(x$model)
}




#' Plot training history
#' 
#' @param x a model fitted by \code{\link{sjSDM}} with \code{\link{envDNN}} object
#' @param y unused argument
#' @param ... Additional arguments to pass to \code{plot()}
#' @export
plot.sjSDM_DNN = function(x, y, ...) {
  plot.sjSDM_model(x$model)
}


#' Predict from a fitted sjSDM DNN
#' 
#' @param object a model fitted by \code{\link{sjSDM}} with \code{\link{envDNN}} object
#' @param newdata newdata for predictions
#' @param ... optional arguments for compatibility with the generic function, no function implemented
#' @export
predict.sjSDM_DNN = function(object, newdata = NULL, ...) {
  object = checkModel(object)
  if(is.null(newdata)) {
    return(object$model$predict(newdata = object$data$X))
  } else {
    if(is.data.frame(newdata)) {
      newdata = stats::model.matrix(object$formula, newdata)
    } else {
      newdata = stats::model.matrix(object$formula, data.frame(newdata))
    }
  }
  pred = object$model$predict(newdata = newdata, ...)
  return(pred)
}

