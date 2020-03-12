#' Configure an object
#' 
#' Finalizes or completes an object.
#' Return coefficients from a fitted sjSDM model
#' 
#' @param object a model created by \code{\link{sjSDM_model}}
#' @param df degree of freedom for covariance matrix
#' @param l1_cov lasso regularization on covariances
#' @param l2_cov ridge regularization on covariances
#' @param optimizer optimizer to use
#' @param ... degree of freedom, l1_cov, l2_cov
#' @export
compile = function(object, df = NULL, l1_cov = 0.0, l2_cov = 0.0, optimizer = NULL, ...) {
  object$build(df = as.integer(df), l1 = l1_cov, l2 = l2_cov, optimizer = optimizer)
  return(invisible(object))
}




#' Adamax optimizer
#' 
#' create instance of adamax optimizer
#' @param learning_rate learning_rate 
#' @param weight_decay weight decay for optimizer
#' @export
optimizer_adamax = function(learning_rate = 0.01, weight_decay = 0.01) {
  return(fa$optimizer_adamax(lr = learning_rate, weight_decay = 0.01))
}


#' RMSprop optimizer
#' 
#' create instance of adamax optimizer
#' @param learning_rate learning_rate 
#' @param weight_decay weight decay for optimizer
#' @param ... additional arguments passed to RMSprop
#' @export
optimizer_RMSprop = function(learning_rate = 0.01, weight_decay = 0.01, ...) {
  return(fa$optimizer_RMSprop(lr = learning_rate, weight_decay = 0.01, ...))
}

#' Fit sjSDM_model
#' 
#' fit sjSDM_model in n epochs
#' @param object a model created by \code{\link{sjSDM_model}}
#' @param X environmental matrix
#' @param Y species occurence matrix
#' @param batch_size batch_size
#' @param epochs number of epochs
#' @param parallel number of cores for data loader
#' @export
fit = function(object, X = NULL, Y = NULL, batch_size = 25L, epochs = 100L, parallel = 0L) {
  object$fit(X, Y, batch_size = as.integer(batch_size), epochs = as.integer(epochs), parallel = parallel)
  return(invisible(object))
}



#' layer_dense
#' 
#' Add a fully connected layer to an output
#' @param object Model object
#' @param units Number of hidden units
#' @param activation Name of activation function, if NULL no activation is applied
#' @param use_bias Use bias or not
#' @param kernel_l1 l1 regularization on the weights
#' @param kernel_l2 l2 regularization on the weights
#' 
#' @seealso \code{\link{sjSDM}}
#' @export
layer_dense = function(object = NULL, units = NULL, activation = NULL, use_bias = FALSE, kernel_l1 = 0.0, kernel_l2 = 0.0) {
  
  stopifnot(
    units > 0,
    kernel_l1 >= 0.0,
    kernel_l2 >= 0.0,
    is.logical(use_bias),
    inherits(object, "sjSDM_model")
  )
  
  object$add_layer(fa$layers$Layer_dense(hidden = as.integer(units), 
                                         activation = activation, 
                                         bias = use_bias, 
                                         l1 = kernel_l1, 
                                         l2 = kernel_l2, 
                                         device = object$device, 
                                         dtype = object$dtype))
  return(invisible(object))
}



