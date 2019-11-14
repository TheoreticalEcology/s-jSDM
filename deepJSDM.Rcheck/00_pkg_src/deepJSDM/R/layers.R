
#' add_dense
#' add dense layer function
#' @param model model
#' @param hidden hidden units
#' @param activation activation function
#' @param bias use bias or not
#' @param l1 l1 regularizer
#' @param l2 l2 regularizer
#' @export

layer_dense = function(model, hidden = 10L, activation = "relu", bias = TRUE, l1 = 0.0, l2 = 0.0) {

  ####
  hidden = as.integer(abs(hidden))
  ####

  if(model$previous == -1L) shape = c(model$shapes[1], hidden)
  else shape = c(model$previous, hidden)
  layer_env = list(hidden = hidden, activation = activation, bias = bias, shape = shape, l1 = l1, l2 = l2)
  layer_expr = rlang::expr({
    w = .torch$tensor(matrix(rnorm(shape[1]*shape[2], 0, 0.001), shape[1], shape[2]),dtype = .dtype, requires_grad = TRUE,device = .device)$to(.device)
    l1 = .torch$tensor(l1, dtype = .dtype, device = .device)$to(.device)
    l2 = .torch$tensor(l2, dtype = .dtype, device = .device)$to(.device)
    if(bias) B = .torch$tensor(matrix(rnorm(shape[2], 0, 0.001), shape[2],1), dtype = .dtype, requires_grad = TRUE,device = .device)$to(.device)
    else B = NULL
    returnList =
      if(!is.null(activation) && activation == "relu")
        list(weights = list(w = w,B = B), loss_env = list(l1 = l1, l2 = l2),
             feedforward = rlang::expr(function(X, W, train) {
                if(!is.null(W[[2]])) .torch$nn$functional$relu(.torch$nn$functional$linear(X, W[[1]]$t(), W[[2]]$t()))
                else .torch$nn$functional$relu(.torch$nn$functional$linear(X, W[[1]]$t()))
               }),
             loss = rlang::expr(function() return(.torch$sum(l2 * w * w) + .torch$sum(l1 * .torch$abs(w)))))
      else list(weights = list(w = w, B = B),  loss_env = list(l1 = l1, l2 = l2),
              feedforward = rlang::expr(function(X, W, train) {
      if(!is.null(W[[2]])) return(.torch$nn$functional$linear(X, W[[1]]$t(), W[[2]]$t()))
      else return(.torch$nn$functional$linear(X, W[[1]]$t()))
    }),
    loss = rlang::expr(function() return(.torch$sum(l2 * w * w) + .torch$sum(l1 * .torch$abs(w)))))
    returnList
  })
  layer = list(layer_expr = layer_expr, layer_env = layer_env, call_params = list())
  model$evals[[length(model$evals) + 1L]] = layer
  model$previous = hidden
  model$layer = model$layer + 1
  return(model)
}

#' add_varational_dense function
#' @param model model
#' @param hidden hidden units
#' @param activation activation function
#' @param sd sd for gaussian prior
#' @param kl_weight kl_weight for prior
#' @export

layer_varational_dense = function(model, hidden = 10L, activation = "relu", sd = 20.0, kl_weight = 0.1) {

  ####
  hidden = as.integer(abs(hidden))
  ####

  if(model$previous == -1L) shape = c(model$shapes[1], hidden)
  else shape = c(model$previous, hidden)
  layer_env = list(hidden = hidden, activation = activation, shape = shape, sd = sd, kl_weight = kl_weight)

  layer_expr = rlang::expr({
    zeros = .torch$zeros(as.integer(c(shape[1], shape[2])),dtype = .dtype, device = .device)$to(.device)
    ones = .torch$tensor(matrix(sd, shape[1], shape[2]),dtype = .dtype, device = .device)$to(.device)
    prior = .torch$distributions$Normal(zeros, ones)
    one = .torch$tensor(1.0)
    # precision = function(a) {
    #   return(.torch$tensor(0.0001) + .torch$nn$functional$softplus(a))
    # }
    w = .torch$tensor(matrix(rnorm(shape[1]*shape[2], 0, 0.001), shape[1], shape[2]),dtype = .dtype, requires_grad = TRUE,device = .device)$to(.device)
    sd = .torch$tensor(exp(matrix(rnorm(shape[1]*shape[2], 0, 0.001), shape[1], shape[2])),dtype = .dtype, requires_grad = TRUE,device = .device)$to(.device)
    kl_weight = .torch$tensor(kl_weight, device = .device, dtype = .dtype)$to(.device)
    #W = .torch$distributions$Normal(W[[1]], precision(W[[2]]))$sample()
    #if(bias) B = .torch$tensor(matrix(rnorm(shape[2], 0, 0.001), shape[2],1), dtype = .dtype, requires_grad = TRUE,device = .device)$to(.device)
    #else B = NULLI
    returnList =
      list(weights = list(w = w, sd = sd),
           loss_env = list(prior = prior, kl_weight = kl_weight),
           feedforward = rlang::expr({
             precision = function(a) {
               return(.torch$tensor(0.0001) + .torch$nn$functional$softplus(a))
             }
              function(X, W, train) {
                wr = .torch$distributions$Normal(W[[1]], precision(W[[2]]))$sample()
                .torch$nn$functional$linear(X, wr$t())
                }}),
           loss = rlang::expr({
             precision = function(a) {
               return(.torch$tensor(0.0001) + .torch$nn$functional$softplus(a))
             }
             function() {
               wr = .torch$distributions$Normal(w, precision(sd))
               return(.torch$sum(.torch$distributions$kl_divergence(wr, prior)) * kl_weight)
             }
           }))
    returnList
  })
  layer = list(layer_expr = layer_expr, layer_env = layer_env, call_params = list(), loss = function(W) {
    wr = .torch$distributions$Normal(W[[1]], precision(W[[2]]))

  })
  model$evals[[length(model$evals) + 1L]] = layer
  model$previous = hidden
  model$layer = model$layer + 1
  return(model)
}





#' dropout layer
#' @param model model
#' @param rate dropout rate
#' @export
layer_dropout = function(model, rate) {
  layer_env = list(rate = rate)
  layer_expr = rlang::expr({
    returnList = list(weights = list(),feedforward = rlang::expr(function(X, W, train, rate) .torch$nn$functional$dropout(X, p = rate,training = train)))
  })
  layer= list(layer_expr = layer_expr, layer_env = layer_env, call_params = list(rate = rate))
  model$evals[[length(model$evals) + 1L]] = layer
  model$layer = model$layer + 1
  return(model)
}

