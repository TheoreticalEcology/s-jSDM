#' model class
#'
#' @param X env matrix
#' @param Y occ matrix
#' @param Traits trait matrix
#' @export

createModel = function(X = NULL, Y = NULL, Traits = NULL){
  out = list()
  out$weights = list()
  out$feedforward = list()
  out$X = X
  out$Y = Y
  out$Traits = Traits
  out$layer = 0
  out$shapes = c(ncol(X), ncol(Y))
  out$previous = -1
  out$rawSigma = NULL
  out$optimizer = NULL
  out$compiled = FALSE
  out$layers = list()
  out$params  = list()
  out$evals = list()
  out$raw_weights = list()
  out$init = FALSE
  out$previous = -1L
  class(out) = "deepJmodel"
  return(out)
}


#' compileModel
#' compile model
#'
#' @param model model of class deepJmodel
#' @param nLatent number of latent dimension
#' @param lr learning rate for optimizer
#' @param optimizer optimizer, adamax or LBFGS
#' @param reset reset weights
#' @param control control options for optimizer "adamax" or "LBFGS"
#' @param l1 l1 on covariances
#' @param l2 l2 on covariances
#' @export
compileModel = function(model, nLatent = 5L, lr = 0.001, optimizer = "adamax", reset = TRUE, control = list(),l1 = 0.0, l2 = 0.0, device = .device, dtype = .dtype) {
  if(!is.null(model$params$nLatent)) nLatent = model$params$nLatent
  n_latent = nLatent
  model$params$nLatent = nLatent
  model$losses = list()

  ## build model
  model$weights = list()
  model$feedforward = vector("list", length(model$evals))
  for(i in 1:length(model$evals)){
    tmp = rlang::eval_tidy(
      model$evals[[i]]$layer_expr,
      data = model$evals[[i]]$layer_env,
      env = rlang::current_env())
    model$weights[[length(model$weights) + 1]] = tmp$weights
    model$feedforward[[i]]$f = eval(tmp$feedforward)
    model$feedforward[[i]]$p = model$evals[[i]]$call_params
    if(!is.null(tmp$loss)) model$losses[[length(model$losses) + i]] = rlang::eval_tidy(tmp$loss,c(tmp$loss_env, tmp$weights))
    names(model$weights[[length(model$weights)]] ) = NULL
  }
  if(!length(model$losses) > 0)model$losses = NULL
  if((length(model$raw_weights) > 0) && reset) model = assignWeights(model)
  

  
  
  ## cov and optimizer
  r_dim = ncol(model$Y)
  model$rawSigma =
    .torch$tensor(
      matrix(runif(as.integer(r_dim*n_latent),-sqrt(6.0/(r_dim+n_latent)), sqrt(6.0/(r_dim+n_latent))),r_dim,n_latent),
      dtype = .dtype,
      requires_grad = TRUE,
      device = .device
    )$to(.device)

  if(!reset){
    if(length(model$raw_weights) != 0) {
      model = assignWeights(model)
    }
  }

  if(!is.null(model$raw_sigma)) {
    model$rawSigma$data = .torch$tensor(model$raw_sigma, dtype = .dtype, device = .device)$to(.device)$data
  }
  
  
  if(l1 > 0.0) {
    l1 = .torch$tensor(l1, device = .device, dtype = .dtype)$to(.device)
    model$losses[[length(model$losses) + 1]] = function() {
      ss = .torch$matmul(model$rawSigma, model$rawSigma$t())
      .torch$mul(l1,  .torch$sum(.torch$abs(.torch$triu(ss, 1L))))
    }
  }
  
  if(l2 > 0.0) {
    l2 = .torch$tensor(l2, device = .device, dtype = .dtype)$to(.device)
    model$losses[[length(model$losses) + 1]] = function() {
      ss = .torch$matmul(model$rawSigma, model$rawSigma$t())
      .torch$mul(l2,  .torch$sum(.torch$pow(.torch$triu(ss, 1L), 2.0)))
    }
  }
    
  model$losses[sapply(model$losses, is.null)] <- NULL
  

  model$nLatent = nLatent

  ## TO DO: DIRTY!!!
  if(length(control) > 0) {
    opt = switch(optimizer,
                 adamax = .torch$optim$Adamax,
                 LBFGS = function(params, lr) return(function(params, lr) do.call(.torch$optim$LBFGS, c(list(params = params, lr = lr), control))))
    if(optimizer == "adamax") pars = c(list(params = c(unlist(model$weights), model$rawSigma),lr = lr), control)
    else pars = c(list(params = c(unlist(model$weights), model$rawSigma),lr = lr))
  }
  else {
    opt = switch(optimizer,
                 adamax = .torch$optim$Adamax,
                 LBFGS = function(params, lr) return(function(params, lr) do.call(.torch$optim$LBFGS, list(params = params, lr = lr))))
    pars = list(params = c(unlist(model$weights), model$rawSigma),lr = lr)
  }
  model$optimizer = do.call(opt, pars)
  model$optimizer_info = optimizer



  if(!is.null(model$optimizer_state)) {
    model$optimizer$load_state_dict(model$optimizer_state)
  }
  model$params$lr = lr
  model$sigma = function() return(.torch$matmul(model$rawSigma, model$rawSigma$t())$data$cpu()$numpy())
  model$compiled = TRUE
  model$init = TRUE
  return(model)
}

#' extractWeights
#' extract weights func
#' @param model model deepJSDM
extractWeights = function(model) {
  w = vector("list", length(model$weights))
  for(i in 1:length(model$weights)){
    w[[i]]$w = vector("list", length(model$weights[[i]]))
    if(length(model$weights[[i]]) > 0) {
      for(j in 1:length(model$weights[[i]])) {
        if(!is.null(model$weights[[i]][[j]])) w[[i]]$w[[j]] = model$weights[[i]][[j]]$data$cpu()$numpy()
      }
    }
  }
  return(w)
}

#' assignWeights
#' assign weights
#' @param model model deepJSDM
assignWeights = function(model, device = .device, dtype = .dtype) {
  for(i in 1:length(model$weights)){
    for(j in 1:length(model$weights[[i]])){
      if(!is.null(model$weights[[i]][[j]])) model$weights[[i]][[j]]$data = .torch$tensor(model$raw_weights[[i]]$w[[j]], device =.device, dtype = .dtype)$to(.device)$data
    }
  }
  return(model)
}


#' @export
predict.deepJmodel = function(model, newdata = NULL, train = FALSE, batch_size = NULL, device = .device, dtype = .dtype){
  if(is.null(newdata)) result = model$X
  else result = newdata

  # check if compiled/initialized -> workaround with optimizer
  if(length(names(model$optimizer)) < 2) model = compileModel(model)

  if(!.torch$is_tensor(result)) result = .torch$tensor(result, dtype = .dtype, device = .device)$to(.device)

  if(is.null(batch_size)) batch_size = as.integer(.torch$as_tensor(result$shape)$data$cpu()$numpy()[1])

  n_latent = as.integer(model$params$nLatent)
  eps = .torch$tensor(0.00001, dtype = .dtype)$to(.device)
  zero = .torch$tensor(0.0, dtype = .dtype)$to(.device)
  one = .torch$tensor(1.0, dtype = .dtype)$to(.device)
  alpha = .torch$tensor(1.70169, dtype = .dtype)$to(.device)
  half = .torch$tensor(0.5, dtype = .dtype)$to(.device)

  for(i in 1:model$layer){
    result = do.call(model$feedforward[[i]]$f,
                     c(X = result,
                       W = list(model$weights[[i]]),
                       train = train,
                       model$feedforward[[i]]$p))
  }
  noise = .torch$randn(size = c(100L, batch_size, n_latent),dtype = .dtype, device = .device)
  samples = .torch$add(.torch$tensordot(noise, model$rawSigma$t(), dims = 1L), result)
  E = .torch$add(.torch$mul(.torch$sigmoid(.torch$mul(alpha, samples)) , .torch$sub(one,eps)), .torch$mul(eps, half))
  E = .torch$mean(E, dim = 0L)
  return(E$data$cpu()$numpy())
}


#' train.deepJmodel
#' 
#' train function
#' 
#' @param model model
#' @param data data
#' @export
train.deepJmodel = function(model, data){
  for(i in 1:model$layer){
    data = do.call(model$feedforward[[i]]$f,
                   c(X = data,
                     W = list(model$weights[[i]]),
                     train = TRUE,
                     model$feedforward[[i]]$p))
  }
  return(data)
}
