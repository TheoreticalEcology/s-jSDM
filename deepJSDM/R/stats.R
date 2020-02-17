#' zScores
#' calculate zScores
#' @param model model of class deepJmodel
zScores = function(model, parallel = 0L) {
  parameter = c(unlist(model$weights), model$rawSigma)
  n_app = 100L
  r_dim = ncol(model$Y)
  n_latent = model$nLatent
  batch_size = nrow(model$X)
  

  eps = .torch$tensor(0.00001, dtype = .dtype)$to(.device)
  zero = .torch$tensor(0.0, dtype = .dtype)$to(.device)
  one = .torch$tensor(1.0, dtype = .dtype)$to(.device)
  alpha = .torch$tensor(1.70169, dtype = .dtype)$to(.device)
  half = .torch$tensor(0.5, dtype = .dtype)$to(.device)

  Xs = .torch$tensor(model$X, dtype = .dtype, device = .device)$to(.device)
  Ys = .torch$tensor(model$Y, dtype = .dtype, device = .device)$to(.device)

  mu = train.deepJmodel(model, Xs)
  noise = .torch$randn(size = c(n_app, batch_size, n_latent),dtype = .dtype, device = .device)
  samples = .torch$add(.torch$tensordot(noise, model$rawSigma$t(), dims = 1L), mu)
  E = .torch$add(.torch$mul(.torch$sigmoid(.torch$mul(alpha, samples)) , .torch$sub(one,eps)), .torch$mul(eps, half))
  indll = .torch$neg(.torch$add(.torch$mul(.torch$log(E), Ys), .torch$mul(.torch$log(.torch$sub(one,E)),.torch$sub(one,Ys))))
  logprob = .torch$neg(.torch$sum(indll, dim = 2L))
  maxlogprob = .torch$max(logprob, dim = 0L)[[1]]
  Eprob = .torch$mean(.torch$exp(.torch$sub(logprob,maxlogprob)), dim = 0L)
  loss = .torch$sum(.torch$sub(.torch$neg(.torch$log(Eprob)),maxlogprob))

  gradients = .torch$autograd$grad(loss,inputs = parameter,retain_graph = TRUE,create_graph = TRUE,allow_unused = TRUE)
  gradients = .torch$cat(lapply(gradients, function(g) g$reshape(-1L)))
  gradients2 = vector("list", length(gradients))
 # for(i in 1:length(gradients)){
  #  tmp = gradients[[i]]$reshape(-1L)
   # len = .torch$as_tensor(tmp$shape)$numpy()
    len = dim(gradients)$data$cpu()$numpy()
    gradients2 = vector("list", len)
    
    step = ceiling(len*0.01)
    pb = progress::progress_bar$new(
      format = "Iter: :ep [:bar] :percent ETA: :eta",
      total = len/step,
      clear = FALSE,
      width = 80
    )
    for(j in 1:len) {
      tmp2 =
        .torch$autograd$grad(.torch$index_select(gradients, .torch$tensor(0L, device = .device), .torch$tensor(as.integer(j - 1L), device = .device)),inputs = parameter,retain_graph = T,create_graph = F,allow_unused = F)
      tmp2 = lapply(tmp2, function(t) t$reshape(-1L))
      tmp3 = .torch$cat(tmp2)
      gradients2[[j]] = tmp3$reshape(c(as.integer(dim( tmp3)$data$cpu()$numpy()), 1L))$data$cpu()$numpy()
      rm(tmp2, tmp3)
      .torch$cuda$empty_cache()
      
      if(j %% step == 0) {
        tokens = list(ep = j/step) #, time = round((setup$epochs - i)*mean(time), digits = 2))
        pb$tick(tokens = tokens)
      }
    }
  #}

  #hessian = .torch$cat(unlist(gradients2),1L)
  hessian = abind::abind(gradients2)
  coefDim = dim(parameter[[1]])$data$cpu()$numpy()
  tmp1 = paste0("sp:",1:coefDim[2])
  tmp2 = paste0("_Par_",1:coefDim[1])
  names = apply(expand.grid(tmp1, tmp2), 1, function(t) paste0(t[1], t[2]))
  intercept = dim(parameter[[2]]$reshape(-1L))$data$cpu()$numpy()
  if(intercept == coefDim[2]) {
    names = c(names, paste0("(intercept)sp ", 1: coefDim[2]))
  }

  # hessian = hessian$data$cpu()$numpy()[1:length(names), 1:length(names)]
  hessian = hessian[1:length(names), 1:length(names)]
  se = sqrt(diag(solve(hessian)))
  #se = .torch$sqrt(.torch$diag(.torch$inverse(hessian)))
  estimates = .torch$cat(lapply(parameter, function(t) t$reshape(-1L)))$data$cpu()$numpy()[1:length(names)]
  Zscores = estimates / se
  P = 2*pnorm(as.vector(abs(Zscores)), lower.tail = FALSE)

  result = data.frame(names = names, estimates = estimates,sd = se, Zscores = Zscores, P = format.pval(pv = P,digits = 1))
  return(result)
}


#' @export
logLik.deepJmodel = function(model, X = NULL, Y = NULL, torch = .torch, device = .device, dtype = .dtype) {
  if(is.null(X)) {
    X = model$X
    Y = model$Y
  }
  X = as.matrix(X)
  Y = as.matrix(Y)
  
  n_app = 100L
  r_dim = ncol(model$Y)
  n_latent = model$nLatent
  batch_size = nrow(model$X)
  
  
  eps = .torch$tensor(0.00001, dtype = .dtype)$to(.device)
  zero = .torch$tensor(0.0, dtype = .dtype)$to(.device)
  one = .torch$tensor(1.0, dtype = .dtype)$to(.device)
  alpha = .torch$tensor(1.70169, dtype = .dtype)$to(.device)
  half = .torch$tensor(0.5, dtype = .dtype)$to(.device)
  
  Xs = .torch$tensor(X, dtype = .dtype, device = .device)$to(.device)
  Ys = .torch$tensor(Y, dtype = .dtype, device = .device)$to(.device)
  
  mu = train.deepJmodel(model, Xs)
  noise = .torch$randn(size = c(n_app, batch_size, n_latent),dtype = .dtype, device = .device)
  samples = .torch$add(.torch$tensordot(noise, model$rawSigma$t(), dims = 1L), mu)
  E = .torch$add(.torch$mul(.torch$sigmoid(.torch$mul(alpha, samples)) , .torch$sub(one,eps)), .torch$mul(eps, half))
  indll = .torch$neg(.torch$add(.torch$mul(.torch$log(E), Ys), .torch$mul(.torch$log(.torch$sub(one,E)),.torch$sub(one,Ys))))
  logprob = .torch$neg(.torch$sum(indll, dim = 2L))
  maxlogprob = .torch$max(logprob, dim = 0L)[[1]]
  Eprob = .torch$mean(.torch$exp(.torch$sub(logprob,maxlogprob)), dim = 0L)
  loss = .torch$sum(.torch$sub(.torch$neg(.torch$log(Eprob)),maxlogprob))
  return(loss$data$cpu()$numpy())
}
