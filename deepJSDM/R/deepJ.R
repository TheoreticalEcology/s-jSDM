#' deepJ
#' Fit deepJ function
#'
#' @param model model of class deepJmodel
#' @param nLatent number of latent variables
#' @export

deepJ = function(model, epochs = 150, batch_size = NULL, corr = FALSE){

  ### define constants ###
  n_app = 100L
  if(is.null(batch_size)) batch_size = nrow(model$X)
  r_dim = ncol(model$Y)
  n_latent = model$nLatent
  cpu_wo_dataset = matrix(0, nrow = epochs)

  stepSize = floor(nrow(model$X)/batch_size)
  steps = stepSize * epochs

  eps = .torch$tensor(0.00001, dtype = .dtype)$to(.device)
  zero = .torch$tensor(0.0, dtype = .dtype)$to(.device)
  one = .torch$tensor(1.0, dtype = .dtype)$to(.device)
  alpha = .torch$tensor(1.70169, dtype = .dtype)$to(.device)
  half = .torch$tensor(0.5, dtype = .dtype)$to(.device)
  old = Inf

  state =
  ### get step function ####
  stepFunc = switch(model$optimizer_info,
      adamax = function(Xs, Ys) {
        mu = train.deepJmodel(model, Xs)
        if(!corr) {
          noise = .torch$randn(size = c(n_app, batch_size, n_latent),dtype = .dtype, device = .device)
          samples = .torch$add(.torch$tensordot(noise, model$rawSigma$t(), dims = 1L), mu)
        } else {
          noise = .torch$randn(size = c(n_app, batch_size, r_dim),dtype = .dtype, device = .device)
          sig = model$rawSigma
          sig = .torch$matmul(sig, sig$t())
          b = .torch$diag(.torch$div(one, .torch$sqrt(.torch$diag(sig))))
          sig = .torch$matmul(.torch$matmul(b, sig), b)
          samples = .torch$add(.torch$tensordot(noise, sig, dims = 1L), mu)
        }
        E = .torch$add(.torch$mul(.torch$sigmoid(.torch$mul(alpha, samples)) , .torch$sub(one,eps)), .torch$mul(eps, half))
        indll = .torch$neg(.torch$add(.torch$mul(.torch$log(E), Ys), .torch$mul(.torch$log(.torch$sub(one,E)),.torch$sub(one,Ys))))
        logprob = .torch$neg(.torch$sum(indll, dim = 2L))
        maxlogprob = .torch$max(logprob, dim = 0L)[[1]]
        Eprob = .torch$mean(.torch$exp(.torch$sub(logprob,maxlogprob)), dim = 0L)
        loss = .torch$mean(.torch$sub(.torch$neg(.torch$log(Eprob)),maxlogprob))
        if(!is.null(model$losses)) for(k in 1:length(model$losses))  loss = loss + do.call(model$losses[[k]],list())
        model$optimizer$zero_grad()
        loss$backward()
        model$optimizer$step()
        return(loss)
     #   rm(mu, noise, samples, E, indll, logprob, maxlogprob, Eprob)
      #  .torch$cuda$empty_cache()
      },
      LBFGS = function(Xs, Ys) {
        pars = c(unlist(model$weights), model$rawSigma)
        previous = lapply(pars, function(w) w$clone()$detach())
        opt = model$optimizer(params = pars, lr = model$params$lr)
        model$params$lr <<- 0.9*model$params$lr
        loss =
          opt$step(function(){
            mu = train.deepJmodel(model, Xs)
            if(!corr) {
              noise = .torch$randn(size = c(n_app, batch_size, n_latent),dtype = .dtype, device = .device)
              samples = .torch$add(.torch$tensordot(noise, model$rawSigma$t(), dims = 1L), mu)
            } else {
              noise = .torch$randn(size = c(n_app, batch_size, r_dim),dtype = .dtype, device = .device)
              sig = model$rawSigma
              sig = .torch$matmul(sig, sig$t())
              b = .torch$diag(.torch$div(one, .torch$sqrt(.torch$diag(sig))))
              sig = .torch$matmul(.torch$matmul(b, sig), b)
              samples = .torch$add(.torch$tensordot(noise, sig, dims = 1L), mu)
            }
            E = .torch$add(.torch$mul(.torch$sigmoid(.torch$mul(alpha, samples)) , .torch$sub(one,eps)), .torch$mul(eps, half))
            indll = .torch$neg(.torch$add(.torch$mul(.torch$log(E), Ys), .torch$mul(.torch$log(.torch$sub(one,E)),.torch$sub(one,Ys))))
            logprob = .torch$neg(.torch$sum(indll, dim = 2L))
            maxlogprob = .torch$max(logprob, dim = 0L)[[1]]
            Eprob = .torch$mean(.torch$exp(.torch$sub(logprob,maxlogprob)), dim = 0L)
            loss = .torch$mean(.torch$sub(.torch$neg(.torch$log(Eprob)),maxlogprob))
            if(!is.null(model$losses)) for(k in 1:length(model$losses))  loss = loss + do.call(model$losses[[k]],list())
            opt$zero_grad()
            loss$backward()
            rm(mu, noise, samples, E, indll, logprob, maxlogprob, Eprob)
            .torch$cuda$empty_cache()
            return(loss)
        })
        if(loss$data$cpu()$numpy() > old) {
          for(i in 1:length(pars)) {
            pars[[i]]$data = previous[[i]]$data
          }
        } else {
          old <<- loss$data$cpu()$numpy()
        }
        return(loss)
      }

  )


  pb = progress::progress_bar$new(
    format = "Epoch: :ep [:bar] :percent loss: :loss  ETA: :eta s",
    total = epochs,
    clear = FALSE,
    width = 80
  )

  data = .torch$utils$data$TensorDataset(.torch$tensor(model$X, dtype = .dtype), .torch$tensor(model$Y, dtype = .dtype))
  dataLoader =  .torch$utils$data$DataLoader(data,batch_size = batch_size,
                                             shuffle = TRUE,
                                             num_workers = 0L,
                                             drop_last = TRUE,
                                             pin_memory = TRUE)


  for(i in 1:epochs){
    batch_loss = vector(length = stepSize)
    iterator = reticulate::as_iterator(dataLoader)
    for(j in 1:stepSize){
      batch = reticulate::iter_next(iterator)
      Xs = batch[[1]]$to(.device,non_blocking=TRUE)
      Ys = batch[[2]]$to(.device,non_blocking=TRUE)
      batch_loss[j] = stepFunc(Xs, Ys)$data$cpu()$numpy()
     # .torch$cuda$empty_cache()
    }
    .torch$cuda$empty_cache()
    bl = mean(batch_loss)
    tokens = list(ep = i, loss = round(bl,digits = 3)) #, time = round((setup$epochs - i)*mean(time), digits = 2))
    pb$tick(tokens = tokens)
    cpu_wo_dataset[[i]] = bl
  }

  model$raw_weights = extractWeights(model)
  model$history = cpu_wo_dataset
  try({model$optimizer_state = model$optimizer$state_dict()},silent = TRUE)
  try({.torch$cuda$empty_cache()},silent = TRUE)
  return(model)
}


