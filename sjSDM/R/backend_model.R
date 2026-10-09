# R torch backend. `Model_sjSDM` is an nn_module holding the predictor (`net`, a list of
# additive blocks) and the likelihood (`loss`, which owns sigma and theta). Everything the
# port used to hand-maintain -- the parameter list, the weight setters, the device moves --
# comes from the module system.

# bumped whenever the state dict or the two property lists change shape; checkModel() refuses
# an object that does not carry this exact string
sjsdm_backend_version = "1.1.0"

sjsdm_set_seed = function(seed) {
  seed = as.integer(seed)
  set.seed(seed)
  torch::torch_manual_seed(seed)
  invisible(seed)
}

sjsdm_device = function(device) {
  if (is.numeric(device)) return(torch::torch_device("cuda", as.integer(device)))
  if (identical(device, "gpu") || identical(device, "cuda")) return(torch::torch_device("cuda", 0L))
  torch::torch_device(device)
}

# An integer matrix carrying NA turns into the int minimum, not NaN, when handed to
# torch, which silently destroys the missing value masking. Always go through double.
sjsdm_tensor = function(x, dtype = torch::torch_float32(), device = NULL) {
  # as.matrix() silently flattens a 3d array into a column vector, so refuse rather than
  # hand torch something with the wrong shape
  if (length(dim(x)) > 2L) stop("sjsdm_tensor expects a vector or a matrix, got ",
                                length(dim(x)), " dimensions", call. = FALSE)
  x = as.matrix(x)
  storage.mode(x) = "double"
  if (is.null(device)) torch::torch_tensor(x, dtype = dtype)
  else torch::torch_tensor(x, dtype = dtype, device = device)
}

sjsdm_dtype = function(dtype) {
  if (inherits(dtype, "torch_dtype")) return(dtype)
  switch(as.character(dtype),
         float32 = torch::torch_float32(),
         float64 = torch::torch_float64(),
         torch::torch_float32())
}

# anova fits models with `~0` formulas, i.e. design matrices with zero columns.
# torch::nn_linear cannot be built with in_features = 0 (its kaiming init divides by
# sqrt(fan_in)), so the degenerate case gets its own module. The layer output is the
# empty sum, i.e. zero, plus the bias, which is what PyTorch produces there too.
nn_linear_empty = torch::nn_module(
  "nn_linear_empty",
  initialize = function(in_features, out_features, bias = TRUE) {
    self$in_features = in_features
    self$out_features = out_features
    self$weight = torch::nn_parameter(torch::torch_zeros(c(out_features, in_features)))
    if (bias) self$bias = torch::nn_parameter(torch::torch_zeros(out_features))
  },
  forward = function(input) {
    out = input$new_zeros(c(input$shape[1], self$out_features))
    if (!is.null(self$bias)) out$add(self$bias) else out
  }
)

sjsdm_linear = function(in_features, out_features, bias) {
  if (in_features > 0) torch::nn_linear(in_features, out_features, bias = bias)
  else nn_linear_empty(in_features, out_features, bias = bias)
}

build_nn = function(input_shape, output_shape, hidden = list(), bias = list(FALSE),
                    activation = "linear", dropout = NULL) {
  input_shape = as.integer(input_shape)
  output_shape = as.integer(output_shape)
  # A block whose design has no columns is switched off, and off means it contributes exact
  # zeros to the sum. Building it with its hidden layers would leak a learned species
  # constant out of the last bias instead, which is what broke anova()'s DNN refits
  # (BACKEND_rewrite_plan.md 6.5).
  if (input_shape == 0L)
    return(torch::nn_sequential(nn_linear_empty(0L, output_shape, bias = FALSE)))
  hidden = as.integer(unlist(hidden))
  activation = as.character(unlist(activation))
  bias = as.logical(unlist(bias))
  if (length(activation) != length(hidden)) activation = rep(activation[1], length(hidden))
  if (length(bias) == 1) bias = rep(bias[1], length(hidden))
  bias = c(FALSE, bias)
  layers = list()
  add = function(m) layers[[length(layers) + 1]] <<- m
  if (length(hidden) > 0) {
    for (i in seq_along(hidden)) {
      add(sjsdm_linear(if (i == 1) input_shape else hidden[i - 1], hidden[i], bias = bias[i]))
      act = switch(activation[i],
                   relu = torch::nn_relu(), selu = torch::nn_selu(),
                   leakyrelu = torch::nn_leaky_relu(), tanh = torch::nn_tanh(),
                   sigmoid = torch::nn_sigmoid(), NULL)
      if (!is.null(act)) add(act)
      if (!is.null(dropout) && dropout > 0.0) add(torch::nn_dropout(p = dropout))
    }
    add(sjsdm_linear(hidden[length(hidden)], output_shape, bias = bias[length(bias)]))
  } else {
    add(sjsdm_linear(input_shape, output_shape, bias = FALSE))
  }
  do.call(torch::nn_sequential, layers)
}

off = function(x) is.null(x) || x <= 0.0

# L1/L2 on weight matrices only, never on a bias, and never on the intercept column
reg_weights = function(params, p) {
  if (off(p$l1) && off(p$l2)) return(NULL)
  ws = Filter(function(w) length(w$shape) > 1, params)
  tot = NULL
  for (k in seq_along(ws)) {
    w = ws[[k]]
    # narrow is python's p[:,1:]: on an intercept-only [S, 1] weight it is empty and
    # contributes 0. `p[, 2:p$shape[2]]` hit R's 2:1 trap and penalised the intercept.
    if (k == 1 && isTRUE(p$intercept)) w = w$narrow(2, 2, w$shape[2] - 1)
    v = NULL
    if (!off(p$l1)) v = w$abs()$sum()$mul(p$l1)
    if (!off(p$l2)) {
      q = w$pow(2.0)$sum()$mul(p$l2)
      v = if (is.null(v)) q else v$add(q)
    }
    tot = if (is.null(tot)) v else tot$add(v)
  }
  tot
}

# The predictor: mu is the sum of the blocks. Today env + spatial, but the list is the
# construction path for every future term as well. The penalty configuration sits beside the
# block list and is read through `self` -- a closure over the module would be a dead pointer
# after torch_load (BACKEND_rewrite_plan.md 6.7).
sjsdm_net = torch::nn_module(
  "sjsdm_net",
  initialize = function(blocks) {
    self$blocks = torch::nn_module_list(lapply(blocks, function(b)
      build_nn(b$input_shape, b$output_shape, b$hidden, b$bias, b$activation, b$dropout)))
    self$penalties = lapply(blocks, function(b)
      list(l1 = b$l1, l2 = b$l2, intercept = isTRUE(b$intercept)))
  },
  forward = function(inputs) {
    mu = NULL
    for (i in seq_along(self$blocks)) {
      if (i > length(inputs) || is.null(inputs[[i]])) next
      h = self$blocks[[i]](inputs[[i]])
      mu = if (is.null(mu)) h else mu$add(h)
    }
    mu
  },
  # a list, not a sum: the fit loop adds the terms one at a time and the float result
  # depends on that order
  penalty = function() {
    out = list()
    for (i in seq_along(self$blocks)) {
      v = reg_weights(self$blocks[[i]]$parameters, self$penalties[[i]])
      if (!is.null(v)) out = c(out, list(v))
    }
    out
  }
)

# list of (start, length) pairs, 1 based, for torch narrow()
batch_ranges = function(n, batch_size, drop_last) {
  nb = n %/% batch_size
  if (drop_last && nb >= 1) return(lapply(seq_len(nb), function(k)
    c((k - 1L) * batch_size + 1L, batch_size)))
  nb = ceiling(n / batch_size)
  lapply(seq_len(nb), function(k) {
    s = (k - 1L) * batch_size + 1L
    c(s, min(batch_size, n - s + 1L))
  })
}

narrow_all = function(inputs, start, len)
  lapply(inputs, function(z) if (is.null(z)) NULL else z$narrow(1, start, len))

#' @noRd
Model_sjSDM = torch::nn_module(
  "Model_sjSDM",
  initialize = function(blocks, loss, optimizer = NULL, scheduler = FALSE, patience = 2L,
                        factor = 0.95, mixed = FALSE, device = "cpu", dtype = "float32",
                        seed = 42L) {
    self$seed = as.integer(seed)
    sjsdm_set_seed(self$seed)
    self$device = sjsdm_device(device)
    self$dtype = sjsdm_dtype(dtype)
    self$net = sjsdm_net(blocks)
    self$net$to(device = self$device, dtype = self$dtype)
    self$loss = do.call(mvp_loss, c(loss, list(dtype = self$dtype, device = self$device)))
    self$optimizer_config = optimizer
    self$sched = list(on = isTRUE(scheduler), patience = as.integer(patience), factor = factor)
    self$mixed = mixed
    self$optimizer = NULL
    self$scheduler = NULL
    self$history = NULL
  },

  forward = function(inputs) self$net(inputs),

  # built on the first fit() rather than on construction: anova() and sjSDM_cv() rebuild the
  # model for prediction far more often than they train it
  build_optimizer = function() {
    if (is.null(self$optimizer_config)) return(invisible(NULL))
    pars = Filter(function(p) p$requires_grad && p$numel() > 0,
                  c(self$net$parameters, self$loss$parameters))
    if (length(pars) == 0) return(invisible(NULL))
    self$optimizer = do.call(self$optimizer_config$ff(), self$optimizer_config$params)(pars)
    if (self$sched$on)
      self$scheduler = torch::lr_reduce_on_plateau(self$optimizer, mode = "min",
                                                   patience = self$sched$patience,
                                                   factor = self$sched$factor)
    invisible(NULL)
  },

  as_tensors = function(inputs, Y = NULL) {
    mk = function(m) if (is.null(m)) NULL else sjsdm_tensor(m, self$dtype, self$device)
    list(inputs = lapply(inputs, mk), Y = mk(Y))
  },

  batch_loss = function(inputs, Yb, sampling) {
    l = self$loss(self$net(inputs), Yb, sampling)$mean()
    for (p in c(self$net$penalty(), self$loss$penalty())) l = l$add(p)
    l
  },

  reg_loss = function() {
    v = 0.0
    torch::with_no_grad({
      for (p in c(self$net$penalty(), self$loss$penalty())) v = v + as.numeric(p$cpu())
    })
    v
  },

  fit = function(X, Y, SP = NULL, batch_size = 25L, epochs = 100L, sampling = 1000L,
                 parallel = 0L, early_stopping_training = -1, verbose = TRUE) {
    d = self$as_tensors(list(X, SP), Y)
    n = d$inputs[[1]]$shape[1]
    batch_size = max(1L, as.integer(batch_size))
    epochs = as.integer(epochs)
    sampling = as.integer(sampling)
    # one gather per epoch, the batches are then free views into it
    rng = batch_ranges(n, batch_size, drop_last = TRUE)
    if (is.null(self$optimizer)) self$build_optimizer()
    losses = numeric(epochs)
    done = 0L
    es_on = early_stopping_training > 0
    es_best = Inf
    es_count = 0L
    if (self$mixed) message("mixed precision is not available in the torch backend, ignored")
    self$net$train()
    train_l = ""
    if (verbose)
      pb = cli::cli_progress_bar(format = "{cli::pb_bar} {cli::pb_current}/{cli::pb_total} loss: {train_l}",
                                 total = epochs, clear = FALSE)

    for (epoch in seq_len(epochs)) {
      perm = torch::torch_randperm(n, dtype = torch::torch_long(),
                                   device = self$device)$add(1L)
      Xs = lapply(d$inputs, function(z) if (is.null(z)) NULL else z$index_select(1, perm))
      Ys = d$Y$index_select(1, perm)
      bl = numeric(length(rng))
      for (k in seq_along(rng)) {
        r = rng[[k]]
        if (!is.null(self$optimizer)) self$optimizer$zero_grad()
        l = self$batch_loss(narrow_all(Xs, r[1], r[2]), Ys$narrow(1, r[1], r[2]), sampling)
        if (l$requires_grad) {
          l$backward()
          self$optimizer$step()
        }
        bl[k] = as.numeric(l$detach()$cpu())
      }
      m = round(mean(bl), 3)
      losses[epoch] = m
      done = epoch
      train_l = format(m)
      if (verbose) cli::cli_progress_update(id = pb, set = epoch)
      if (!is.null(self$scheduler)) self$scheduler$step(m)
      if (es_on) {
        if (m < es_best) {
          es_best = m
          es_count = 0L
        } else es_count = es_count + 1L
        if (es_count == early_stopping_training) break
      }
    }
    if (verbose) cli::cli_progress_done(id = pb)
    self$net$eval()
    # only the epochs that ran: an early stopped fit used to leave the rest as zeros and
    # plot.sjSDM drew them as the loss curve
    self$history = data.frame(epoch = seq_len(done), train_l = losses[seq_len(done)])
    invisible(NULL)
  },

  logLik = function(X, Y, SP = NULL, batch_size = 25L, parallel = 0L, sampling = 100L,
                    individual = FALSE, train = TRUE) {
    d = self$as_tensors(list(X, SP), Y)
    n = d$inputs[[1]]$shape[1]
    rng = batch_ranges(n, max(1L, as.integer(batch_size)), drop_last = FALSE)
    vals = vector("list", length(rng))
    torch::with_no_grad({
      for (k in seq_along(rng)) {
        r = rng[[k]]
        vals[[k]] = as.numeric(self$loss(self$net(narrow_all(d$inputs, r[1], r[2])),
                                         d$Y$narrow(1, r[1], r[2]), sampling)$cpu())
      }
    })
    ll = unlist(vals)
    reg = self$reg_loss()
    if (individual) list(matrix(ll, ncol = 1), reg) else list(sum(ll), reg)
  },

  predict = function(newdata = NULL, SP = NULL, Y = NULL, train = FALSE, batch_size = 25L,
                     parallel = 0L, sampling = 1000L, link = TRUE, dropout = FALSE,
                     simulate = FALSE, marginal = TRUE) {
    d = self$as_tensors(list(newdata, SP))
    n = d$inputs[[1]]$shape[1]
    if (dropout) self$net$train() else self$net$eval()
    rng = batch_ranges(n, max(1L, as.integer(batch_size)), drop_last = FALSE)
    out = vector("list", length(rng))
    torch::with_no_grad({
      for (k in seq_along(rng)) {
        r = rng[[k]]
        mu = self$net(narrow_all(d$inputs, r[1], r[2]))
        out[[k]] = if (simulate) {
          noise = mvp_noise(as.integer(sampling), mu$shape[1], self$loss$sigma$shape[2],
                            self$device, self$dtype)
          as.array(mvp_eta(mu, self$loss$sigma, noise, 1.0)$cpu())
        } else {
          as.matrix(self$loss$response(mu, link, marginal, sampling)$cpu())
        }
      }
    })
    self$net$eval()
    predictions = if (simulate) do.call(function(...) abind::abind(..., along = 2), out)
                  else do.call(rbind, out)
    if (is.null(Y)) return(predictions)
    sjsdm_conditional(self, predictions, Y, sampling)
  },

  se = function(X, Y, SP = NULL, batch_size = 25L, parallel = 0L, sampling = 100L,
                verbose = TRUE) {
    d = self$as_tensors(list(X, SP), Y)
    n = d$inputs[[1]]$shape[1]
    w0 = self$env$parameters[[1]]$detach()$t()          # [predictors, species]
    P = w0$shape[1]
    S = w0$shape[2]
    rng = batch_ranges(n, max(1L, as.integer(batch_size)), drop_last = FALSE)
    eye = torch::torch_eye(P, dtype = self$dtype, device = self$device)
    out = vector("list", S)
    if (verbose) cat("\nCalculating standard errors...\n")
    for (i in seq_len(S)) {
      wi = w0[, i]$reshape(c(P, 1))$clone()$requires_grad_(TRUE)
      parts = list()
      if (i > 1) parts = c(parts, list(w0[, 1:(i - 1), drop = FALSE]))
      parts = c(parts, list(wi))
      if (i < S) parts = c(parts, list(w0[, (i + 1):S, drop = FALSE]))
      w = torch::torch_cat(parts, dim = 2)
      H = NULL
      for (r in rng) {
        mu = torch::torch_matmul(d$inputs[[1]]$narrow(1, r[1], r[2]), w)
        if (!is.null(self$spatial) && !is.null(d$inputs[[2]]))
          mu = mu$add(self$spatial(d$inputs[[2]]$narrow(1, r[1], r[2])))
        l = self$loss(mu, d$Y$narrow(1, r[1], r[2]), sampling)$sum()
        g1 = torch::autograd_grad(l, wi, retain_graph = TRUE, create_graph = TRUE,
                                  allow_unused = TRUE)[[1]]
        cols = lapply(seq_len(P), function(j)
          torch::autograd_grad(g1[j, 1], wi, retain_graph = TRUE, create_graph = FALSE)[[1]])
        h = torch::torch_cat(cols, dim = 2)
        H = if (is.null(H)) h else H$add(h)
      }
      out[[i]] = as.numeric(torch::linalg_inv(H$add(eye$mul(0.001)))$diagonal()$sqrt()$cpu())
      if (verbose) cat(sprintf("\rSpecies: %d/%d", i, S))
    }
    if (verbose) cat("\n")
    out
  },

  active = list(
    env = function(value) {
      if (!missing(value)) stop("read only")
      self$net$blocks[[1]]
    },
    spatial = function(value) {
      if (!missing(value)) stop("read only")
      if (length(self$net$blocks) > 1) self$net$blocks[[2]] else NULL
    },
    sigma = function(value) {
      if (!missing(value)) stop("read only")
      self$loss$sigma
    },
    theta = function(value) {
      if (!missing(value)) stop("read only")
      self$loss$theta
    },
    link = function(value) {
      if (!missing(value)) stop("read only")
      self$loss$link
    },
    alpha = function(value) {
      if (!missing(value)) stop("read only")
      self$loss$alpha
    },
    get_sigma = function(value) {
      if (!missing(value)) stop("read only")
      self$loss$get_sigma
    },
    get_theta = function(value) {
      if (!missing(value)) stop("read only")
      self$loss$get_theta
    },
    covariance = function(value) {
      if (!missing(value)) stop("read only")
      self$loss$covariance
    },
    env_weights = function(value) {
      if (!missing(value)) stop("read only")
      lapply(self$env$parameters, function(p) as.array(p$detach()$cpu()))
    },
    spatial_weights = function(value) {
      if (!missing(value)) stop("read only")
      s = self$spatial
      if (is.null(s)) NULL else lapply(s$parameters, function(p) as.array(p$detach()$cpu()))
    }
  )
)

# Rebuild from the two plain-R property lists. Nothing fitted enters here: the weights arrive
# afterwards through one load_state_dict, see checkModel().
sjsdm_model = function(model_properties, loss_properties)
  do.call(Model_sjSDM, c(model_properties, list(loss = loss_properties)))

# The five weight setters the port carried collapse into this. A state dict is the only way
# weights enter a module now, whether they come from the user (setWeights) or from disk.
set_state = function(module, w) {
  if (!is.list(w)) w = list(w)
  sd = module$state_dict()
  at = if (is.null(names(w))) seq_along(w) else match(names(w), names(sd))
  for (k in seq_along(w)) {
    if (is.na(at[k]) || at[k] > length(sd) || is.null(w[[k]])) next
    t0 = sd[[at[k]]]
    v = sjsdm_tensor(w[[k]], t0$dtype, t0$device)
    sd[[at[k]]] = if (identical(v$shape, t0$shape)) v else v$reshape(t0$shape)
  }
  module$load_state_dict(sd)
  invisible(NULL)
}

# P(Y_k = 1 | the fully observed species), for every NA column of Y.
# `predictions` must be on the linear scale (predict(type = "raw")).
sjsdm_conditional = function(model, predictions, Y, sampling) {
  Y = as.matrix(Y)
  predictions = as.matrix(predictions)
  na_cols = which(apply(Y, 2, function(z) any(is.na(z))))
  focal = which(apply(Y, 2, function(z) !any(is.na(z))))
  if (length(focal) == 0) stop("conditional predictions need at least one fully observed species")
  sigma = model$sigma$detach()
  theta = model$theta
  cond_Y = cbind(1, Y[, focal, drop = FALSE])
  tt = function(m) sjsdm_tensor(m, model$dtype, model$device)
  out = matrix(NA_real_, nrow(Y), length(na_cols))
  torch::with_no_grad({
    raw_ll = mvp_logLik(tt(predictions[, focal, drop = FALSE]), tt(Y[, focal, drop = FALSE]),
                        sigma[focal, , drop = FALSE], link = model$link, alpha = model$alpha,
                        sampling = as.integer(sampling),
                        theta = if (is.null(theta)) NULL else theta[focal])
    for (i in seq_along(na_cols)) {
      ind = c(na_cols[i], focal)
      joint_ll = mvp_logLik(tt(predictions[, ind, drop = FALSE]), tt(cond_Y),
                            sigma[ind, , drop = FALSE], link = model$link, alpha = model$alpha,
                            sampling = as.integer(sampling),
                            theta = if (is.null(theta)) NULL else theta[ind])
      out[, i] = pmin(pmax(exp(-as.numeric(joint_ll$sub(raw_ll)$cpu())), 0), 1)
    }
  })
  colnames(out) = colnames(Y)[na_cols]
  out
}

# Variation partitioning, port of utils_fa.importance
sjsdm_importance = function(beta, covX, sigma, covSP = NULL, betaSP = NULL) {
  association = sigma %*% t(sigma) + diag(1, nrow(sigma))
  betaCorrected = t(covX) %*% beta
  Xtotal = colSums(beta * betaCorrected)
  Xsplit = t(beta * betaCorrected)
  PredRandom = abs(colSums(association) - diag(association)) / ncol(association)

  if (!is.null(betaSP)) {
    betaSPCorrected = t(covSP) %*% betaSP
    SPtotal = colSums(betaSP * betaSPCorrected)
    SPsplit = t(betaSP * betaSPCorrected)
    variTotal = Xtotal + PredRandom + SPtotal
    list(env = (Xtotal / variTotal) * (Xsplit / rowSums(Xsplit)),
         spatial = (SPtotal / variTotal) * (SPsplit / rowSums(SPsplit)),
         biotic = PredRandom / variTotal)
  } else {
    variTotal = Xtotal + PredRandom
    list(env = (Xtotal / variTotal) * (Xsplit / rowSums(Xsplit)),
         biotic = PredRandom / variTotal)
  }
}
