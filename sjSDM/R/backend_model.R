# R torch backend. `Model_sjSDM` is an nn_module holding the predictor (`net`) and the
# likelihood (`loss`, which owns sigma and theta).

# bumped whenever the state dict or the two property lists change shape; checkModel() refuses
# an object that does not carry this exact string
sjsdm_backend_version = "1.1.0"

sjsdm_set_seed = function(seed) {
  seed = as.integer(seed)
  set.seed(seed)
  torch::torch_manual_seed(seed)
  invisible(seed)
}

# The single funnel every model build goes through, so the CUDA request is validated here
# rather than in sjSDM() -- "cuda:0" as a string and an out-of-range index both used to reach
# libtorch and die there with an aten::empty_strided backend error.
sjsdm_device = function(device) {
  idx = cuda_index(device)
  if (is.na(idx)) return(torch::torch_device(device))
  if (!torch::cuda_is_available()) {
    warning("CUDA is not available, falling back to the CPU", call. = FALSE)
    return(torch::torch_device("cpu"))
  }
  n = torch::cuda_device_count()
  if (idx >= n) stop("device cuda:", idx, " requested, ", n, " CUDA device(s) present",
                     call. = FALSE)
  torch::torch_device("cuda", idx)
}

# the requested CUDA index, NA for anything that is not a CUDA request
cuda_index = function(device) {
  if (is.numeric(device)) return(as.integer(device))
  if (!grepl("^(gpu|cuda)(:[0-9]+)?$", device)) return(NA_integer_)
  i = sub("^[a-z]+:?", "", device)
  if (nzchar(i)) as.integer(i) else 0L
}

# An integer matrix carrying NA turns into the int minimum, not NaN, when handed to
# torch, which silently destroys the missing value masking. Always go through double.
sjsdm_tensor = function(x, dtype = torch::torch_float32(), device = NULL) {
  # as.matrix() flattens anything with more than two dimensions into a column
  checkmate::assert(checkmate::check_atomic_vector(x), checkmate::check_array(x, max.d = 2),
                    checkmate::check_data_frame(x))
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

# anova fits `~0` formulas, and nn_linear cannot be built with in_features = 0 (its kaiming
# init divides by sqrt(fan_in)).
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

# A DNN takes its intercept from the intercept column of its design, so its first hidden
# layer carries no bias; an NN block has no intercept column, and loses the output bias
# instead because that one would alias with the parametric intercept (PREDICTOR_plan.md 4.2).
layer_bias = function(bias, hidden, output_bias) {
  b = as.logical(unlist(bias))
  if (length(b) == 1) b = rep(b[1], length(hidden))
  if (output_bias) c(FALSE, b) else c(b, FALSE)
}

# bias[i] belongs to hidden layer i and bias[length(hidden) + 1] to the output layer;
# layer_bias() is the only place that resolves it
build_nn = function(input_shape, output_shape, hidden = list(), bias = list(FALSE),
                    activation = "linear", dropout = NULL) {
  input_shape = as.integer(input_shape)
  output_shape = as.integer(output_shape)
  # A switched-off block contributes exact zeros; keeping its hidden layers would leak a
  # learned species constant out of the last bias (BACKEND_rewrite_plan.md 6.5).
  if (input_shape == 0L)
    return(torch::nn_sequential(nn_linear_empty(0L, output_shape, bias = FALSE)))
  hidden = as.integer(unlist(hidden))
  activation = as.character(unlist(activation))
  bias = as.logical(unlist(bias))
  if (length(activation) != length(hidden)) activation = rep(activation[1], length(hidden))
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

# mu is the sum of the blocks. The penalty configuration is read through `self`: a closure
# over the module would be a dead pointer after torch_load (BACKEND_rewrite_plan.md 6.7).
sjsdm_net = torch::nn_module(
  "sjsdm_net",
  initialize = function(blocks, random = list(), block_at = NULL) {
    self$blocks = torch::nn_module_list(lapply(blocks, function(b)
      build_nn(b$input_shape, b$output_shape, b$hidden, b$bias, b$activation, b$dropout)))
    self$penalties = lapply(blocks, function(b)
      list(l1 = b$l1, l2 = b$l2, intercept = isTRUE(b$intercept)))
    self$block_at = if (is.null(block_at)) seq_along(blocks) else as.integer(block_at)
    self$re_at = as.integer(vapply(random, function(r) r$at, 1L))
    if (length(random))
      self$random = torch::nn_module_list(lapply(random, function(r)
        random_block(r$species, r$q, r$df, r$nlevels, r$loading)))
  },
  forward = function(inputs, sample = TRUE, random_noise = NULL) {
    mu = NULL
    for (i in seq_along(self$blocks)) {
      j = self$block_at[i]
      if (j > length(inputs) || is.null(inputs[[j]])) next
      h = self$blocks[[i]](inputs[[j]])
      mu = if (is.null(mu)) h else mu$add(h)
    }
    for (k in seq_along(self$re_at)) {
      h = self$random[[k]](inputs[[self$re_at[k]]], sample,
                           if (is.null(random_noise)) NULL else random_noise[[k]])
      mu = if (is.null(mu)) h else mu$add(h)
    }
    mu
  },
  penalty = function() {
    out = list()
    for (i in seq_along(self$blocks)) {
      v = reg_weights(self$blocks[[i]]$parameters, self$penalties[[i]])
      if (!is.null(v)) out = c(out, list(v))
    }
    out
  },
  # Kept out of penalty(): logLik()[[2]] reports the elastic-net closures and would silently
  # become "penalties plus KL/N" (PREDICTOR_plan.md 6.2). The KL enters batch_loss() alone.
  kl = function() {
    v = NULL
    for (k in seq_along(self$re_at)) {
      q = self$random[[k]]$kl()
      v = if (is.null(v)) q else v$add(q)
    }
    v
  },
  prior_var = function(inputs) {
    v = NULL
    for (k in seq_along(self$re_at)) {
      q = self$random[[k]]$prior_var(inputs[[self$re_at[k]]])
      v = if (is.null(v)) q else v$add(q)
    }
    v
  },
  # m_g and L_g are stepped on every batch but only get a likelihood gradient from the
  # batches their group is in. Under weight decay the idle step is ~ lr * sign(m_g) whatever
  # the decay, because RMSprop normalises by the RMS of that same gradient, which annihilates
  # the small groups (PREDICTOR_plan.md 6.3). Lambda is global and keeps the shared decay.
  parameter_groups = function() {
    np = self$named_parameters()
    v = grepl("^random\\.[0-9]+\\.(m|L_raw)$", names(np))
    list(fixed = np[!v], variational = np[v])
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

# one design, or the list of designs a multi block formula produces
as_input = function(x) if (is.null(x) || (is.list(x) && !is.data.frame(x))) x else list(x)

#' @noRd
Model_sjSDM = torch::nn_module(
  "Model_sjSDM",
  initialize = function(blocks, loss, n_env = 1L, random = list(), block_at = NULL,
                        optimizer = NULL, scheduler = FALSE,
                        patience = 2L, factor = 0.95, mixed = FALSE, device = "cpu",
                        dtype = "float32", seed = 42L) {
    self$n_env = as.integer(n_env)
    self$seed = as.integer(seed)
    sjsdm_set_seed(self$seed)
    self$device = sjsdm_device(device)
    self$dtype = sjsdm_dtype(dtype)
    self$net = sjsdm_net(blocks, random, block_at)
    self$net$to(device = self$device, dtype = self$dtype)
    self$loss = do.call(mvp_loss, c(loss, list(dtype = self$dtype, device = self$device)))
    self$optimizer_config = optimizer
    self$sched = list(on = isTRUE(scheduler), patience = as.integer(patience), factor = factor)
    self$mixed = mixed
    self$optimizer = NULL
    self$scheduler = NULL
    self$history = NULL
  },

  forward = function(inputs, sample = TRUE) self$net(inputs, sample),

  # built on the first fit() rather than on construction: anova() and sjSDM_cv() rebuild the
  # model for prediction far more often than they train it
  build_optimizer = function() {
    if (is.null(self$optimizer_config)) return(invisible(NULL))
    ok = function(p) p$requires_grad && p$numel() > 0
    mk = do.call(self$optimizer_config$ff(), self$optimizer_config$params)
    if (!length(self$net$re_at)) {
      pars = Filter(ok, c(self$net$parameters, self$loss$parameters))
      if (length(pars) == 0) return(invisible(NULL))
      self$optimizer = mk(pars)
    } else {
      g = self$net$parameter_groups()
      self$optimizer = mk(list(
        list(params = Filter(ok, unname(c(g$fixed, self$loss$parameters)))),
        list(params = Filter(ok, unname(g$variational)), weight_decay = 0)))
    }
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

  batch_loss = function(inputs, Yb, sampling, kl_scale = 0.0) {
    l = self$loss(self$net(inputs), Yb, sampling)$mean()
    for (p in c(self$net$penalty(), self$loss$penalty())) l = l$add(p)
    kl = self$net$kl()
    if (!is.null(kl)) l = l$add(kl$mul(kl_scale))
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
    d = self$as_tensors(c(as_input(X), as_input(SP)), Y)
    n = d$inputs[[1]]$shape[1]
    batch_size = max(1L, as.integer(batch_size))
    epochs = as.integer(epochs)
    sampling = as.integer(sampling)
    # one gather per epoch, the batches are then free views into it
    rng = batch_ranges(n, batch_size, drop_last = TRUE)
    # the sites actually visited per epoch, not n: fit() drops the remainder, and dividing
    # the KL by n would under-weight it by exactly that (PREDICTOR_plan.md 6.2)
    kl_scale = 1.0 / sum(vapply(rng, function(r) r[2], 1L))
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
        l = self$batch_loss(narrow_all(Xs, r[1], r[2]), Ys$narrow(1, r[1], r[2]), sampling,
                            kl_scale)
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
    d = self$as_tensors(c(as_input(X), as_input(SP)), Y)
    n = d$inputs[[1]]$shape[1]
    rng = batch_ranges(n, max(1L, as.integer(batch_size)), drop_last = FALSE)
    vals = vector("list", length(rng))
    torch::with_no_grad({
      for (k in seq_along(rng)) {
        r = rng[[k]]
        vals[[k]] = as.numeric(self$loss(self$net(narrow_all(d$inputs, r[1], r[2]), FALSE),
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
    d = self$as_tensors(c(as_input(newdata), as_input(SP)))
    n = d$inputs[[1]]$shape[1]
    if (dropout) self$net$train() else self$net$eval()
    rng = batch_ranges(n, max(1L, as.integer(batch_size)), drop_last = FALSE)
    out = vector("list", length(rng))
    torch::with_no_grad({
      for (k in seq_along(rng)) {
        r = rng[[k]]
        inp = narrow_all(d$inputs, r[1], r[2])
        mu = self$net(inp, FALSE)
        out[[k]] = if (simulate) {
          noise = mvp_noise(as.integer(sampling), mu$shape[1], self$loss$sigma$shape[2],
                            self$device, self$dtype)
          as.array(mvp_eta(mu, self$loss$sigma, noise, 1.0)$cpu())
        } else {
          # a group the fit never saw contributes no mean and its prior variance instead
          as.matrix(self$loss$response(mu, link, marginal, sampling,
                                       self$net$prior_var(inp))$cpu())
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
    d = self$as_tensors(c(as_input(X), as_input(SP)), Y)
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
        # every other block enters as a fixed offset, so the errors are conditional on them
        for (b in seq_along(self$net$blocks)[-1]) {
          j = self$net$block_at[b]
          if (j > length(d$inputs) || is.null(d$inputs[[j]])) next
          mu = mu$add(self$net$blocks[[b]](d$inputs[[j]]$narrow(1, r[1], r[2])))
        }
        for (b in seq_along(self$net$re_at))
          mu = mu$add(self$net$random[[b]](
            d$inputs[[self$net$re_at[b]]]$narrow(1, r[1], r[2]), FALSE))
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
    env = function() self$net$blocks[[1]],
    # Lambda is identified only up to a rotation of R^d, exactly as sigma is, so what is
    # reported is G_j = Lambda[j,,] Lambda[j,,]' and the group means m_g' Lambda[j,k,]
    random_effects = function() {
      lapply(seq_along(self$net$re_at), function(k) {
        b = self$net$random[[k]]
        L = b$lambda()$detach()
        M = torch::torch_matmul(b$m$detach(), L$reshape(c(-1L, b$d))$t())
        G = as.array(torch::torch_matmul(L, L$transpose(2, 3))$cpu())
        list(q = b$q, df = b$d, nlevels = b$nlevels,
             loading = if (b$shared) "shared" else "species",
             lambda = as.array(b$Lambda$detach()$cpu()),
             L = as.array(b$L_mat()$detach()$cpu()),
             cov = if (b$shared) G[1, , ] else G,
             mean = aperm(as.array(M$reshape(c(b$nlevels, b$species, b$q))$cpu()), c(1, 3, 2)))
      })
    },
    spatial = function() if (length(self$net$blocks) > self$n_env)
                           self$net$blocks[[self$n_env + 1L]] else NULL,
    sigma = function() self$loss$sigma,
    theta = function() self$loss$theta,
    link = function() self$loss$link,
    alpha = function() self$loss$alpha,
    get_sigma = function() self$loss$get_sigma,
    get_theta = function() self$loss$get_theta,
    covariance = function() self$loss$covariance,
    env_weights = function() lapply(self$env$parameters, function(p) as.array(p$detach()$cpu())),
    spatial_weights = function() {
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
  bad = anyNA(at) || (is.null(names(w)) && length(w) != length(sd)) || max(at) > length(sd)
  if (bad) stop("cannot set state: this module holds ", length(sd), " tensor(s), named ",
                paste(names(sd), collapse = ", "), call. = FALSE)
  for (k in seq_along(w)) {
    t0 = sd[[at[k]]]
    v = sjsdm_tensor(w[[k]], t0$dtype, t0$device)
    if (v$numel() != t0$numel())
      stop("cannot set state: '", names(sd)[at[k]], "' needs ", t0$numel(), " values, got ",
           v$numel(), call. = FALSE)
    sd[[at[k]]] = if (identical(v$shape, t0$shape)) v else v$reshape(t0$shape)
  }
  module$load_state_dict(sd)
  invisible(NULL)
}

# A reference cell, not a raw vector: setWeights() changes the live module, and the bytes the
# object would be restored from after saveRDS() have to follow it.
sjsdm_state = function(module, cell = new.env(parent = emptyenv())) {
  cell$raw = torch::torch_serialize(module$state_dict())
  cell
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
