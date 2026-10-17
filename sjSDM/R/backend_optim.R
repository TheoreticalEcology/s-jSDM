# The `optim_ignite_*` builds, never `torch::optim_rmsprop` / `optim_sgd`: those two apply
# weight_decay with the parameter *index* instead of its value. See INSIGHTS.md.

optimizer_RMSprop = function(lr = 1e-2, alpha = 0.99, eps = 1e-8, weight_decay = 0,
                             momentum = 0, centered = FALSE)
  function(params) torch::optim_ignite_rmsprop(params, lr = lr, alpha = alpha, eps = eps,
                                               weight_decay = weight_decay, momentum = momentum,
                                               centered = centered)

optimizer_SGD = function(lr = 1e-2, momentum = 0, dampening = 0, weight_decay = 0,
                         nesterov = FALSE)
  function(params) torch::optim_ignite_sgd(params, lr = lr, momentum = momentum,
                                           dampening = dampening, weight_decay = weight_decay,
                                           nesterov = nesterov)

optimizer_Adam = function(lr = 1e-3, betas = c(0.9, 0.999), eps = 1e-8, weight_decay = 0,
                          amsgrad = FALSE)
  function(params) torch::optim_ignite_adam(params, lr = lr, betas = betas, eps = eps,
                                            weight_decay = weight_decay, amsgrad = amsgrad)

optimizer_AdamW = function(lr = 1e-3, betas = c(0.9, 0.999), eps = 1e-8, weight_decay = 1e-2,
                           amsgrad = FALSE)
  function(params) torch::optim_ignite_adamw(params, lr = lr, betas = betas, eps = eps,
                                             weight_decay = weight_decay, amsgrad = amsgrad)

optimizer_Adagrad = function(lr = 1e-2, lr_decay = 0, weight_decay = 0,
                             initial_accumulator_value = 0, eps = 1e-10)
  function(params) torch::optim_ignite_adagrad(params, lr = lr, lr_decay = lr_decay,
                                               weight_decay = weight_decay,
                                               initial_accumulator_value = initial_accumulator_value,
                                               eps = eps)
