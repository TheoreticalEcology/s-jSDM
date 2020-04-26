# 
# compile = function(object, df = NULL, l1_cov = 0.0, l2_cov = 0.0, optimizer = NULL, ...) {
#   object$build(df = as.integer(df), l1 = l1_cov, l2 = l2_cov, optimizer = optimizer)
#   return(invisible(object))
# }
# 
# 
# optimizer_adamax = function(learning_rate = 0.01, weight_decay = 0.01) {
#   return(fa$optimizer_adamax(lr = learning_rate, weight_decay = 0.01))
# }
# 
# 
# optimizer_RMSprop = function(learning_rate = 0.01, weight_decay = 0.01, ...) {
#   return(fa$optimizer_RMSprop(lr = learning_rate, weight_decay = 0.01, ...))
# }
# 
# 
# fit = function(object, X = NULL, Y = NULL, batch_size = 25L, epochs = 100L, parallel = 0L) {
#   object$fit(X, Y, batch_size = as.integer(batch_size), epochs = as.integer(epochs), parallel = parallel)
#   return(invisible(object))
# }
# 
# 
# 
# layer_dense = function(object = NULL, units = NULL, activation = NULL, use_bias = FALSE, kernel_l1 = 0.0, kernel_l2 = 0.0) {
#   
#   stopifnot(
#     units > 0,
#     kernel_l1 >= 0.0,
#     kernel_l2 >= 0.0,
#     is.logical(use_bias),
#     inherits(object, "sjSDM_model")
#   )
#   
#   object$add_layer(fa$layers$Layer_dense(hidden = as.integer(units), 
#                                          activation = activation, 
#                                          bias = use_bias, 
#                                          l1 = kernel_l1, 
#                                          l2 = kernel_l2, 
#                                          device = object$device, 
#                                          dtype = object$dtype))
#   return(invisible(object))
# }
# 
# 
# 
