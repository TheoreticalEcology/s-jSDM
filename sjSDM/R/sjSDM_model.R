# 
# sjSDM_model = function(input_shape,  device = NULL, dtype = "float32") {
#   
#   if(reticulate::py_is_null_xptr(fa)) .onLoad()
#   if(is.null(device)) {
#     if(torch$cuda$is_available()) device = 0L
#     else device = "cpu"
#   }
#   if(is.numeric(device)) device = as.integer(device)
#   if(device == "gpu") device = 0L
#   
#   object = fa$Model_base(as.integer(input_shape), device = device, dtype = dtype)
#   class(object) = c("sjSDM_model",class(object))
#   return(object)
# }
# 
# summary.sjSDM_model = function(object, ...) {
#   shapes = lapply(object$layers, function(l) l$shape)
#   cat("Model architecture\n")
#   cat("=======================\n")
#   for(i in 1:length(shapes)) {
#     cat("Layer_dense_", i,":\t (", shapes[[i]][1],", ", shapes[[i]][2], ")\n")
#   }
#   cat("=======================\n")
#   wN = sum(sapply(shapes, function(s) cumprod(s)[2]))
#   cat("Weights (w/o sigma):\t ", wN, "\n")
#   cat("Weights (w sigma):\t ", wN + object$df*shapes[[length(shapes)]][2], "\n")
# }
# 
# 
# print.sjSDM_model = function(x, ...) {
#   cat("sjSDM_model, see summary(model) for details \n")
# }
# 
# 
# 
# predict.sjSDM_model = function(object, newdata = NULL, ...) {
#   pred = object$predict(newdata = newdata, ...)
#   return(pred)
# }
# 
# 
# plot.sjSDM_model = function(x, y, ...) {
#   hist = x$history
#   graphics::plot(y = hist, x = 1:length(hist), xlab = "Epochs", ylab = "Loss", main = "Training history", type = "o",...)
#   return(invisible(hist))
# }
# 
