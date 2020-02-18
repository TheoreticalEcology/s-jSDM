#' @export
"dim.torch.Tensor" = function(x) {
  if (reticulate::py_is_null_xptr(x)) return(NULL)
  else torch$as_tensor(x$shape)
}

#' @export
"+.torch.Tensor" = function(x, y) {
  if (reticulate::py_is_null_xptr(x) || reticulate::py_is_null_xptr(y)) return(NULL)
  else return(torch$add(x, y))
}

#' @export
"-.torch.Tensor" = function(x, y) {
  if (reticulate::py_is_null_xptr(x) || reticulate::py_is_null_xptr(y) ) return(NULL)
  else return(torch$sub(x,y))
}

#' @export
"*.torch.Tensor" = function(x, y) {
  if (reticulate::py_is_null_xptr(x) || reticulate::py_is_null_xptr(y) ) return(NULL)
  else return(torch$mul(x,y))
}

#' @export
"/.torch.Tensor" = function(x, y) {
  if (reticulate::py_is_null_xptr(x)  || reticulate::py_is_null_xptr(y) ) return(NULL)
  else return(torch$div(x,))
}

#' @export
"^.torch.Tensor" = function(x, y) {
  if (reticulate::py_is_null_xptr(x) || reticulate::py_is_null_xptr(y) ) return(NULL)
  return(torch$pow(x,y))
}


#' @export
"exp.torch.Tensor" = function(x) {
  if (reticulate::py_is_null_xptr(x)) return(NULL)
  else return(torch$exp(x))
}


#' @export
"sqrt.torch.Tensor" = function(x) {
  if (reticulate::py_is_null_xptr(x)) return(NULL)
  else return(torch$sqrt(x))
}

#' @export
"mean.torch.Tensor" = function(x, dim = 0L, ...) {
  if (reticulate::py_is_null_xptr(x)) return(NULL)
  torch$mean(x,dim = as.integer(dim))
}
