#' @export
"dim.torch.Tensor" = function(a) {
  .torch$as_tensor(a$shape)
}


#' @export
"+.torch.Tensor" = function(a, b) {
  .torch$add(a, b)
}

#' @export
"-.torch.Tensor" = function(a, b) {
  .torch$sub(a,b)
}

#' @export
"*.torch.Tensor" = function(a, b) {
  .torch$mul(a,b)
}

#' @export
"/.torch.Tensor" = function(a, b) {
  .torch$div(a,b)
}

#' @export
"^.torch.Tensor" = function(a, b) {
  .torch$pow(a,b)
}


#' @export
"%*%.torch.Tensor" = function(a, b) {
  .torch$matmul(a,b)
}

#' @export
"exp.torch.Tensor" = function(a) {
  .torch$exp(a)
}


#' @export
"sqrt.torch.Tensor" = function(a) {
  .torch$sqrt(a)
}

#' @export
"mean.torch.Tensor" = function(a, b, dim = 0L) {
  .torch$mean(a,b,dim = as.integer(dim))
}


