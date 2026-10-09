#' @title Installation help
#' @name installation_help
#' @description Trouble shooting guide for the installation of the sjSDM package
#'
#' Since version 1.1.0 sjSDM runs on the 'torch' package (libtorch) and no longer needs
#' 'python', 'conda', 'reticulate' or 'PyTorch'. Installing sjSDM from CRAN pulls in
#' 'torch', and 'torch' then downloads the libtorch and liblantern binaries on first use.
#' Almost every installation problem is a problem with that download.
#'
#' @section Before you start:
#'
#' Check what is missing:
#' \preformatted{
#' sjSDM::install_diagnostic()
#' }
#' If it reports \code{libtorch installed: FALSE}, the binaries are missing. Get them with
#' \preformatted{
#' torch::install_torch()
#' # or, equivalently
#' sjSDM::install_sjSDM()
#' }
#' and restart R afterwards.
#'
#' @section Behind a proxy or without an internet connection:
#'
#' 'torch' can be installed from local files. Download the libtorch and liblantern
#' archives matching your platform on a machine that has access, then point 'torch' at
#' them before loading it:
#' \preformatted{
#' Sys.setenv(TORCH_URL = "/path/to/libtorch.zip")
#' Sys.setenv(LANTERN_URL = "/path/to/liblantern.zip")
#' torch::install_torch()
#' }
#' See \code{?torch::install_torch} for the current variable names and the download URLs.
#'
#' @section GPU support:
#'
#' There is no separate 'gpu' version of sjSDM any more, and \code{install_sjSDM()} has no
#' \code{version} argument. Whether the GPU can be used is decided entirely by the 'torch'
#' installation:
#' \itemize{
#'  \item CUDA: install a CUDA enabled 'torch' build, then check
#'    \code{torch::cuda_is_available()}. Pass \code{device = "gpu"} or
#'    \code{device = 0L} (the CUDA device index) to \code{\link{sjSDM}}.
#'  \item Apple silicon: \code{torch::backends_mps_is_available()}, then
#'    \code{device = "mps"}.
#'  \item otherwise everything runs on the CPU, which is the default.
#' }
#'
#' @section Migrating from sjSDM 1.0.x:
#'
#' The 'r-sjsdm' conda environment is no longer used and can be deleted. Model code does
#' not change. Two things behave differently:
#' \itemize{
#'  \item results are not bit-identical to the 'PyTorch' backend. The two use different
#'    random number streams, so weight initialisation and the Monte-Carlo draws differ.
#'    Estimates agree within Monte-Carlo noise, they do not agree to the last digit.
#'  \item \code{sjSDMControl(mixed = TRUE)} (half precision) is accepted but ignored.
#' }
#'
#' @section Help and bugs:
#'
#' To report bugs or ask for help, post a
#' \href{https://stackoverflow.com/questions/5963269/how-to-make-a-great-r-reproducible-example/}{reproducible example}
#' via the sjSDM \href{https://github.com/TheoreticalEcology/s-jSDM/issues/}{issue tracker}
#' with a copy of the \code{\link{install_diagnostic}} output as a quote.
"_PACKAGE"
