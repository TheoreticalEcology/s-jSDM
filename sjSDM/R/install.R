#' Install sjSDM's dependencies
#'
#' @param ... ignored, kept for backward compatibility (e.g. \code{version = "gpu"})
#'
#' @details
#'
#' Since version 1.1.0 sjSDM runs on 'torch' (libtorch) and no longer needs 'python',
#' 'conda' or 'PyTorch'. The only remaining dependency is the libtorch binary, which is
#' downloaded by \code{\link[torch]{install_torch}}. This function is a thin wrapper
#' around it so that existing scripts keep working. Whether the GPU is used is decided
#' by the torch installation, not by an argument here.
#'
#' @return
#'
#' No return value, called for side effects (installation of the 'torch' binaries).
#'
#' @seealso \code{\link{installation_help}}, \code{\link{install_diagnostic}}
#' @export
install_sjSDM = function(...) {

  args = list(...)
  if("version" %in% names(args))
    cli::cli_alert_info("'version' is ignored, GPU support is decided by the torch installation")

  if(is_torch_available()) {
    cli::cli_alert_success("libtorch is already installed, nothing to do.")
    return(invisible(NULL))
  }

  error = tryCatch(torch::install_torch(), error = function(e) e)

  if(!inherits(error, "error")) {
    cli::cli_alert_success("\nInstallation complete.\n\n")
    invisible(NULL)
  } else {
    cli::cli_alert_danger("\nInstallation failed, see ?installation_help\n")
    cli::cli_alert_info("If the installation still fails, please report the following error on https://github.com/TheoreticalEcology/s-jSDM/issues\n")
    cli::cli_alert(error$message)
  }
}



is_windows = function() {
  identical(.Platform$OS.type, "windows")
}

is_unix = function() {
  identical(.Platform$OS.type, "unix")
}

is_osx = function() {
  Sys.info()["sysname"] == "Darwin"
}

is_linux = function() {
  identical(tolower(Sys.info()[["sysname"]]), "linux")
}


#' @title install diagnostic
#'
#' @description Print information about the 'torch' installation and the compute devices.
#' @details If the trouble shooting guide \code{\link{installation_help}} did not help with
#'   the installation, please create an issue on
#'   \href{https://github.com/TheoreticalEcology/s-jSDM/issues}{issue tracker} with the
#'   output of this function as a quote.
#'
#' @seealso \code{\link{installation_help}}, \code{\link{install_sjSDM}}
#'
#' @return
#'
#' No return value, called to extract dependency information.
#'
#' @export
install_diagnostic = function() {
  cat("sjSDM:              ", as.character(utils::packageVersion("sjSDM")), "\n")
  cat("torch:              ", as.character(utils::packageVersion("torch")), "\n")
  cat("libtorch installed: ", is_torch_available(), "\n")
  if(is_torch_available()) {
    cat("CUDA available:     ", torch::cuda_is_available(), "\n")
    if(torch::cuda_is_available()) cat("CUDA devices:       ", torch::cuda_device_count(), "\n")
    cat("MPS available:      ", torch::backends_mps_is_available(), "\n")
  }
  cat("\n\n\n")
  print(utils::sessionInfo())
}
