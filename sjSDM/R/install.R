#' Install sjSDM and its dependencies
#'
#' @param conda path to conda
#' @param version version = "cpu" for CPU version, or "gpu" for GPU version. (note MacOS users have to install 'cuda' binaries by themselves)
#' @param restart_session Restart R session after installing (note this will
#'   only occur within RStudio).
#' @param ... not supported
#' 
#' @return 
#' 
#' No return value, called for side effects (installation of 'python' dependencies).
#'
#' @export
install_sjSDM = function(conda = "auto",
                         version = c("cpu", "gpu"),
                         restart_session = TRUE, ...) {
  
  version = match.arg(version)
  
  method = "conda"
  envname = "r-sjsdm"
  
  # install conda if not installed 
  conda = tryCatch(reticulate::conda_binary(), error = function(e) e)
  if(inherits(conda, "error")) {
    reticulate::install_miniconda(update = TRUE)
  }
  
  # get python dependencies
  pkgs = get_pkgs(version = version)
  
  # torch will be installed via pip on macOS because of mkl dependencies
  pip = FALSE
  channel = "pytorch"
  if(is_osx()) {
    pip = TRUE
    channel = NULL
  }
  
  # install dependencies
  error = tryCatch({
    
    reticulate::py_install(
      pkgs$conda,
      envname = envname,
      method = "conda",
      conda = "auto",
      python_version = "3.7.1",
      channel = channel,
      pip = pip
    )
    
    reticulate::py_install(
      c("numpy", "pyro-ppl", "torch_optimizer", "madgrad"),
      envname = envname,
      method = "conda",
      conda = "auto",
      pip = TRUE
    )
  }, error = function(e) e)
  
  # check if instllation was successfull
  if(!inherits(error, "error")) {
    cli::cli_alert_success("\nInstallation complete.\n\n")
    
    if (restart_session && rstudioapi::hasFun("restartSession"))
      rstudioapi::restartSession()
    
    invisible(NULL)
  } else {
    cli::cli_alert_danger("\nInstallation failed... try to install manually PyTorch (install instructions: https://github.com/TheoreticalEcology/s-jSDM\n")
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

get_pkgs = function(version="cpu") {
  
  channel = "pytorch"
  if(is_windows() || is_linux() || is_unix()) {
    package = list()
    package$conda =
      switch(version,
             cpu = "pytorch torchvision torchaudio cpuonly",
             gpu = "pytorch torchvision torchaudio cudatoolkit=11.3")
   }

  if(is_osx()) {
    package = list()
    package$conda = "torch torchvision torchaudio"
    if(version == "gpu") message("PyTorch does not provide cuda binaries for macOS, installing CPU version...\n")
  }
  
  packages = strsplit(unlist(package), " ", fixed = TRUE)
  return(packages)
}

#' @title install diagnostic
#' 
#' @description Print information about available conda environments, python configs, and pytorch versions. 
#' @details If the trouble shooting guide \code{\link{installation_help}} did not help with the installation, please create an issue on \href{https://github.com/TheoreticalEcology/s-jSDM/issues}{issue tracker} with the output of this function as a quote. 
#' 
#' @seealso \code{\link{installation_help}}, \code{\link{install_sjSDM}}
#' 
#' @return
#' 
#' No return value, called to extract dependency information.
#' 
#' @export
install_diagnostic = function() {
  conda_envs = reticulate::conda_list()
  
  conda = reticulate::conda_binary()
  
  configs = ""
  conda_info = ""
  suppressWarnings({
    try({
      for(n in conda_envs$name) {
        configs = paste0(configs, "\n\n\nENV: ", n)
        configs = paste0(configs, "\n\n\ntorch:\n", paste0(system(paste0(conda, " list -n" ,n, " torch*"), intern = TRUE) ,collapse = "\n" ))
        configs = paste0(configs, "\n\n\nnumpy:\n", paste0(system(paste0(conda, " list -n" ,n, " numpy*"),  intern = TRUE)  ,collapse = "\n" )   )
      }
      conda_info = paste0(system(paste0(conda, " info"), intern=TRUE), collapse = "\n")
    }, silent = TRUE)})
  
  print(conda_envs)
  cat(configs)
  cat("\n\n\n")
  print(reticulate::py_config())
  cat("\n\n\n")
  cat(conda_info)
}

