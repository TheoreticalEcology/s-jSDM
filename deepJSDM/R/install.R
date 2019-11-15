# Taken and modified from: https://github.com/f0nzie/rTorch/blob/master/R/install.R

#' Install PyTorch and its dependencies
#'
#' @inheritParams reticulate::conda_list
#'
#' @param method Installation method. By default, "auto" automatically finds a
#'   method that will work in the local environment. Change the default to force
#'   a specific installation method. Note that the "virtualenv" method is not
#'   available on Windows (as this isn't supported by PyTorch). Note also
#'   that since this command runs without privillege the "system" method is
#'   available only on Windows.
#'
#' @param version PyTorch version to install. Specify "default" to install
#'   the CPU version of the latest release. Specify "gpu" to install the GPU
#'   version of the latest release.
#'
#'   You can also provide a full major.minor.patch specification (e.g. "1.1.0"),
#'   appending "-gpu" if you want the GPU version (e.g. "1.1.0-gpu").
#'
#'   Alternatively, you can provide the full URL to an installer binary (e.g.
#'   for a nightly binary).
#'
#' @param envname Name of Python environment to install within
#'
#' @param extra_packages Additional Python packages to install along with
#'   PyTorch
#'
#' @param restart_session Restart R session after installing (note this will
#'   only occur within RStudio).
#'
#' @param conda_python_version the python version installed in the created conda
#'   environment. Python 3.6 is installed by default.
#'
#' @param pip logical
#' @param channel conda channel
#'
#' @param ... other arguments passed to [reticulate::conda_install()] or
#'   [reticulate::virtualenv_install()].
#'
#' @importFrom jsonlite fromJSON
#'
#' @export
install_pytorch <- function(method = c("conda", "virtualenv", "auto"),
                            conda = "auto",
                            version = "default",
                            envname = "r-torch",
                            extra_packages = NULL,
                            restart_session = TRUE,
                            conda_python_version = "3.6",
                            pip = FALSE,
                            channel = "pytorch",
                            ...) {
  
  # verify 64-bit
  if (.Machine$sizeof.pointer != 8) {
    stop("Unable to install PyTorch on this platform.",
         "Binary installation is only available for 64-bit platforms.")
  }
  
  method <- match.arg(method)
  
  # unroll version
  ver <- parse_torch_version(version)
  
  version <- ver$version
  gpu <- ver$gpu
  package <- ver$package
  
  # Packages in this list should always be installed.
  
  default_packages <- c("torchvision-cpu")
  
  if (!is.na(version) && substr(version, 1, 4) %in% c("1.1.0", "1.1", "1.1.0")) {
    default_packages <- c(default_packages, "pandas")
    # install tfp-nightly
  } else if (is.na(version) ||(substr(version, 1, 4) %in% c("2.0.") || version == "nightly")) {
    default_packages <- c(default_packages, "numpy")
  }
  
  extra_packages <- unique(c(default_packages, extra_packages))
  
  
  # Main OS verification.
  if (is_osx() || is_linux()) {
    
    if (method == "conda") {
      install_conda(
        package = package,
        extra_packages = extra_packages,
        envname = envname,
        conda = conda,
        conda_python_version = conda_python_version,
        channel = channel,
        pip = pip,
        ...
      )
    } else if (method == "virtualenv" || method == "auto") {
      install_virtualenv(
        package = package,
        extra_packages = extra_packages,
        envname = envname,
        ...
      )
    }
    
  } else if (is_windows()) {
    
    if (method == "virtualenv") {
      stop("Installing PyTorch into a virtualenv is not supported on Windows",
           call. = FALSE)
    } else if (method == "conda" || method == "auto") {
      
      install_conda(
        package = package,
        extra_packages = extra_packages,
        envname = envname,
        conda = conda,
        conda_python_version = conda_python_version,
        channel = channel,
        pip = pip,
        ...
      )
      
    }
    
  } else {
    stop("Unable to install PyTorch on this platform. ",
         "Binary installation is available for Windows, OS X, and Linux")
  }
  
  message("\nInstallation complete.\n\n")
  
  if (restart_session && rstudioapi::hasFun("restartSession"))
    rstudioapi::restartSession()
  
  invisible(NULL)
}

install_conda <- function(package, extra_packages, envname, conda,
                          conda_python_version, channel, pip, ...) {
  
  # find if environment exists
  envname_exists <- envname %in% reticulate::conda_list(conda = conda)$name
  
  # remove environment
  if (envname_exists) {
    message("Removing ", envname, " conda environment... \n")
    reticulate::conda_remove(envname = envname, conda = conda)
  }
  
  
  message("Creating ", envname, " conda environment... \n")
  reticulate::conda_create(
    envname = envname, conda = conda,
    packages = paste0("python=", conda_python_version)
  )
  
  message("Installing python modules...\n")
  # rTorch::conda_install(envname="r-torch-37", packages="pytorch-cpu",
  #         channel = "pytorch", conda="auto", python_version = "3.7")
  rTorch::conda_install(
    envname = envname,
    packages = c(package, extra_packages),
    conda = conda,
    pip = pip,       # always use pip since it's the recommend way.
    channel = channel,
    ...
  )
  
}

install_virtualenv <- function(package, extra_packages, envname, ...) {
  
  # find if environment exists
  envname_exists <- envname %in% reticulate::virtualenv_list()
  
  # remove environment
  if (envname_exists) {
    message("Removing ", envname, " virtualenv environment... \n")
    reticulate::virtualenv_remove(envname = envname, confirm = FALSE)
  }
  
  message("Creating ", envname, " virtualenv environment... \n")
  reticulate::virtualenv_create(envname = envname)
  
  message("Installing python modules...\n")
  reticulate::virtualenv_install(
    envname = envname,
    packages = c(package, extra_packages),
    ...
  )
  
}

parse_torch_version <- function(version) {
  
  default_version <- "1.1"
  
  ver <- list(
    version = default_version,
    gpu = FALSE,
    package = NULL
  )
  
  if (version == "default") {
    
    ver$package <- paste0("pytorch-cpu==", ver$version)
    
    # default gpu version
  } else if (version == "gpu") {
    
    ver$gpu <- TRUE
    ver$package <- paste0("pytorch-gpu==", ver$version)
    
    # gpu qualifier provided
  } else if (grepl("-gpu$", version)) {
    
    split <- strsplit(version, "-")[[1]]
    ver$version <- split[[1]]
    ver$gpu <- TRUE
    
    # full path to whl.
  } else if (grepl("^.*\\.whl$", version)) {
    
    ver$gpu <- NA
    ver$version <- NA
    
    if (grepl("^http", version))
      ver$package <- version
    else
      ver$package <- normalizePath(version)
    
    # another version
  } else {
    
    ver$version <- version
    
  }
  
  # find the right package for nightly and other versions
  if (is.null(ver$package)) {
    
    if (ver$version == "nightly") {
      
      if (ver$gpu) {
        ver$package <- "pytorch-nightly-gpu"
      } else {
        ver$package <- "pytorch-nightly"
      }
      
    } else {
      
      if (ver$gpu) {
        ver$package <- paste0("pytorch-gpu==", ver$version)
      } else {
        ver$package <- paste0("pytorch-cpu==", ver$version)
      }
      
    }
    
  }
  
  ver
}
