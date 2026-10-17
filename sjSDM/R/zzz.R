
# Following https://stackoverflow.com/questions/12598242/global-variables-in-packages-in-r
# We will use an environment for global variables
pkg.env = new.env()
pkg.env$name = "sjSDM"
pkg.env$torch_available = FALSE


.onLoad = function(libname, pkgname){
  msg( text_col( cli::rule(left = "Attaching sjSDM", right = utils::packageVersion("sjSDM")) ), startup = TRUE)

  check = check_installation()
  pkg.env$torch_available = all(check[, 2] == "1")

  info = apply(check, 1, function(d) paste0(d[1], " ", crayon::black(d[3]), "\n"))
  msg(info, startup = TRUE)

  if(!pkg.env$torch_available) {
    msg( crayon::red( "'torch' is installed but libtorch is missing:" ), startup = TRUE)
    msg(c("\t1. Run torch::install_torch() to download the libtorch binaries \n",
          "\t2. Installation trouble shooting guide: ?installation_help \n",
          paste0("\t3. If 1) and 2) did not help, please create an issue on ",
                 crayon::italic(crayon::blue("<https://github.com/TheoreticalEcology/s-jSDM/issues>")),
                 " (see ?install_diagnostic) ")), startup = TRUE)
  }
  invisible()
}

# copied from the tidyverse package
msg <- function(..., startup = FALSE) {
  if (startup) {
    if (!isTRUE(getOption("tidyverse.quiet"))) {
      packageStartupMessage(text_col(...))
    }
  } else {
    message(text_col(...))
  }
}

# copied from the tidyverse package
text_col <- function(x) {
  # If RStudio not available, messages already printed in black
  if (!rstudioapi::isAvailable()) {
    return(x)
  }
  if (!rstudioapi::hasFun("getThemeInfo")) {
    return(x)
  }
  theme <- rstudioapi::getThemeInfo()

  if (isTRUE(theme$dark)) crayon::white(x) else crayon::black(x)

}
