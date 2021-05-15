context("examples")

source("utils.R")

testthat::test_that("sjSDM examples", {
  skip_if_no_torch()
  testthat::skip_on_ci()
  testthat::skip_on_cran()
  
  library(sjSDM)
  path = system.file("examples", package = "sjSDM")
  to_do = list.files(path, full.names = TRUE)
  run_raw = function(rr) suppressWarnings(eval(str2expression(rr[c(-1, -length(rr))])))
  
  raw = readLines(to_do[1], warn = FALSE)
  testthat::expect_error(run_raw(raw), NA, info = paste0(raw, collapse = "\n"))
  
  raw = readLines(to_do[2], warn = FALSE)
  testthat::expect_error(run_raw(raw), NA, info = paste0(raw, collapse = "\n"))
  
  raw = readLines(to_do[3], warn = FALSE)
  testthat::expect_error(run_raw(raw), NA, info = paste0(raw, collapse = "\n"))
  
  raw = readLines(to_do[4], warn = FALSE)
  testthat::expect_error(run_raw(raw), NA, info = paste0(raw, collapse = "\n"))
})

