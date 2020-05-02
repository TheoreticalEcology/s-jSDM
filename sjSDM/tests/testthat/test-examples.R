context("examples")

source("utils.R")

testthat::test_that("sjSDM examples", {
  skip_if_no_torch()
  
  library(sjSDM)
  path = system.file("examples", package = "sjSDM")
  to_do = list.files(path, full.names = TRUE)
  run_raw = function(rr) suppressWarnings(eval(str2expression(rr[c(-1, -length(rr))])))
  
  raw = readLines(to_do[1])
  testthat::expect_error(run_raw(raw), NA, info = paste0(raw, collapse = "\n"))
  
  raw = readLines(to_do[2])
  testthat::expect_error(run_raw(raw), NA, info = paste0(raw, collapse = "\n"))
  
  raw = readLines(to_do[3])
  testthat::expect_error(run_raw(raw), NA, info = paste0(raw, collapse = "\n"))
})

