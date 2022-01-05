source("utils.R")

testthat::test_that("sjSDM examples", {
  skip_if_no_torch()
  testthat::skip_on_ci()
  testthat::skip_on_cran()
  
  library(sjSDM)
  path = system.file("examples", package = "sjSDM")
  to_do = list.files(path, full.names = TRUE)
  run_raw = function(rr) suppressWarnings(eval(str2expression(rr[c(-1, -length(rr))])))
  for(i in 1:length(to_do)) {
    raw = readLines(to_do[i])
    testthat::expect_error(run_raw(raw), NA, info = paste0(raw, collapse = "\n"))
  }
})

