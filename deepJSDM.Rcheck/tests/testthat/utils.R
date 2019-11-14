skip_if_no_torch <- function(required_version = NULL) {
  if (!is_torch_available())
    skip("required torch version not available for testing")
}


test_succeeds <- function(desc, expr, required_version = NULL) {
  
  py_capture_output <- NULL
  if (reticulate::py_module_available("IPython")) {
    IPython <- reticulate::import("IPython")
    py_capture_output <- IPython$utils$capture$capture_output
  }
  
  invisible(
    capture.output({
      test_that(desc, {
        skip_if_no_torch()
        with(py_capture_output(), {
          expect_error(force(expr), NA)
        })
      })
    })
  )
}

test_call_succeeds <- function(call_name, expr, required_version = NULL) {
  test_succeeds(paste(call_name, "call succeeds"), expr)
}
