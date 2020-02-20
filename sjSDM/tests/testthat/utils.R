skip_if_no_torch = function(required_version = NULL) {
  if (!is_torch_available())
    skip("required torch version not available for testing")
}
