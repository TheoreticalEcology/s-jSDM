\dontrun{
  PA = eucalypts$PA
  E = eucalypts$env
  LatLon = eucalypts$lat_lon
  
  m = sjSDM(PA, 
            scale(E), 
            spatial = DNN(scale(LatLon), formula = ~0+.), 
            se = TRUE,
            verbose = FALSE)
  summary(m)
  plot(m)
  
}
