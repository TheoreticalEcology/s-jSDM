\dontrun{
  PA = butterflies$PA
  E = butterflies$env
  LatLon = butterflies$lat_lon
  
  m = sjSDM(PA, 
            scale(E), 
            spatial = DNN(scale(LatLon), formula = ~0+.), 
            se = TRUE,
            iter = 20L, # increase to 100
            step_size = 200L,
            verbose = FALSE)
  summary(m)
  plot(m)

}
