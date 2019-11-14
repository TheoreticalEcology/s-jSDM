library(sp)
library(rgdal)
# WGS84 (lat/long) and OSGB36 (UK national grid) coordinate systems in proj4
wgs84 <- "+init=epsg:4326"
osgb36 <- "+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.999601272 +x_0=400000 +y_0=-100000 +datum=OSGB36 +units=m +no_defs +ellps=airy +towgs84=446.448,-125.157,542.060,0.1502,0.2470,0.8421,-20.4894"
usacaeac <- "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96 +x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"

# load mosquito data and extract coordinates
mosq <- read.csv('C:/Users/davidpw/Desktop/Compiled datasets/Birds/Birds-Compiled.csv',
                stringsAsFactors = FALSE)
coord_cols <- c('Longitude', 'Latitude')
coords <- mosq[, coord_cols]

# define SpatialPoints object with correct coordinate system
coords_sp <- SpatialPoints(coords, CRS(wgs84))

# transform to OSGB36 and extract coordinates matrix
coords_sp_osgb <- spTransform(coords_sp, CRS(usacaeac))
coords_osgb <- coords_sp_osgb@coords

# replace in mosq, these are all measured in metres
mosq[, coord_cols] <- coords_osgb

# get maximum distance (in metres) between sampling lcoations within each site
max <- -Inf

# largest pairwise distance
max_site <- max(dist(mosq[, coord_cols]))
  
# update the counter
max <- max(max, max_site)
  
# rescale the coordinates, so that the maximum intra-site site distance is 1
mosq[, coord_cols] <- mosq[, coord_cols] / max

write.csv(mosq[, coord_cols],"data/Birds/xy.csv")
