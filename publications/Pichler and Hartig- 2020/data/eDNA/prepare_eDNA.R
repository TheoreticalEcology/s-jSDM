# prepare eDNA dataset
# Code was taken and modified from https://github.com/tobiasgf/man_vs_machine/ 

all_tabs <- c("ds_survey_full.txt","ds_survey_agaricomycetes.txt","ds_survey_agaricales.txt","ds_soilsurvey_full.txt","ds_soilsurvey_agaricomycetes.txt","ds_soilsurvey_agaricales.txt","ds_dna_full.txt","ds_dna_agaricomycetes.txt","ds_dna_agaricales.txt")
read_otuk <- c(
  "data/ds_survey_full.txt"              ,
  "data/ds_survey_agaricomycetes.txt"    ,
  "data/ds_survey_agaricales.txt"        ,
  "data/ds_soilsurvey_full.txt"          ,
  "data/ds_soilsurvey_agaricomycetes.txt",
  "data/ds_soilsurvey_agaricales.txt"    ,
  "data/ds_dna_full.txt"                 ,
  "data/ds_dna_agaricomycetes.txt"       ,
  "data/ds_dna_agaricales.txt" 
)
com_dis_ds <- list() # 
ttab_ds <- list()
for(i in 1:length(read_otuk)){
  tab_ds <- read.csv(read_otuk[i],sep='\t',header=T,as.is=TRUE,row.names = 1) #read table
  tab_ds <- tab_ds[,-c(29,36,54,103,115)]
  tab_ds[tab_ds>1] <- 1
  ttab_ds[[i]] <- t(tab_ds)
}


env <- read.csv(here::here("in_data","environmental_variables.txt"),sep='\t',header=T,as.is=TRUE)
env2 <- env[-c(29,36,54,103,115),]

save(ttab_ds, env2, "data/eDNA/SDM_data.RData")