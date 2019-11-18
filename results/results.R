cpu = readRDS(file = "results/cpu_dmvp2.RDS")
gpu = readRDS(file = "results/gpu_dmvp2.RDS")
gllvm = readRDS(file = "results/gllvm.RDS")
bc = readRDS(file = "results/BayesComm.RDS")
hmsc = readRDS(file = "results/hmsc.RDS")



apply(gpu$result_corr_acc[complete.cases(gpu$result_corr_acc),],1,mean)
apply(cpu$result_corr_acc[complete.cases(cpu$result_corr_acc),],1,mean)
apply(gllvm$result_corr_acc[complete.cases(gllvm$result_corr_acc),],1,mean)
apply(bc$result_corr_acc[complete.cases(bc$result_corr_acc),],1,mean)
apply(hmsc$result_corr_acc[complete.cases(hmsc$result_corr_acc),],1,mean)


len = ncol(gpu$auc[[100]][[1]]$pred)
mean(sapply(1:len, function(l) Metrics::auc(gpu$auc[[100]][[1]]$true[,l], gpu$auc[[100]][[1]]$pred[,l])))


number = setup$species
cpu_log = log(apply(cpu$result_time[complete.cases(cpu$result_time),],1,mean))
gpu_log = log(apply(gpu$result_time[complete.cases(gpu$result_time),],1,mean))
gllvm_log = log(apply(gllvm$result_time[complete.cases(gllvm$result_time),],1,mean))
bc_log = log(apply(bc$result_time[complete.cases(bc$result_time),],1,mean))
hmsc_log = log(apply(hmsc$result_time[complete.cases(hmsc$result_time),],1,mean))



######### Run time ######### 

spar = 0.5
e = 5L
lwd = 2.0
pdf(file = "Fig1_speed.pdf", width = 5.5, height = 3.5)
par(mfrow = c(1,3), mgp = c(3,0.6,0))
for(i in (as.character(unique(number)))[c(1,3,5)]){
#for(e in c(3,5,7)){
  lineT = rep(1,5) #1:5
  names(lineT) =  (as.character(unique(number)))
  plot(NULL, NULL, xlim = c(0,nrow(setup)-4), ylim = c(-0.5,8), xaxt = "n", main = paste0(as.numeric(i)*100, "% species"),yaxt = "n", xlab = "Number of Sites", ylab = "Log(time) in seconds")
  # 1 -> 150
  tt = seq(0.01, log(4000), length.out = 10)
  tt = exp(tt)
  #axis(1)
  axis(2, at = round(log(tt),2),labels = c(round(tt,1)), las = 2)
  axis(1, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites))
  axis(3, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites)* as.numeric(i))
  for(k in seq(1, nrow(setup), by = 15)[-1]){
    abline(v = k, col = "grey")
  }
  
  #for(i in (as.character(unique(number)))[c(1,3,5)]){
    cat(i, "\n")
    X = (1:nrow(setup))[as.character(number) == i & setup$env == e]
    tmp_cpu = cpu_log[as.character(number) == i & setup$env == e]
    tmp_gpu = gpu_log[as.character(number) == i & setup$env == e]
    tmp_gllvm =gllvm_log[as.character(number) == i & setup$env == e]
    tmp_bc = bc_log[as.character(number) == i & setup$env == e]
    tmp_hmsc =hmsc_log[as.character(number) == i & setup$env == e]
    
    tmp_cpu = tmp_cpu[complete.cases(tmp_cpu)]
    tmp_gllvm = tmp_gllvm[complete.cases(tmp_gllvm)]
    tmp_bc = tmp_bc[complete.cases(tmp_bc)]
    tmp_hmsc = tmp_hmsc[complete.cases(tmp_hmsc)]

    lines(smooth.spline(x = X[1:length(tmp_cpu)], tmp_cpu,spar = spar), lwd=lwd, lty = lineT[[i]])
    lines(smooth.spline(x = X[1:length(tmp_gpu)], tmp_gpu,spar = spar), col = "red", lwd= lwd, lty = lineT[[i]])
    lines(smooth.spline(x = X[1:length(tmp_gllvm)], tmp_gllvm,spar = spar), col = "blue", lwd= lwd, lty = lineT[[i]])
    try(lines(smooth.spline(x = X[1:length(tmp_bc)], tmp_bc,spar = spar), col = "green", lwd= lwd, lty = lineT[[i]]))
    try(lines(smooth.spline(x = X[1:length(tmp_hmsc)], tmp_hmsc,spar = spar), col = "violet", lwd= lwd, lty = lineT[[i]]))
  
 # legend("topleft", legend = c("10% Species",  "30% Species",  "50% Species"), lty = c(1,3,5), bty = "n")
  legend("topleft", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm", "bayesComm", "Hmsc"), col = c("red", "black", "blue", "green", "violet"), bty="n", lty = 1)
}
  #}
dev.off()



######### Covariance accuracy ######### 
number = setup$species
cpu_cov = (apply(cpu$result_corr_acc[complete.cases(cpu$result_corr_acc),],1,mean))
gpu_cov = (apply(gpu$result_corr_acc[complete.cases(gpu$result_corr_acc),],1,mean))
gllvm_cov = (apply(gllvm$result_corr_acc[complete.cases(gllvm$result_corr_acc),],1,mean))
bc_cov = (apply(bc$result_corr_acc[complete.cases(bc$result_corr_acc),],1,mean))
hmsc_cov = (apply(hmsc$result_corr_acc[complete.cases(hmsc$result_corr_acc),],1,mean))



par(mfrow = c(1,3))
for(i in (as.character(unique(number)))[c(1,3,5)]){
#for(e in c(3,5,7)){
e = 5L
  lineT = rep(1,5) #1:5
  names(lineT) =  (as.character(unique(number)))
  plot(NULL, NULL, xlim = c(0,nrow(setup)), ylim = c(0.5,1), xaxt = "n", yaxt = "n", xlab = "Number of Sites", ylab = "COV accuracy", main = paste0(as.numeric(i)*100, "% species"))
  # 1 -> 150
  #axis(1)
  axis(2)
  axis(1, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites))
  axis(3, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites)* as.numeric(i))
  
  for(k in seq(1, nrow(setup), by = 15)[-1]){
    abline(v = k, col = "grey")
  }
  
    cat(i, "\n")
    X = (1:nrow(setup))[as.character(number) == i & setup$env == e]
    tmp_cpu = cpu_cov[as.character(number) == i & setup$env == e]
    tmp_gpu = gpu_cov[as.character(number) == i & setup$env == e]
    tmp_gllvm =gllvm_cov[as.character(number) == i& setup$env == e]
    tmp_bc =bc_cov[as.character(number) == i& setup$env == e]
    tmp_hmsc =hmsc_cov[as.character(number) == i& setup$env == e]
    
    
    
    tmp_cpu = tmp_cpu[complete.cases(tmp_cpu)]
    tmp_gllvm = tmp_gllvm[complete.cases(tmp_gllvm)]
    tmp_bc = tmp_bc[complete.cases(tmp_bc)]
    tmp_hmsc = tmp_hmsc[complete.cases(tmp_hmsc)]
    
    
    
    lines(smooth.spline(x = X[1:length(tmp_cpu)], tmp_cpu,spar = 0.5), lwd= 1.5, lty = lineT[[i]])
    lines(smooth.spline(x = X[1:length(tmp_gpu)], tmp_gpu,spar = 0.5), col = "red", lwd= 1.5, lty = lineT[[i]])
    lines(smooth.spline(x = X[1:length(tmp_gllvm)], tmp_gllvm,spar = 0.5), col = "blue", lwd= 1.5, lty = lineT[[i]])
    lines(smooth.spline(x = X[1:length(tmp_bc)], tmp_bc,spar = 0.5), col = "green", lwd= 1.5, lty = lineT[[i]])
    lines(smooth.spline(x = X[1:length(tmp_hmsc)], tmp_hmsc,spar = 0.5), col = "violet", lwd= 1.5, lty = lineT[[i]])
    legend("topleft", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm", "bayesComm", "Hmsc"), col = c("red", "black", "blue", "green", "violet"), bty="n", lty = 1)
    
  }
 # legend("bottomright", legend = c("10% Species", "20% Species", "30% Species", "40% Species", "50% Species"), lty = 1:5, bty = "n")
  #legend("topleft", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm"), col = c("red", "black", "blue"), bty="n", lty = 1)
# }


  

######### ENV accuracy ######### 
  
par(mfrow = c(1,1))
number = setup$species
cpu_cov = (apply(cpu$result_env[complete.cases(cpu$result_env),],1,mean))
gpu_cov = (apply(gpu$result_env[complete.cases(gpu$result_env),],1,mean))
gllvm_cov = (apply(gllvm$result_env[complete.cases(gllvm$result_env),],1,mean))
bc_cov = (apply(bc$result_env[complete.cases(bc$result_env),],1,mean))
hmsc_cov = (apply(hmsc$result_env[complete.cases(hmsc$result_env),],1,mean))


#for(e in c(3,5,7)){
e = 5L
lineT = rep(1,5) #1:5
names(lineT) =  (as.character(unique(number)))
par(mfrow = c(1,3))
for(i in (as.character(unique(number)))[c(1,3,5)]){
plot(NULL, NULL, xlim = c(0,nrow(setup)), ylim = c(0.5,1), xaxt = "n", yaxt = "n", xlab = "Number of Sites", ylab = "ENV sign accuracy", main = paste0(as.numeric(i)*100, "% species"))
# 1 -> 150
#axis(1)
axis(2)
axis(1, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites))
axis(3, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites)* as.numeric(i))

for(k in seq(1, nrow(setup), by = 15)[-1]){
  abline(v = k, col = "grey")
}

#for(i in (as.character(unique(number)))){
  cat(i, "\n")
  X = (1:nrow(setup))[as.character(number) == i & setup$env == e]
  tmp_cpu = cpu_cov[as.character(number) == i & setup$env == e]
  tmp_gpu = gpu_cov[as.character(number) == i & setup$env == e]
  tmp_gllvm =gllvm_cov[as.character(number) == i& setup$env == e]
  tmp_bc =bc_cov[as.character(number) == i& setup$env == e]
  tmp_hmsc =hmsc_cov[as.character(number) == i& setup$env == e]
  
  
  tmp_cpu = tmp_cpu[complete.cases(tmp_cpu)]
  tmp_gllvm = tmp_gllvm[complete.cases(tmp_gllvm)]
  tmp_bc = tmp_bc[complete.cases(tmp_bc)]
  tmp_hmsc = tmp_hmsc[complete.cases(tmp_hmsc)]
  
  
  lines(smooth.spline(x = X[1:length(tmp_cpu)], tmp_cpu,spar = 0.5), lwd= 1.5, lty = lineT[[i]])
  lines(smooth.spline(x = X[1:length(tmp_gpu)], tmp_gpu,spar = 0.5), col = "red", lwd= 1.5, lty = lineT[[i]])
  lines(smooth.spline(x = X[1:length(tmp_gllvm)], tmp_gllvm,spar = 0.5), col = "blue", lwd= 1.5, lty = lineT[[i]])
  lines(smooth.spline(x = X[1:length(tmp_bc)], tmp_bc,spar = 0.5), col = "green", lwd= 1.5, lty = lineT[[i]])
  lines(smooth.spline(x = X[1:length(tmp_hmsc)], tmp_hmsc,spar = 0.5), col = "violet", lwd= 1.5, lty = lineT[[i]])
  legend("bottomright", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm", "bayesComm", "Hmsc"), col = c("red", "black", "blue", "green", "violet"), bty="n", lty = 1)
}
# }


 cpu_cov_sd = (apply(cpu$result_corr_acc[complete.cases(cpu$result_corr_acc),],1,sd))
 gpu_cov_sd = (apply(gpu$result_corr_acc[complete.cases(gpu$result_corr_acc),],1,sd))
 gllvm_cov_sd = (apply(gllvm$result_corr_acc[complete.cases(gllvm$result_corr_acc),],1,sd))



######### ENV RMSE ######### 
# number = setup$species
# cpu_cov = (apply(cpu$result_rmse_env[complete.cases(cpu$result_rmse_env),],1,mean))
# gpu_cov = (apply(gpu$result_rmse_env[complete.cases(gpu$result_rmse_env),],1,mean))
# gllvm_cov = (apply(gllvm$result_rmse_env[complete.cases(gllvm$result_rmse_env),],1,mean))
# 
# 
# #for(e in c(3,5,7)){
# lineT = 1:5
# names(lineT) =  (as.character(unique(number)))
# plot(NULL, NULL, xlim = c(0,nrow(setup)), ylim = c(0.0,100), xaxt = "n", yaxt = "n", xlab = "Number of Sites", ylab = "ENV sign accuracy")
# # 1 -> 150
# #axis(1)
# axis(2)
# axis(1, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites))
# for(i in seq(1, nrow(setup), by = 15)[-1]){
#   abline(v = i, col = "grey")
# }
# 
# for(i in (as.character(unique(number)))){
#   cat(i, "\n")
#   X = (1:nrow(setup))[as.character(number) == i ]
#   tmp_cpu = cpu_cov[as.character(number) == i ]
#   tmp_gpu = gpu_cov[as.character(number) == i ]
#   tmp_gllvm =gllvm_cov[as.character(number) == i]
#   
#   tmp_cpu = tmp_cpu[complete.cases(tmp_cpu)]
#   tmp_gllvm = tmp_gllvm[complete.cases(tmp_gllvm)]
#   
#   lines(smooth.spline(x = X[1:length(tmp_cpu)], tmp_cpu,spar = 0.5), lwd= 1.5, lty = lineT[[i]])
#   lines(smooth.spline(x = X[1:length(tmp_gpu)], tmp_gpu,spar = 0.5), col = "red", lwd= 1.5, lty = lineT[[i]])
#   lines(smooth.spline(x = X[1:length(tmp_gllvm)], tmp_gllvm,spar = 0.5), col = "blue", lwd= 1.5, lty = lineT[[i]])
# }
# legend("bottomright", legend = c("10% Species", "20% Species", "30% Species", "40% Species", "50% Species"), lty = 1:5, bty = "n")
# legend("topleft", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm"), col = c("red", "black", "blue"), bty="n", lty = 1)
 
 
 
##### Predictive Performance ####
 
auc_gpu = vector("list", length(nrow(setup)))
for(k in 1:nrow(setup)){
  auc_gpu[[k]] = t(sapply(1:10, function(j) sapply(1:ncol(gpu$auc[[k]][[j]]$pred), function(i) Metrics::auc(gpu$auc[[k]][[j]]$true[,i], gpu$auc[[k]][[j]]$pred[,i]))))
}
hist(unlist(lapply(auc_gpu, function(tmp) mean(apply(tmp, 1,mean)))))
