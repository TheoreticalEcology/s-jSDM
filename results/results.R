cpu = readRDS(file = "results/cpu_dmvp3.RDS")
gpu = readRDS(file = "results/gpu_dmvp3.RDS")
gllvm = readRDS(file = "results/gllvm.RDS")
bc = readRDS(file = "results/BayesComm.RDS")
hmsc = readRDS(file = "results/hmsc.RDS")

bc = readRDS(file = "results/BayesCommDiag.RDS")
hmsc = readRDS(file = "results/hmscDiag.RDS")


pdf(file = "figures/Fig_1.pdf", width = 9, height = 9.8)


addA = function(col, alpha = 0.25) apply(sapply(col, col2rgb)/255, 2, function(x) rgb(x[1], x[2], x[3], alpha=alpha)) 
mean_conf = function(sites, mat, col = "red", alpha = 0.1, spar = 0.4) {
  sites2 = sites[complete.cases(mat)]
  mat = mat[complete.cases(mat),]
  m = apply(mat, 1, mean)
  sd = apply(mat, 1, sd)
  upper = smooth.spline(y = m + sd, x = sites2, spar = spar)$y
  lower = smooth.spline(y = m - sd, x = sites2, spar = spar)$y
  polygon(c(sites2, rev(sites2)), c(upper, rev(lower)), border = NA, col =addA(col, alpha))
  lines(smooth.spline(y = m, x = sites2, spar = spar), col = col, lwd = 2.0)
}
par(mfrow = c(4,3), mgp = c(2.3,0.6,0), mar = c(0.6, 2.3, 0.6,0.7), oma = c(3,2.0,3,0))
######### Run time ######### 


number = setup$species
spar = 0.5
e = 5L
lwd = 2.2
for(i in (as.character(unique(number)))[c(1,3,5)]){
  lineT = rep(1,5) #1:5
  names(lineT) =  (as.character(unique(number)))
  if( i == "0.1") {
    ylab = "time in minutes"
  } else {
    ylab = ""
  }
  plot(NULL, NULL, xlim = c(1,nrow(setup)-4), ylim = c(-0.5,8), xaxt = "n", main = "",yaxt = "n", xlab = "", ylab = ylab, xpd = NA, xaxs = "i", font = 2)
  title(paste0(as.numeric(i)*100, "% species"), line = 2, xpd = NA)
  if(i == "0.1") text(-18, 8*1.06, pos = 2, labels = "A", font = 2, xpd = NA, cex = 1.2)
  tt = seq(2, log(4000), length.out = 7)
  tt = exp(tt)
  axis(2, at = round(log(tt),1),labels = c(round(tt/60,1)), las = 2)
  axis(3, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites)* as.numeric(i))
  for(k in seq(1, nrow(setup), by = 15)[-1]){
    abline(v = k, col = "grey")
  }
    cat(i, "\n")
    X = (1:nrow(setup))[as.character(number) == i & setup$env == e]
    mean_conf(X, log(cpu$result_time[as.character(number) == i & setup$env == e, ]), col = "black")
    mean_conf(X, log(gpu$result_time[as.character(number) == i & setup$env == e, ]), col = "red")
    mean_conf(X, log(gllvm$result_time[as.character(number) == i & setup$env == e, ]), col = "blue")
    mean_conf(X, log(bc$result_time[as.character(number) == i & setup$env == e, ]), col = "green")
    mean_conf(X, log(hmsc$result_time[as.character(number) == i & setup$env == e, ]), col = "violet")
  legend("topleft", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm", "bayesComm", "Hmsc"), col = c("red", "black", "blue", "green", "violet"), bty="n", lty = 1)
}


######### Covariance accuracy #########


for(i in (as.character(unique(number)))[c(1,3,5)]){
e = 5L
  lineT = rep(1,5) #1:5
  names(lineT) =  (as.character(unique(number)))
  if( i == "0.1") {
    ylab = "Covariance accuracy"
  } else {
    ylab = ""
  }
  plot(NULL, NULL, xlim = c(1,nrow(setup)), ylim = c(0.5,1), xaxt = "n", yaxt = "n", xlab = "", ylab = ylab, xpd = NA, xaxs = "i")
  axis(2, las = 2)
  if(i == "0.1") text(-18, 1*1.06, pos = 2, labels = "B", font = 2, xpd = NA, cex = 1.2)
  
  for(k in seq(1, nrow(setup), by = 15)[-1]){
    abline(v = k, col = "grey")
  } 
    cat(i, "\n")
     X = (1:nrow(setup))[as.character(number) == i & setup$env == e]
    mean_conf(X, cpu$result_corr_acc[as.character(number) == i & setup$env == e, ], col = "black")
    mean_conf(X, gpu$result_corr_acc[as.character(number) == i & setup$env == e, ], col = "red")
    mean_conf(X, gllvm$result_corr_acc[as.character(number) == i & setup$env == e, ], col = "blue")
    mean_conf(X, bc$result_corr_acc[as.character(number) == i & setup$env == e, ], col = "green")
    mean_conf(X, hmsc$result_corr_acc[as.character(number) == i & setup$env == e, ], col = "violet")
    
    legend("topleft", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm", "bayesComm", "Hmsc"), col = c("red", "black", "blue", "green", "violet"), bty="n", lty = 1)
    
  }


  

######### ENV accuracy ######### 
e = 5L
lineT = rep(1,5) #1:5
names(lineT) =  (as.character(unique(number)))
for(i in (as.character(unique(number)))[c(1,3,5)]){
  
  if( i == "0.1") {
    ylab = "Env sign accuracy"
  } else {
    ylab = ""
  }
plot(NULL, NULL, xlim = c(1,nrow(setup)), ylim = c(0.5,1), xaxt = "n", yaxt = "n", xlab = "", ylab = ylab, xpd = NA, xaxs = "i")
axis(2, las = 2)
if(i == "0.1") text(-18, 1*1.06, pos = 2, labels = "C", font = 2, xpd = NA, cex = 1.2)
for(k in seq(1, nrow(setup), by = 15)[-1]){
  abline(v = k, col = "grey")
}
  cat(i, "\n")
  X = (1:nrow(setup))[as.character(number) == i & setup$env == e]
  mean_conf(X, cpu$result_env[as.character(number) == i & setup$env == e, ], col = "black")
  mean_conf(X, gpu$result_env[as.character(number) == i & setup$env == e, ], col = "red")
  mean_conf(X, gllvm$result_env[as.character(number) == i & setup$env == e, ], col = "blue")
  mean_conf(X, bc$result_env[as.character(number) == i & setup$env == e, ], col = "green")
  mean_conf(X, hmsc$result_env[as.character(number) == i & setup$env == e, ], col = "violet")
  legend("bottomright", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm", "bayesComm", "Hmsc"), col = c("red", "black", "blue", "green", "violet"), bty="n", lty = 1)
}




######### ENV rmse ######### 

e = 5L
lineT = rep(1,5) #1:5
names(lineT) =  (as.character(unique(number)))
for(i in (as.character(unique(number)))[c(1,3,5)]){
  if( i == "0.1") {
    ylab = "Env rmse"
  } else {
    ylab = ""
  }
  plot(NULL, NULL, xlim = c(1,nrow(setup)), ylim = c(0.0,1), xaxt = "n", yaxt = "n", xlab = "Number of Sites", ylab = ylab, xpd = NA, xaxs = "i")
  if(i == "0.1") text(-18, 1*1.06, pos = 2, labels = "D", font = 2, xpd = NA, cex = 1.2)
  
  axis(2, las = 2)
  axis(1, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites))
  for(k in seq(1, nrow(setup), by = 15)[-1]){
    abline(v = k, col = "grey")
  }
  cat(i, "\n")
  X = (1:nrow(setup))[as.character(number) == i & setup$env == e]
  mean_conf(X, cpu$result_rmse_env[as.character(number) == i & setup$env == e, ], col = "black")
  mean_conf(X, gpu$result_rmse_env[as.character(number) == i & setup$env == e, ], col = "red")
  mean_conf(X, gllvm$result_rmse_env[as.character(number) == i & setup$env == e, ], col = "blue")
  mean_conf(X, bc$result_rmse_env[as.character(number) == i & setup$env == e, ], col = "green")
  mean_conf(X, hmsc$result_rmse_env[as.character(number) == i & setup$env == e, ], col = "violet")
  legend("bottomright", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm", "bayesComm", "Hmsc"), col = c("red", "black", "blue", "green", "violet"), bty="n", lty = 1)
}
dev.off()

 



 
##### Predictive Performance ####
 
auc_gpu = auc_cpu = auc_gllvm = auc_bc = auc_hmsc= vector("list", length(nrow(setup)))
for(k in 1:nrow(setup)){
  try({auc_gpu[[k]] = t(sapply(1:10, function(j) sapply(1:ncol(gpu$auc[[k]][[j]]$pred), function(i) Metrics::auc(gpu$auc[[k]][[j]]$true[,i], gpu$auc[[k]][[j]]$pred[,i]))))}, silent = TRUE)
  try({auc_cpu[[k]] = t(sapply(1:10, function(j) sapply(1:ncol(cpu$auc[[k]][[j]]$pred), function(i) Metrics::auc(cpu$auc[[k]][[j]]$true[,i], cpu$auc[[k]][[j]]$pred[,i]))))}, silent = TRUE)
  try({auc_gllvm[[k]] = t(sapply(1:10, function(j) sapply(1:ncol(gllvm$auc[[k]][[j]]$pred), function(i) Metrics::auc(gllvm$auc[[k]][[j]]$true[,i], gllvm$auc[[k]][[j]]$pred[,i]))))}, silent = TRUE)
  try({auc_bc[[k]] = t(sapply(1:10, function(j) sapply(1:ncol(bc$auc[[k]][[j]]$pred), function(i) Metrics::auc(bc$auc[[k]][[j]]$true[,i], bc$auc[[k]][[j]]$pred[,i]))))}, silent = TRUE)
  try({auc_hmsc[[k]] = t(sapply(1:10, function(j) sapply(1:ncol(hmsc$auc[[k]][[j]]$pred), function(i) Metrics::auc(hmsc$auc[[k]][[j]]$true[,i], hmsc$auc[[k]][[j]]$pred[,i]))))}, silent = TRUE)
}

number = setup$species
cpu_auc = t(unlist(sapply(auc_cpu, function(tmp) (apply(tmp, 1,mean)))))
gpu_auc = t(unlist(sapply(auc_gpu, function(tmp) (apply(tmp, 1,mean)))))
gllvm_auc = t(unlist(sapply(auc_gllvm, function(tmp) (apply(tmp, 1,mean)))))
bc_auc = t(unlist(sapply(auc_bc, function(tmp) (apply(tmp, 1,mean)))))
hmsc_auc = t(unlist(sapply(auc_hmsc, function(tmp) (apply(tmp, 1,mean)))))


pdf(file = "figures/Fig_3.pdf", width = 9, height = 3.2)
e = 5L
lineT = rep(1,5) #1:5
names(lineT) =  (as.character(unique(number)))
par(mfrow = c(1,3), mgp = c(2.1,0.6,0), mar = c(0.6, 1.5, 0.6,0.7), oma = c(2.5,2.0,2.6,0))
for(i in (as.character(unique(number)))[c(1,3,5)]){
  if( i == "0.1") {
    ylab = "AUC"
  } else {
    ylab = ""
  }
  plot(NULL, NULL, xlim = c(1,nrow(setup)), ylim = c(0.5,1), xaxt = "n", yaxt = "n", xlab = "Number of Sites", ylab = ylab, main = "", xaxs = "i", xpd = NA)
  title(paste0(as.numeric(i)*100, "% species"), line = 2, xpd = NA)
  axis(2, las = 2)
  axis(1, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites))
  axis(3, at = seq(1, nrow(setup), by = 15)+7, labels = unique(setup$sites)* as.numeric(i))
  
  for(k in seq(1, nrow(setup), by = 15)[-1]){
    abline(v = k, col = "grey")
  }
  cat(i, "\n")
  X = (1:nrow(setup))[as.character(number) == i & setup$env == e]
  mean_conf(X, cpu_auc[as.character(number) == i & setup$env == e, ], col = "black")
  mean_conf(X, gpu_auc[as.character(number) == i & setup$env == e, ], col = "red")
  mean_conf(X, rbind(gllvm_auc,matrix(NA, nrow(setup) - nrow(gllvm_auc), 10L))[as.character(number) == i & setup$env == e, ], col = "blue")
  mean_conf(X, rbind(bc_auc,matrix(NA, nrow(setup) - nrow(bc_auc), 10L))[as.character(number) == i & setup$env == e, ], col = "green")
  mean_conf(X, rbind(hmsc_auc,matrix(NA, nrow(setup) - nrow(hmsc_auc), 10L))[as.character(number) == i & setup$env == e, ], col = "violet")
  legend("bottomright", legend = c("gpu_dmvp", "cpu_dmvp", "gllvm", "bayesComm", "Hmsc"), col = c("red", "black", "blue", "green", "violet"), bty="n", lty = 1)
}
dev.off()


#### runtime case study ####
runtime = readRDS("results/case_study_runtime.RDS")
runtime_result = data.frame( data_set = names(runtime), cpu = rep(0, length(runtime)), gpu = rep(0, length(runtime)))
for(i in 1:length(runtime)){
  runtime_result[i,2] = runtime[[i]]$cpu$time/3600
  runtime_result[i,3] = runtime[[i]]$gpu$time/3600
}


#### covariance behaviour ####
gpu_beh_adam = readRDS(file = "results/gpu_behaviour_sites_adamax.RDS")
gpu_beh_lbfgs = readRDS(file = "results/gpu_behaviour_sites_lbfgs.RDS")
gllvm_beh = readRDS(file = "results/gllvm_behaviour_sites.RDS")
bc_beh = readRDS(file = "results/bc_behaviour_sites.RDS")
hmsc_beh = readRDS(file = "results/hmsc_behaviour_sites.RDS")


xx = 1:25
plot(apply(gpu_beh$result_corr_acc, 1, mean))
summary(lm(apply(bc_beh$result_corr_acc, 1, function(k) mean(k, na.rm = T))~xx))
summary(lm(apply(gpu_beh$result_corr_acc, 1, mean)~xx))

sites = seq(50,by = 20, length.out = 15)
addA = function(col, alpha = 0.25) apply(sapply(col, col2rgb)/255, 2, function(x) rgb(x[1], x[2], x[3], alpha=alpha)) 
mean_conf = function(mat, col = "red", alpha = 0.1, spar = 0.4) {
  sites2 = sites[complete.cases(mat)]
  mat = mat[complete.cases(mat),]
  
  m = apply(mat, 1, mean)
  sd = apply(mat, 1, sd)
  upper = smooth.spline(y = m + sd, x = sites2, spar = spar)$y
  lower = smooth.spline(y = m - sd, x = sites2, spar = spar)$y
  polygon(c(sites2, rev(sites2)), c(upper, rev(lower)), border = NA, col = addA(col, alpha))
  lines(smooth.spline(y = m, x = sites2, spar = spar), col = col, lwd = 2.0)
}


par(mfrow = c(1,2), mar = c(2.3, 3, 1, 1), mgp = c(2.7, 1, 0))

plot(NULL, NULL, xlim = c(min(sites), max(sites)), ylim = c(0.5, 1.0), ylab = "accuracy", xlab = "Sites", yaxt = "n", xpd = NA, main = "", xaxt = "n")
axis(1, at = sites[seq(1, length(sites), by = 2)], labels = sites[seq(1, length(sites), by = 2)])
text(x = 30, y = 1.07, pos = 2, labels = "A", xpd = NA, cex = 1.2, font = 2)
for(i in seq(1, length(sites), by = 2)) abline(v = sites[i], col = addA("grey", 0.3))
for(i in seq(0.5, 1.0, 0.1)) abline(h = i, col = addA("grey", 0.3))
title("covariance accuracy",line = 1, xpd = NA)
axis(2, las = 2)

mean_conf(gpu_beh_adam$result_corr_acc)
#mean_conf(gpu_beh_lbfgs$result_corr_acc, "black")
mean_conf(bc_beh$result_corr_acc, "green")
mean_conf(gllvm_beh$result_corr_acc, "blue")
mean_conf(hmsc_beh$result_corr_acc, "violet")
#legend("bottomright", legend = c("G-DMVP", "BC", "GLLVM", "HMSC"), lty = 1L, col = c("red", "green", "blue", "violet"), lwd = 2, bty = "n")


plot(NULL, NULL, xlim = c(min(sites), max(sites)), ylim = c(0.5, 1.0), ylab = "accuracy", xlab = "Sites", yaxt = "n", xpd = NA, main = "", xaxt = "n")
axis(1, at = sites[seq(1, length(sites), by = 2)], labels = sites[seq(1, length(sites), by = 2)])
text(x = 30, y = 1.07, pos = 2, labels = "B", xpd = NA, cex = 1.2, font = 2)
for(i in seq(1, length(sites), by = 2)) abline(v = sites[i], col = addA("grey", 0.3))
for(i in seq(0.5, 1.0, 0.1)) abline(h = i, col = addA("grey", 0.3))
title("env accuracy",line = 1, xpd = NA)
axis(2, las = 2)
mean_conf(gpu_beh_adam$result_env)
#mean_conf(gpu_beh_lbfgs$result_env, "black")
mean_conf(bc_beh$result_env, "green")
mean_conf(gllvm_beh$result_env, "blue")
mean_conf(hmsc_beh$result_env, "violet")

legend("bottomright", legend = c("G-DMVP", "BC", "GLLVM", "HMSC"), lty = 1L, col = c("red", "green", "blue", "violet"), lwd = 2, bty = "n")






#### large scale results ####

large_scale = readRDS("results/large_scale.RDS")
setup = large_scale$setup


lwd = 2.3
pdf(file = "figures/Fig_2.pdf", width = 9, height = 6)
par(mfrow = c(3,3), mgp = c(3,0.6,0), mar = c(0.5, 2.4, 0.5, 1), oma = c(4,2,2,1))
xx = unique(setup$sites)
for(i in unique(setup$species)) {
    lineT = rep(1,5) #1:5
    if(i == 300) {
      ylab = "time in minutes"
    } else {
      ylab = ""
    }
    plot(NULL, NULL, xlim = c(min(xx),max(xx)), ylim = c(2.5,max(log(large_scale$result_time[,1]))*1.1), xaxt = "n", xlab = "",main = "",yaxt = "n",  ylab = ylab, xpd = NA)
    title(main = paste0(i, " species"), line = 1.1, xpd = NA)
    if(i == 300) text(x = 1700, y = 7.2, pos = 2, labels = "A", xpd = NA, cex = 1.5, font = 2)
    for(k in seq(1, length(xx), by = 2)) abline(v = xx[k], col = addA("grey", 0.3))
    tt = seq(2.5, 6.5, length.out = 6)
    tt = exp(tt)
    axis(2, at = round(log(tt),2),labels = c(round(tt/60,1)), las = 2)
    for(k in tt) abline(h = log(k), col = addA("grey", 0.3))
    lines(smooth.spline(x = xx, log(result$result_time[setup$species == i]),spar = 0.5), lwd= lwd, lty = 1, col = "red")
    
}

for(i in unique(setup$species)) {
  lineT = rep(1,5) #1:5
  if(i == 300) {
    ylab = "Cov accuracy"
  } else {
    ylab = ""
  }
  plot(NULL, NULL, xlim = c(min(xx),max(xx)), ylim = c(0.5, 1.0), xaxt = "n",yaxt = "n",  ylab = ylab, xlab = "",  xpd = NA)
  if(i == 300) text(x = 1700, y = 1.0*1.028, pos = 2, labels = "B", xpd = NA, cex = 1.5, font = 2)
  for(k in seq(1, length(xx), by = 2)) abline(v = xx[k], col = addA("grey", 0.3))
  axis(2, las = 2)
  for(k in seq(0.5, 1.0, 0.1)) abline(h =k, col = addA("grey", 0.3))
  # axis(1, at = xx[seq(1, 14,by = 3)], labels = xx[seq(1, 14,by = 3)])
  lines(smooth.spline(x = xx, (result$result_corr_acc[setup$species == i,1]),spar = 0.5),  lty = 1, col = "red", lwd = lwd)
  
}

for(i in unique(setup$species)) {
  lineT = rep(1,5) #1:5
  if(i == 300) {
    ylab = "Env accuracy"
  } else {
    ylab = ""
  }
  plot(NULL, NULL, xlim = c(min(xx),max(xx)), ylim = c(0.5, 1.0), xaxt = "n",yaxt = "n",  ylab = ylab,xlab = "Number of sites x100", xpd = NA, lwd = lwd)
  if(i == 300) text(x = 1700, y = 1.0*1.028, pos = 2, labels = "C", xpd = NA, cex = 1.5, font = 2)
  for(k in seq(1, length(xx), by = 2)) abline(v = xx[k], col = addA("grey", 0.3))
  axis(2, las = 2)
  for(k in seq(0.5, 1.0, 0.1)) abline(h =k, col = addA("grey", 0.3))
  #axis(1)
  axis(2, las = 2)
  axis(1, at = xx[seq(1, length(xx), by = 2)], labels = xx[seq(1, length(xx), by = 2)]/100)
  lines(smooth.spline(x = xx, (result$result_env[setup$species == i,1]),spar = 0.5), lwd= lwd, lty = 1, col = "red")
  
}
dev.off()
