
library(data.table)
library(igraph)
library(RandomWalkRestartMH)
library(ggplot2)
###Randowm walk Re-start
# m6Asites_sim_PCA <- read.csv( "F:\\m6Adisease_data\\m6Asites_sim_PCA_infor.csv")
f1 <- "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6Asite_cosine_sim.csv"
m6A_cosine <- read.csv(f1)
colnames(m6A_cosine) <- c("source_m6A","target_m6A","sim_value")
##计算z-score和显著性p值
# m6A_cosine$Z_score <- (m6A_cosine$sim_value - mean(m6A_cosine$sim_value))/sd(m6A_cosine$sim_value)
# m6A_cosine$p_value <- 2*pnorm(m6A_cosine$Z_score,lower.tail = F)
# ##筛选显著的结果
# m6A_cosine <- m6A_cosine[m6A_cosine$p_value<0.05,]
# ##校正p值
# m6A_cosine$p_adj <- p.adjust(m6A_cosine$p_value,method = "fdr")
##不叫正剩余45298，矫正后剩余17376
##将所有m6A作为种子
seeds <- union(unique(m6A_cosine$source_m6A),unique(m6A_cosine$target_m6A))
##将种子进行排序
seeds <- sort(seeds)


##结果网络图和性质分析
specific <- m6A_cosine
##生成网络
m6Asites_Network <- graph.data.frame(specific,directed=FALSE)
##边权重
E(m6Asites_Network)$weight <- specific$sim_value

##网络简化，删除多边和环
m6Asites_Network <- igraph::simplify(m6Asites_Network, remove.multiple = TRUE, remove.loops = TRUE)

#RWR
m6A_MultiplexObject <- create.multiplex(list(m6Asites_Network),Layers_Name=c("PPI"))
AdjMatrix_m6A <- compute.adjacency.matrix(m6A_MultiplexObject)
AdjMatrixNorm_m6A <- normalize.multiplex.adjacency(AdjMatrix_m6A)
##模块挖掘
module <- list()
for (i in 1:length(seeds)) {
  module[[i]] <- Random.Walk.Restart.Multiplex(AdjMatrixNorm_m6A,m6A_MultiplexObject,seeds[i])[[1]]
  #print(i)
}
names(module) <- seeds

##添加种子
for (i in 1:length(module)) {
  module[[i]] <- cbind(seed=rep(seeds[i],nrow(module[[i]])),module[[i]])
}
##备份
bf <- module
##筛选网络中可以与种子对应上的节点
for (i in 1:length(module)) {
  module[[i]] <- module[[i]][module[[i]]$NodeNames%in%m6A_cosine$target_m6A[m6A_cosine$source_m6A%in%names(module)[i]],]
}

##去除空列表
n <- c()
for (i in 1:length(module)) {
  n <- c(n,nrow(module[[i]]))
}
sum(n)
which(n==0)
module[which(n==0)] <- NULL
##剩余1325个模块

##拼接结果
result <- data.frame()
for (i in 1:length(module)) {
  result <- rbind(result,module[[i]])
  print(i)
}
##查看m6A数目，2869
length(union(result$seed,result$NodeNames))

rwr <- result
colnames(rwr) <- c(colnames(result)[-3],"RWR")
##筛选大于中位数, 0.002608145
rwr <- rwr[rwr$RWR>median(rwr$RWR),]
# write.csv(rwr,file = "F:\\m6Adisease_data\\new_dis_m6A_data\\m6Asites_rwr.csv",row.names = F)
####
source_num <- as.numeric(str_remove(rwr$seed,"m6A"))
target_num <-  as.numeric(str_remove(rwr$NodeNames,"m6A"))
rwr_value <- as.numeric(as.character(rwr$RWR))
new_rwr <- data.frame(source=source_num,target=target_num,RWR=rwr_value  )

 write.csv(new_rwr,file = "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6Asites_RWR.csv",row.names = F)
####

# ##添加其他信息
# num <- c()
# for (i in 1:nrow(rwr)) {
#   num <- c(num,which(m6Asites_sim_PCA$source_m6A%in%rwr$seed[i]&m6Asites_sim_PCA$target_m6A%in%rwr$NodeNames[i]))
# }
# 
# rwr$cosine_sim <- m6Asites_sim_PCA$cosine_sim[num]
# rwr$m6AmicRNA_jaccsimi <- m6Asites_sim_PCA$m6AmicRNA_jaccsimi[num]
# rwr$m6AgeneBP_jaccsimi <- m6Asites_sim_PCA$m6AgeneBP_jaccsimi[num]
# rwr$m6AREG_jaccsimi <- m6Asites_sim_PCA$m6AREG_jaccsimi[num]
# 
# rwr$PCA <- m6Asites_sim_PCA$PCA[num]
# rwr$z_score <- m6Asites_sim_PCA$Z_score[num]
# rwr$p_value <- m6Asites_sim_PCA$p_value[num]
# rwr$p_adj <- m6Asites_sim_PCA$p_adj[num]
# ####