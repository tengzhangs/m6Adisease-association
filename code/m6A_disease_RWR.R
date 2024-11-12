library(data.table)
library(igraph)
library(RandomWalkRestartMH)
library(ggplot2)
fa <- "F:\\m6Adisease_data\\new_dis_m6A_data\\m6Adis_similariy.csv"
m6Adisease_sim <- read.csv(fa)
colnames(m6Adisease_sim) <- paste0("dis",seq(1,ncol(m6Adisease_sim)))
dis_ID <- colnames(m6Adisease_sim)
rownames(m6Adisease_sim) <- dis_ID
disease_sim <- data.frame()
for (i in 1:length(dis_ID)) {
  onedis_sim <- m6Adisease_sim[rownames(m6Adisease_sim)== dis_ID[i],]
  target_sim <- onedis_sim[-which(rownames(onedis_sim)==colnames(onedis_sim))]
  target_ID <- dis_ID[-i]
  match_label <-  which(!is.na(match(target_ID,colnames(target_sim))))
  onetarget_sim <- vector(length = length(target_ID))
  onetarget_sim[match_label] <- as.numeric(as.character(target_sim))
  
  source_label <- rep(dis_ID[i],length(target_ID))
  one_dis_infor <- data.frame(source_dis=source_label,target_dis=target_ID,dis_sim=onetarget_sim)
  disease_sim <- rbind(disease_sim,one_dis_infor)
}
write.csv(disease_sim,file = "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6Adis_simi.csv",row.names = F)
##random walk with restart on disease-disease similarity
##结果网络图和性质分析
fa <- "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6Adis_simi.csv"
disease_sim <-(fread(fa))

# source_num <- round(as.numeric(as.character(str_remove(disease_sim$source_dis,"DOID."))))
# target_num <-  round(as.numeric(as.character(str_remove(disease_sim$target_dis,"DOID."))))
# sim_value <- as.numeric(as.character(disease_sim$dis_sim))
# new_dissim <- data.frame(source=source_num,target=target_num,Sim=sim_value  )
# 
# write.csv(new_dissim,file = "F:\\m6Adisease_data\\new_dis_m6A_data\\m6Adis_Simlarity.csv",row.names = F)

seeds <- union(unique(disease_sim$source_dis),unique(disease_sim$target_dis))
##将种子进行排序
seeds <- sort(seeds)

specific <- disease_sim
##生成网络
dis_Network <- graph.data.frame(specific,directed=FALSE)
##边权重
E(dis_Network)$weight <- specific$dis_sim

##网络简化，删除多边和环
dis_Network <- igraph::simplify(dis_Network, remove.multiple = TRUE, remove.loops = TRUE)

#RWR
dis_MultiplexObject <- create.multiplex(list(dis_Network),Layers_Name=c("PPI"))
AdjMatrix_dis <- compute.adjacency.matrix(dis_MultiplexObject)
AdjMatrixNorm_dis <- normalize.multiplex.adjacency(AdjMatrix_dis)
##模块挖掘
module <- list()
for (i in 1:length(seeds)) {
  module[[i]] <- Random.Walk.Restart.Multiplex(AdjMatrixNorm_dis,dis_MultiplexObject,seeds[i])[[1]]
 # print(i)
}
names(module) <- seeds



##添加种子
for (i in 1:length(module)) {
  module[[i]] <- cbind(seed=rep(seeds[i],nrow(module[[i]])),module[[i]])
}

##筛选网络中可以与种子对应上的节点
for (i in 1:length(module)) {
  module[[i]] <- module[[i]][module[[i]]$NodeNames%in%disease_sim$target_dis[disease_sim$source_dis%in%names(module)[i]],]
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
 # print(i)
}
##查看m6A数目，2869
length(union(result$seed,result$NodeNames))

rwr <- result
colnames(rwr) <- c(colnames(result)[-3],"RWR")
##筛选大于中位数, 0.002608145
rwr <- rwr[rwr$RWR>median(rwr$RWR),]
####
source_num <- as.numeric(str_remove(rwr$seed,"dis"))
target_num <-  as.numeric(str_remove(rwr$NodeNames,"dis"))
rwr_value <- as.numeric(as.character(rwr$RWR))
new_rwr <- data.frame(source=source_num,target=target_num,RWR=rwr_value  )
write.csv(new_rwr,file = "F:\\m6Adisease_data\\new_dis_m6A_data\\only_use_simvalue\\m6Adis_rwr.csv",row.names = F)
  


#  ###
# fb <- "F:\\m6Adisease_data\\m6Adis_rwr.csv"
# dis_rwr <- read.csv(fb)
# source_num <- round(as.numeric(as.character(str_remove(dis_rwr$seed,"DOID."))))
# target_num <-  round(as.numeric(as.character(str_remove(dis_rwr$NodeNames,"DOID."))))
# rwr_value <- as.numeric(as.character(dis_rwr$RWR))
# new_rwr <- data.frame(source=source_num,target=target_num,RWR=rwr_value  )
# 
# write.csv(new_rwr,file = "F:\\m6Adisease_data\\m6Adis_RWR.csv",row.names = F)
