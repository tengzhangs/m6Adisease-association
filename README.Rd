# Introduction
We propose a novel computational framework, m6A-Disease Prediction using Graph Convolutional Networks and Positive Unlabeled Learning with self-Adaptive Sampling (m6ADP-GCNPUAS), to accurately predict m6A-disease associations. m6ADP-GCNPUAS effectively captures the embedded features of 
m6A sites or diseases using GCN model. Given the very limited m6A-disease associations, m6ADP-GCNPUAS adopts the PUAS framework to augment the potential positive samples, which improves the predictive power for m6A-disease associations.
# Usage Example
## Step by Step 
### Calculate the similarity for m6A-m6A and disease-disease
#### Calculate the similarity for m6A sites
'''r
#Get the sequences of m6A sites
fa <- "./m6Adis_asso.csv"
m6Adis_sites <- read.csv(fa)
library(GenomicRanges)

sites_GR <- GRanges(seqnames = as.character(m6Adis_sites$seqnames),
                  IRanges(start = as.numeric(as.character(m6Adis_sites$m6A_site)),
                          end = as.numeric(as.character(m6Adis_sites$m6A_site))),
                  strand = as.character(m6Adis_sites$strand))

sites_range <- resize(sites_GR, 501,fix="center")

library(BSgenome.Hsapiens.UCSC.hg19)
genome <- BSgenome.Hsapiens.UCSC.hg19
get_seq<- getSeq(genome,sites_range)
seqs <- as.character(get_seq)
write.table(seqs,file = "./m6Adis_seq.txt",row.names=F,col.names = F,quote=F)
'''
'''python
import logging
from gensim.models import  Word2Vec
from gensim.models.word2vec import LineSentence
import numpy as np
import pandas as pd

seq_path="./m6Adis_seq.txt"
with open(seq_path,"r") as fr:
        lines = fr.readlines()
        
fr.close()
words=np.zeros(shape=(len(lines),499)).astype(np.str_)

i=0
k=3
for line in lines:
    j=0
    if line.startswith(">hsa") or len(line)<=1:
        continue
    else:
        line=line[:-1]
        seq_len=len(line)
        for index in range(0,seq_len,1):
            if index+k >= seq_len+1:
                break
            a=line[index:index+k]
            words[i,j]=a
            j=j+1
    i=i+1

pd.DataFrame(words).to_csv("./pos_m6a_dis_word_10.csv",index=False)

word_path="./pos_m6a_dis_word_10.txt"

with open(word_path,"w") as fw:
    for line in lines:
        if line.startswith(">hsa") or len(line)<=1:
                continue
        else:
            line=line[:-1]
            seq_len=len(line)
            for index in range(0,seq_len,1):
                if index+k>=seq_len+1:
                    break
                fw.write("".join(line[index:index+k]))
                fw.write(" ")
                fw.write("\n")
    fw.close()

'''
word2vec train 
'''
logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s",level=logging.INFO)


sentences=LineSentence("./pos_m6a_dis_word_10.txt")

    
vector_dim=100
model = Word2Vec(sentences, window=5, min_count=1, epochs=30, vector_size=vector_dim)
model.save("./nm6a_vec")
    
dataset = pd.read_csv("./pos_m6a_dis_word_10.csv")
word=model.wv.index_to_key
vector=model.wv.vectors
    # feature=np.zeros((499,32))
#    features=[]
#    for idx, data in dataset.iterrows():
#        wv_feature = np.zeros((99, 100))
#        i=0
#        for ix,char in data.items():
#            wv_index=word.index(char)
#            wv_feature[i,:]=vector[wv_index]
#            i=i+1
#        features.append(wv_feature)
'''
根据每个位点的分词获得每个分词的embeding，然后将每个位点分词的embeding平均就和
'''
feature = np.zeros((len(lines),100,499))
    
for i in range(0,len(lines)):
    m=0        
    for j in range(0,499):
        char = dataset.iloc[i,j]
        index = word.index(char)
        feature[i,:,m] = vector[index,0:]
        m=m+1
        
features = np.zeros((len(lines),100))
features = np.sum(feature,axis=2)/499

pd.DataFrame(features).to_csv("./m6A_word2vec_feature.csv",index=False)
# calculate the simialrity of m6A sites by cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
m1=features
m1_similarity = cosine_similarity(m1)
pd.DataFrame(m1_similarity).to_csv("./m6A_cosine_similarity.csv",index=False)
'''
#### Calculate the similarity for diseases-disease
'''r
library(DOSE)
library(data.table)
f1 <- "./disease_IDinfor_new.xlsx"
DOID_infor <- readxl::read_xlsx(f1)
f2 <- "./m6Adis_asso.csv"
m6Adis_sites <- read.csv(f2)
match_m6Adis_name <- colnames(m6Adis_sites)[-c(1:5)]
newdis_infor <- data.frame()
for (i in 1:length(match_m6Adis_name)) {
  
  onedis_infor <- DOID_infor[DOID_infor$disease_name==match_m6Adis_name[i],]
  newdis_infor <- rbind(newdis_infor,onedis_infor)
}
a1 <- as.character(newdis_infor$ID)
a2 <- as.character(newdis_infor$ID)
s <- doSim(a1, a2, measure="Wang")
rownames(s) <- a1
s <- as.data.frame(s)
write.csv(s,file = "./dis_similariy.csv",row.names = F)
'''
### Construct m6A-m6A interaction network and disease-disease interaction network by RWR

'''r
m6A_sim <- read.csv(f1="./m6A_cosine_similarity.csv")
dis_sim <- read.csv(f2="./dis_similariy.csv"
m6A_m6A_net <- m6Asites_RWR(m6A_sim)
dis_dis_net <- dis_RWR(dis_sim)
'''
### Extract the embedding feartures for m6A sites and diseases by GCN model

'''python
#Runing the following python code
m6A_embeddingGCN.py
dis_embeddingGCN.py
'''
