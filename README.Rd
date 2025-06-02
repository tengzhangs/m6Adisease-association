# Introduction
We propose a novel computational framework, m6A-Disease Prediction using Graph Convolutional Networks and Positive Unlabeled Learning with self-Adaptive Sampling (m6ADP-GCNPUAS), to accurately predict m6A-disease associations. m6ADP-GCNPUAS effectively captures the embedded features of 
m6A sites or diseases using GCN model. Given the very limited m6A-disease associations, m6ADP-GCNPUAS adopts the PUAS framework to augment the potential positive samples, which improves the predictive power for m6A-disease associations.
# Usage Example
## Step by Step 
### Calculate the similarity for m6A-m6A and disease-disease

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
