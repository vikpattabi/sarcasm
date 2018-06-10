
data1 = read.csv("evaluation-1-trials.tsv", sep="\t")
data2 = read.csv("evaluation-2-trials.tsv", sep="\t")
data2$workerid = data2$workerid + 10


data = rbind(data1, data2)

library(tidyr)
library(dplyr)

data %>% group_by(type) %>% summarise(pertinent = mean(pertinent), sarcastic = mean(sarcastic))


