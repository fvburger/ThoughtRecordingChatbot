require(plyr)
require(dplyr)
require(reshape2)
require(irr)
require(tidyr)
require(ggplot2)
require(magrittr)

setwd("~/surfdrive/Documents/Projects/ThoughtRecordChatbot/Chatbot/rasa/supp_materials")

#print session info
sessionInfo()

#set seed for everything to follow
char2seed("Burger",set=TRUE)

#read in the core thought record data
df <- read.csv("CoreData.csv",na.strings = "", 
               header=TRUE, sep=";",fill=TRUE)

schemas <- c("Attach","Comp","Global","Health","Control",
             "MetaCog","Others","Hopeless","OthViews")

#select only relevant columns and rows
df <- df[which(df$UttEnum!="NA"),
         c("Reply",schemas,"Exclude","UttEnum","Scenario",
           "Depth","Participant.ID")] %>% na.omit(.)

#rename Reply column to Utterance
names(df)[names(df) == "Reply"] <- "Utterance"

df[,2:11] <- lapply(df[,2:11], function(x) as.numeric(as.character(x)))

# we remove the exclude sentences from the set
df <- df[which(df$Exclude==0),]
# then we can also remove the exclude column
df$Exclude <- NULL

# we also want a column that says whether the thought 
# record was scenario-based (closed) or a personal one (open)
df$TRtype <- ifelse(df$Scenario=="PTR","open","closed")
df$TRtype <- as.factor(df$TRtype)

# to get number of thought record that were of a certain maximum depth
# we first convert depth column to numeric
df$Depth <- as.numeric(as.character(df$Depth))
# we then get the maximum depth per thought record
df.maxdepth <- df %>% dplyr::group_by(Participant.ID,Scenario) %>% summarise(levels=max(Depth))
# finally, we count the number of thought records per max depth level
df.countdepths <- df.maxdepth %>% group_by(levels) %>% summarise(trsOFdepth=n())
# export to csv file
write.table(df.countdepths, "trs_per_depth.txt", sep="\t",row.names=FALSE)