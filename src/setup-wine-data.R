setwd("~/Development/intro-data-science/src")


redwine <- read.csv("../data/winequality-red.csv", sep=";")
redwine$color <- "red"

whitewine <- read.csv("../data/winequality-white.csv", sep=";")
whitewine$color <- "white"

wine <- rbind(redwine, whitewine)

write.csv(wine, file="../data/wine.csv")
