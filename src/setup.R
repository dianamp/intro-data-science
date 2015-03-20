require(ggplot2)
require(plyr)
require(lubridate)
require(zoo)

library("rpart.plot")      	# Enhanced tree plots
library(RColorBrewer)				# Color selection for fancy tree plot
library(party)					# Alternative decision tree algorithm
library(partykit)				# Convert rpart object to BinaryTree
library(caret)		
library(rattle)