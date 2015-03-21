# Code used for the Intro to Data Science GDI class for the Titanic dataset
# Source for much of this code is https://github.com/wehrley/wehrley.github.io/blob/master/SOUPTONUTS.md


titanic<- read.csv("../data/titanic_train.csv", na.strings=c("NA", ""))
titanic$Survived <- as.factor(titanic$Survived)
titanic$Pclass <- as.factor(titanic$Pclass)

# Fix missing values

## extract honorific (i.e. title) from the Name feature
titanic$Title <- as.factor(gsub(".*\\, ([A-Za-z ]+)\\..*", "\\1", data$Name))

unique(titanic$Title)
#[1] "Mr"     "Mrs"     "Miss"    "Master"        "Don"        "Rev"
#[7] "Dr"     "Mme"      "Ms"     "Major"         "Lady"       "Sir"
#[13] "Mlle"   "Col"     "Capt"    "the Countess"  "Jonkheer"

options(digits=2)
require(Hmisc)
bystats(titanic$Age, titanic$Title, 
        fun=function(x)c(Mean=mean(x),Median=median(x)))
# N Missing Mean Median
# Capt           1       0 70.0   70.0
# Col            2       0 58.0   58.0
# Don            1       0 40.0   40.0
# Dr             6       1 42.0   46.5
# Jonkheer       1       0 38.0   38.0
# Lady           1       0 48.0   48.0
# Major          2       0 48.5   48.5
# Master        36       4  4.6    3.5
# Miss         146      36 21.8   21.0
# Mlle           2       0 24.0   24.0
# Mme            1       0 24.0   24.0
# Mr           398     119 32.4   30.0
# Mrs          108      17 35.9   35.0
# Ms             1       0 28.0   28.0
# Rev            6       0 43.2   46.5
# Sir            1       0 49.0   49.0
# the Countess   1       0 33.0   33.0
# ALL          714     177 29.7   28.0

imputeMedian <- function(impute.var, filter.var, var.levels) {
  for (v in var.levels) {
    impute.var[ which( filter.var == v)] <- impute(impute.var[ 
      which( filter.var == v)])
  }
  return (impute.var)
}
## list of titles with missing Age value(s) requiring imputation
titanic$Age <- imputeMedian(titanic$Age, titanic$Title, c("Dr", "Master", "Miss", "Mr", "Mrs"))

# Replace 2 missing Embarked values with median
summary(titanic$Embarked)
titanic$Embarked[which(is.na(titanic$Embarked))] <- 'S'

# Fix 0 fare classes
titanic$Fare[ which( titanic$Fare == 0 )] <- NA
titanic$Fare <- imputeMedian(titanic$Fare, titanic$Pclass, 
                              as.numeric(levels(titanic$Pclass)))


require(plyr)     # for the revalue function 
require(stringr)  # for the str_sub function

## test a character as an EVEN single digit
isEven <- function(x) x %in% c("0","2","4","6","8") 
## test a character as an ODD single digit
isOdd <- function(x) x %in% c("1","3","5","7","9") 

## function to add features to training or test data frames
featureEngrg <- function(data) {
  ## Using Fate ILO Survived because term is shorter and just sounds good
  data$Fate <- data$Survived
  ## Revaluing Fate factor to ease assessment of confusion matrices later
  data$Fate <- revalue(data$Fate, c("1" = "Survived", "0" = "Perished"))
  ## Boat.dibs attempts to capture the "women and children first"
  ## policy in one feature.  Assuming all females plus males under 15
  ## got "dibs' on access to a lifeboat
  data$Boat.dibs <- "No"
  data$Boat.dibs[which(data$Sex == "female" | data$Age < 15)] <- "Yes"
  data$Boat.dibs <- as.factor(data$Boat.dibs)
  ## Family consolidates siblings and spouses (SibSp) plus
  ## parents and children (Parch) into one feature
  data$Family <- data$SibSp + data$Parch
  ## Fare.pp attempts to adjust group purchases by size of family
  data$Fare.pp <- data$Fare/(data$Family + 1)
  ## Giving the traveling class feature a new look
  data$Class <- data$Pclass
  data$Class <- revalue(data$Class, 
                        c("1"="First", "2"="Second", "3"="Third"))
  ## First character in Cabin number represents the Deck 
  data$Deck <- substring(data$Cabin, 1, 1)
  data$Deck[ which( is.na(data$Deck ))] <- "UNK"
  data$Deck <- as.factor(data$Deck)
  ## Odd-numbered cabins were reportedly on the port side of the ship
  ## Even-numbered cabins assigned Side="starboard"
  data$cabin.last.digit <- str_sub(data$Cabin, -1)
  data$Side <- "UNK"
  data$Side[which(isEven(data$cabin.last.digit))] <- "port"
  data$Side[which(isOdd(data$cabin.last.digit))] <- "starboard"
  data$Side <- as.factor(data$Side)
  data$cabin.last.digit <- NULL
  return (data)
}

## add remaining features to training data frame
titanic <- featureEngrg(titanic)

features_keep <- c("Fate", "Sex", "Boat.dibs", "Age", "Title", 
                 "Class", "Deck", "Side", "Fare", "Fare.pp", 
                 "Embarked", "Family")
trainingdata <- titanic[features_keep]

