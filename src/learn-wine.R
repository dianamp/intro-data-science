setwd("~/Development/intro-data-science/src")

# Prep the data set
wine <- read.csv("../data/wine.csv")
wine$is_red <- factor(ifelse(wine$color=='red', 1, 0))
wine$high_quality <- factor(ifelse(wine$quality > 6, 1, 0))
wine$quality <- factor(wine$quality) 

summary(wine)

# create training/test set
require('caret')
set.seed(991)
trainI <- createDataPartition(y = wine$high_quality, p = .75, list = FALSE) 
training <- wine[ trainI,]
test <- wine[-trainI,]

training$quality <- NULL
training$color <- NULL
training$X <- NULL

ctrl <- trainControl(method = "cv",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)
dtree1 <- train(high_quality ~ ., data = training,
                method = "rpart",
                metric = "ROC",
                na.action = na.pass,
                trControl = ctrl)
pred_prob <- predict(dtree1, newdata=test, na.action = na.pass)
truth_class <- test$high_quality
pred_class <- predict(dtree1, newdata=test, type="raw", na.action = na.pass)
confusionMatrix(pred_class, truth_class)

fancyRpartPlot(dtree1$finalModel)

# Model 2 - decision tree on upsampled data
set.seed(12)
upSampledTraining <- upSample(training, training$high_quality)
upSampledTraining$Class <- NULL
ctrl <- trainControl(method = "cv",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)
dtree <- train(high_quality ~ ., data = upSampledTraining,
               method = "rpart",
               metric = "ROC",
               na.action = na.pass,
               trControl = ctrl)

truth_class <- test$high_quality
pred_class <- predict(dtree, newdata=test, type="raw")
confusionMatrix(pred_class, truth_class)
fancyRpartPlot(dtree$finalModel)

# random forest
minorityClassSize <- sum(training$high_quality == 1)
minorityClassSize

ctrl <- trainControl(method = "cv",
                     classProbs = TRUE,
                     summaryFunction = twoClassSummary)
set.seed(2)
rfdownSpec <- train(high_quality ~ ., data = training,
                    method = "rf",
                    ntree = 1000,
                    tuneLength = 5,
                    metric = "ROC",
                    trControl = ctrl,
                    na.action=na.omit,
                    ## Tell randomForest to sample by strata. Here,
                    ## that means within each class
                    strata = training$high_quality,
                    ## Now specify that the number of samples selected
                    ## within each class should be the same
                    sampsize = rep(minorityClassSize, 2))

pred_class <- predict(rfdownSpec, newdata=testData, type="raw")
confusionMatrix(pred_class, truth_class)

rfUnbalanced <- train(status ~ ., data = training,
                      method = "rf",
                      ntree = 1500,
                      tuneLength = 5,
                      metric = "ROC",
                      trControl = ctrl)
