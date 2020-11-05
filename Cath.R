# ----------------------------------------------------------------------------------
# FIT3164 Team 8
# Code for the predictive model
# Author: Xinyu Ma, Zhaofa Guo
# Date: 22/10/2020

# System: Windows 10
# RStudio version: 1.3.1073
# How to run: select all lines of code and click 'Run'
# Approximate runtime: 3-6 minutes
# ----------------------------------------------------------------------------------


rm(list=ls())  # clean environment
# ----------------------------------------------------------------------------------
# Library used (if library is not installed, use install.packages(...))
# ----------------------------------------------------------------------------------
library(readxl)
library(dplyr)
library(data.table)
library(mltools)
library(Boruta)
library(caret)
library(mlr)
library(rpart)
library(randomForest)
library(ROCR)
library(C50)
library(neuralnet)
library(cvAUC)
library(pROC)
library(MLeval)
library(caTools)
library(caretEnsemble)
library(ROCR)
library(caTools)


# ----------------------------------------------------------------------------------
# Loading dataset and pre-processing
# ----------------------------------------------------------------------------------
zdata <- read.csv("Z-Alizadeh sani dataset.csv", stringsAsFactors = TRUE)
sum(is.na(zdata))  # looking for missing values
attach(zdata)
str(zdata)
z_feature <- select(zdata,-Cath)        # select all features excluding the label
cath <- zdata[, ncol(zdata)]              
zdmy <- one_hot(as.data.table(z_feature))  # dummy variables (i.e., Yes/No --> 1/0)
z.data = cbind(zdmy,cath)                 


# ----------------------------------------------------------------------------------
# Feature selection (Boruta algorithm)
# ----------------------------------------------------------------------------------
set.seed(9999)
colnames(z.data)[1] <- 'Age'

boruta <- Boruta(cath~., z.data, doTrace=2, maxRuns=1000)  # set maxRuns=1000 to identify all features
plot(boruta,las=2,cex.axis=1)  # plot of importance
legend("topleft",  title="Importance",inset=.02,
       c("Unimportant","Important","worst/average/best","tentative"), fill=c("red","green","blue","yellow"), horiz=F, cex=0.8)
attStats(boruta)
getSelectedAttributes(boruta, withTentative = F)  # return important features

# Note: "Diastolic.Murmur_N" and "Diastolic.Murmur_Y" are essentially the same thing,
# "Diastolic.Murmur_N" is just the opposite of "Diastolic.Murmur_Y".
# We only have to include one of them as important feature, and so on.

# important features 
z.new <- z.data[,c("Age","BMI", "DM" ,  "HTN"  , "BP",
                     "Diastolic.Murmur_Y", "Typical.Chest.Pain", "Atypical_Y", "Nonanginal_Y" ,     
                     "Tinversion", "FBS",  "TG",    "ESR" ,   "EF.TTE" ,  "Region.RWMA" ,      
                     "VHD_Severe","cath")]


# ----------------------------------------------------------------------------------
# Split into training and testing set
# ----------------------------------------------------------------------------------
z.train <- createDataPartition(y = z.new$cath,
                               p = .7,
                               list = FALSE)
training <- z.new[z.train,]  # 70% training set (for training our model)
testing <- z.new[-z.train,]  # 30% testing set (for testing our model performance on unseen data)


# ----------------------------------------------------------------------------------
# K-fold Cross Validation on different classification methods (10 folds, repeated 10 times) and fine-tuning
# ----------------------------------------------------------------------------------
# random forest
z.rf <- caret::train(cath~.,training,method='rf',metric="ROC",trControl = trainControl(number = 5,
                                                                    classProbs = TRUE,
                                                                    savePredictions = TRUE,
                                                                    summaryFunction = twoClassSummary,
                                                                    verboseIter = TRUE))
# prediction and confusion matrix
rf.cath.prob <- predict(z.rf,newdata = testing,type = 'prob')  # return probability of prediction
rf.cath <- predict(z.rf,newdata=testing)   # return prediction label
rf.cm <- confusionMatrix(data = rf.cath, testing$cath)

# tree
z.tree <- caret::train(cath~.,training,method='ctree',metric="ROC",trControl = trainControl(method = 'repeatedcv',
                                                                number = 10,
                                                                repeats = 10,
                                                                classProbs = TRUE,
                                                                savePredictions = TRUE,
                                                                summaryFunction = twoClassSummary,
                                                                verboseIter = TRUE))
# prediction and confusion matrix
tree.cath.prob <- predict(z.tree,newdata = testing,type = 'prob')
tree.cath <- predict(z.tree,newdata=testing)
tree.cm <- confusionMatrix(data = tree.cath, testing$cath)

# C5.0
z.50 <- caret::train(cath~.,training,method='C5.0',metric="ROC",trControl = trainControl(method = 'repeatedcv',
                                                                                number = 10,
                                                                                repeats = 10,
                                                                                classProbs = TRUE,
                                                                                savePredictions = TRUE,
                                                                                summaryFunction = twoClassSummary,
                                                                                verboseIter = TRUE))
# prediction and confusion matrix
C50.cath.prob <- predict(z.50,newdata = testing,type = 'prob')
C50.cath <- predict(z.50,newdata=testing)
C50.cm <- confusionMatrix(data = C50.cath, testing$cath)

# naive bayes
z.nb <- caret::train(cath~.,training,method='naive_bayes',metric="ROC",trControl = trainControl(method = 'repeatedcv',
                                                                               number = 10,
                                                                               repeats = 10,
                                                                               classProbs = TRUE,
                                                                               savePredictions = TRUE,
                                                                               summaryFunction = twoClassSummary,
                                                                               verboseIter = TRUE))
# prediction and confusion matrix
nb.cath.prob <- predict(z.nb,newdata = testing,type = 'prob')
nb.cath <- predict(z.nb,newdata=testing)
nb.cm <- confusionMatrix(data = nb.cath, testing$cath)

# neural network
z.net <- caret::train(cath~.,training,method='nnet',metric="ROC",trControl = trainControl(method = 'repeatedcv',
                                                                     number = 10,
                                                                     repeats = 10,
                                                                     classProbs = TRUE,
                                                                     verboseIter = TRUE,
                                                                     savePredictions = TRUE,
                                                                     summaryFunction = twoClassSummary))
# prediction and confusion matrix
net.cath.prob <- predict(z.net,newdata = testing,type = 'prob')
net.cath <- predict(z.net,newdata=testing)
net.cm <- confusionMatrix(data = net.cath, testing$cath)

# svm
z.svm <- caret::train(cath~.,training,method='svmLinear',metric="ROC",trControl = trainControl(method = 'repeatedcv',
                                                                   number = 10,
                                                                   repeats = 10,
                                                                   classProbs = TRUE,
                                                                   savePredictions = TRUE,
                                                                   summaryFunction = twoClassSummary,
                                                                   verboseIter = TRUE))
# prediction and confusion matrix
svm.cath.prob <- predict(z.svm,newdata = testing,type = 'prob')
svm.cath <- predict(z.svm,newdata=testing)
svm.cm <- confusionMatrix(data = svm.cath, testing$cath)

# J48
z.48 <- caret::train(cath~.,training,method='J48',metric="ROC",trControl = trainControl(method = 'repeatedcv',
                                                                                    number = 10,
                                                                                    repeats = 10,
                                                                                    classProbs = TRUE,
                                                                                    savePredictions = TRUE,
                                                                                    summaryFunction = twoClassSummary,
                                                                                    verboseIter = TRUE))
# prediction and confusion matrix
J48.cath.prob <- predict(z.48,newdata = testing,type = 'prob')
J48.cath <- predict(z.48,newdata=testing)
J48.cm <- confusionMatrix(data = J48.cath, testing$cath)

# knn
z.knn <- caret::train(cath~.,training,method='kknn',metric="ROC",trControl = trainControl(method = 'repeatedcv',
                                                                            number = 10,
                                                                            repeats = 10,
                                                                            classProbs = TRUE,
                                                                            savePredictions = TRUE,
                                                                            summaryFunction = twoClassSummary,
                                                                            verboseIter = TRUE))
# prediction and confusion matrix
knn.cath.prob <- predict(z.knn,newdata = testing,type = 'prob')
knn.cath <- predict(z.knn,newdata = testing)
knn.cm <- confusionMatrix(data = knn.cath,testing$cath)


# ----------------------------------------------------------------------------------
# AUC for each model (Area-Under-Curve)
# ----------------------------------------------------------------------------------
rf.auc <- AUC((rf.cath.prob$Normal),c(testing$cath))
tree.auc <- AUC((tree.cath.prob$Normal),c(testing$cath))
C50.auc <- AUC((C50.cath.prob$Normal),c(testing$cath))
nb.auc <- AUC((nb.cath.prob$Normal),c(testing$cath))
net.auc <- AUC((net.cath.prob$Normal),c(testing$cath))
svm.auc <- AUC((svm.cath.prob$Normal),c(testing$cath))
J48.auc <- AUC((J48.cath.prob$Normal),c(testing$cath))
knn.auc <- AUC((knn.cath.prob$Normal),c(testing$cath))
AUCs <- data.frame(ModelName = c('randomForest','tree','C5.0','NaiveBayes',
                                 'neural network','SVM','J48','KNN'),
                   AUC = c(rf.auc,tree.auc,C50.auc,nb.auc,net.auc,
                           svm.auc,J48.auc,knn.auc))
AUCs


# ----------------------------------------------------------------------------------
# Accuracy for each model
# ----------------------------------------------------------------------------------
rf.ac <- (rf.cm$table[1,1] + rf.cm$table[2,2])/sum(rf.cm$table)
tree.ac <- (tree.cm$table[1,1] + tree.cm$table[2,2])/sum(tree.cm$table)
C50.ac <- (C50.cm$table[1,1] + C50.cm$table[2,2])/sum(C50.cm$table)
nb.ac <- (nb.cm$table[1,1] + nb.cm$table[2,2])/sum(nb.cm$table)
net.ac <- (net.cm$table[1,1] + net.cm$table[2,2])/sum(net.cm$table)
svm.ac <- (svm.cm$table[1,1] + svm.cm$table[2,2])/sum(svm.cm$table)
J48.ac <- (J48.cm$table[1,1] + J48.cm$table[2,2])/sum(J48.cm$table)
knn.ac <- (knn.cm$table[1,1] + knn.cm$table[2,2])/sum(knn.cm$table)
acs <- data.frame(ModelName = c('randomForest','tree','C5.0','NaiveBayes',
                                'neural network','SVM','J48','KNN'),
                  Accuracy = c(rf.ac,tree.ac,C50.ac,nb.ac,net.ac,
                                svm.ac,J48.ac,knn.ac))
acs
ac_AUC <- merge(acs,AUCs)
ac_AUC[order(ac_AUC$AUC, decreasing = TRUE),]

# Best single model: random forest (it has the highest AUC and accuracy)
# Accuracy of random forest: 85.56%
# AUC of random forest: 94.41%

# Top 3 best models: random forest, support vector machine and neural network

# We select the best model based on AUC and accuracy.
# However, AUC is often preferred over accuracy.
# AUC is calculated based on all possible threshold values, so it should be more robust.
# Accuracy is just an attribute of a given random sample,
# AUC is not concerned with how a single threshold value performs, but rather with
# the predictive performance of all threshold values combined.

# The results of each run will be slightly different due to randomness, but roughly very similar.


# ----------------------------------------------------------------------------------
# ROC plot
# ----------------------------------------------------------------------------------
ROCs_train <- evalm(list(z.rf,z.tree,z.50,z.nb,z.net,z.svm,z.48,z.knn),
               gnames = c('randomForest','tree','C5.0','NaiveBayes',
                          'neural network','SVM','J48','KNN'),
               title = 'AUC-ROC Curve',
               rlinethick = 0.8,fsize = 11, plots = 'r')

# ROC plot on testing
rf.pred <- ROCR::prediction(rf.cath.prob[,2],testing$cath)
rf.perf <- ROCR::performance(rf.pred,"tpr","fpr")
net.pred <- ROCR::prediction(net.cath.prob[,2],testing$cath)
net.perf <- ROCR::performance(net.pred,"tpr","fpr")
svm.pred <- ROCR::prediction(svm.cath.prob[,2],testing$cath)
svm.perf <- ROCR::performance(svm.pred,"tpr","fpr")
J48.pred <- ROCR::prediction(J48.cath.prob[,2],testing$cath)
J48.perf <- ROCR::performance(J48.pred,"tpr","fpr")
tree.pred <- ROCR::prediction(tree.cath.prob[,2],testing$cath)
tree.perf <- ROCR::performance(tree.pred,"tpr","fpr")
C50.pred <- ROCR::prediction(C50.cath.prob[,2],testing$cath)
C50.perf <- ROCR::performance(C50.pred,"tpr","fpr")
nb.pred <- ROCR::prediction(nb.cath.prob[,2],testing$cath)
nb.perf <- ROCR::performance(nb.pred,"tpr","fpr")
knn.pred <- ROCR::prediction(knn.cath.prob[,2],testing$cath)
knn.perf <- ROCR::performance(knn.pred,"tpr","fpr")

# ROC plot of all models
plot(rf.perf,col=2,main='ROC Curve on testing set')
abline(0,1)
plot(net.perf,col=3,add=TRUE)
plot(tree.perf,col=4,add=TRUE)
plot(svm.perf,col=5,add=TRUE)
plot(J48.perf,col=6,add=TRUE)
plot(C50.perf,col=7,add=TRUE)
plot(nb.perf,col=8,add=TRUE)
plot(knn.perf,col=9,add=TRUE)
legend("bottomright",legend = c('Random Forest',"Neural Network","Tree",
                                 "Support Vector Machine","J48","C5.0",
                                 "Naive Bayes","KNN"),col=c(2:9),
       lty=1,cex = 0.75)

# ROC plot of top 3 models
plot(rf.perf,col=2,main='ROC Curve of top 3 models')
abline(0,1)
plot(net.perf,col=3,add=TRUE)
plot(svm.perf,col=4,add=TRUE)
legend("bottomright",legend = c('Random Forest',"Neural Network",
                                "Support Vector Machine"),col=c(2:9),
       lty=1,cex = 0.75)


# ----------------------------------------------------------------------------------
# Ensemble methods
# ----------------------------------------------------------------------------------
# bagged CART
z.bag <- caret::train(cath~.,training,method='treebag',metric="ROC",trControl = trainControl(method = 'repeatedcv',
                                                                                          number = 10,
                                                                                          repeats = 10,
                                                                                          classProbs = TRUE,
                                                                                          savePredictions = TRUE,
                                                                                          summaryFunction = twoClassSummary,
                                                                                          verboseIter = TRUE))
bag.cath.prob <- predict(z.bag,newdata = testing,type = 'prob')
bag.cath <- predict(z.bag,newdata=testing)
bag.cm <- confusionMatrix(data = bag.cath, testing$cath)
bag.ac <- (bag.cm$table[1,1] + bag.cm$table[2,2])/sum(bag.cm$table)
bag.auc <- AUC((bag.cath.prob$Normal),c(testing$cath))

# bootstrapping (with 3 best single learners)
model_list <- caretList(cath~., training, trControl = trainControl(
    method = 'boot',
    number = 25,
    savePredictions = "final",
    classProbs = TRUE,
    summaryFunction = twoClassSummary
  ),
  methodList = c('svmLinear','nnet','rf')
)

model_ensemble <- caretEnsemble(model_list, metric='ROC',trControl = trainControl(method = 'repeatedcv',
                                                                                  number = 10,
                                                                                  repeats = 10,
                                                                                  summaryFunction = twoClassSummary,
                                                                                  classProbs = TRUE))
ens_preds_prob <- predict(model_ensemble, newdata = testing, type = 'prob')
ens_preds <- predict(model_ensemble, newdata = testing)
ens.cm <- confusionMatrix(ens_preds, testing$cath)
ens.ac <- (ens.cm$table[1,1] + ens.cm$table[2,2])/sum(ens.cm$table)
ens.auc <- colAUC(ens_preds_prob, testing$cath)

# stacking
algorithms_to_use <- c('svmLinear','rf','nnet')
control_stacking <- trainControl(method = 'repeatedcv',
                                 number = 10,
                                 repeats = 10,
                                 savePredictions = TRUE,
                                 classProbs = TRUE)
stacked_models <- caretEnsemble::caretList(cath~.,training,trControl = control_stacking,methodList = algorithms_to_use)
                                                                                   
stacking_results <- resamples(stacked_models)
summary(stacking_results)

glm_stack <- caretEnsemble::caretStack(stacked_models,method='glm',metric='ROC',trControl=control_stacking)
print(glm_stack)

stacking.cath <- predict(glm_stack, newdata = testing)
stacking.cath.prob <- predict(glm_stack, newdata = testing, type='prob')
stacking.cm <- confusionMatrix(stacking.cath,testing$cath)
stacking.ac <- (stacking.cm$table[1,1] + stacking.cm$table[2,2])/sum(stacking.cm$table)
stacking.auc <- colAUC(stacking.cath.prob, testing$cath)

# performance table
final_ac <- data.frame(ModelName = c('randomForest','Bagged CART','Bootstrapping','Stacking'),
                       Accuracy = c(rf.ac,bag.ac,ens.ac,stacking.ac))
final_auc <- data.frame(ModelName = c('randomForest','Bagged CART','Bootstrapping','Stacking'),
                        AUC = c(rf.auc,bag.auc,ens.auc,stacking.auc))
final <- merge(final_ac, final_auc)
final[order(final$AUC, decreasing = TRUE),]

# The accuracy and AUC of bootstrapping are slightly higher than our best single model (random forest),
# we may use it as our final model.
# Accuracy of bootstrapping: 86.87%
# AUC of bootstrapping: 95.01%

# ----------------------------------------------------------------------------------
# Conclusion
# ----------------------------------------------------------------------------------
# Final predictive model: ensemble methods - bootstrapping (combining random forest, neural network and support vector machine)