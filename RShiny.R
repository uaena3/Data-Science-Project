# ----------------------------------------------------------------------------------
# Code for the R shiny Website
# Author: Xinyu Ma, Zhaofa Guo
# Date: 22/10/2020

# System: macOS
# RStudio version: 1.3.1093
# How to run:  click 'Run App'
# Approximate runtime: 2 minutes
# ----------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------
# User Interface
# ----------------------------------------------------------------------------------
ui <- fluidPage(
  # Application title
  headerPanel("Cad Prediction"),
  sidebarLayout(
    # Sidebar file input to select a csv file
    sidebarPanel(
      fileInput("file1", "Choose CSV File",
                accept = c(
                  "text/csv",
                  "text/comma-separated-values,text/plain",
                  ".csv")
      )
    ),
    # show the result of prediction
    mainPanel(
      h6(tableOutput("table")),# the data will be shown above
      h1("The result is:"),
      h1(textOutput("contents")),# this is the path of result
      h2("--------------------"),
      h3("This result is indicative only.  Please see you Doctor for proper diagnosis."),
      h3("The accuracy of the model is: 86.67%.")
      
      
    )
  )
)

# ----------------------------------------------------------------------------------
# Server
# ----------------------------------------------------------------------------------
server <- function(input, output) {
  
  output$table <-renderTable({
    inFile <- input$file1
    
    if (is.null(inFile)) # even user do not upload, no error message will be displayed
      return(NULL)
    
    read.csv(inFile$datapath) # read the user input
  })
  
  output$contents <- renderText({
    inFile <- input$file1
    # input$file1 will be NULL initially. After the user selects and uploads a 
    # file, it will be a data frame with 'name', 'size', 'type', and 'datapath' 
    # columns. The 'datapath' column will contain the local file names where the 
    # data can be found.
    
    
    # return NULL when there is no input file
    if (is.null(inFile))
      return(NULL)
    
    # ----------------------------------------------------------------------------------
    # The model below is sourced from the code for the predictive model
    # ----------------------------------------------------------------------------------
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
    # ----------------------------------------------------------------------------------
    set.seed(9999)
    colnames(z.data)[1] <- 'Age'
    
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
    z.rf <- caret::train(cath~.,training,method='rf',metric="ROC",trControl = trainControl(method = 'repeatedcv',
                                                                                           number = 10,
                                                                                           repeats = 10,
                                                                                           classProbs = TRUE,
                                                                                           savePredictions = TRUE,
                                                                                           summaryFunction = twoClassSummary,
                                                                                           verboseIter = TRUE))
    # prediction and confusion matrix
    rf.cath.prob <- predict(z.rf,newdata = testing,type = 'prob')  # return probability of prediction
    rf.cath <- predict(z.rf,newdata=testing)   # return prediction label
    rf.cm <- confusionMatrix(data = rf.cath, testing$cath)
    
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
    
    # ----------------------------------------------------------------------------------
    #The sourced code for the predictive model ends here
    # ----------------------------------------------------------------------------------
    
    
    #read the user input csv file
    sick <- read.csv(inFile$datapath)
    result <- predict(model_ensemble,newdata=sick)
    result
    result <- as.character(result)#the result need to be shown as a string
    result
    
  })
  
}

shinyApp(ui, server) # the function to use the R shiny website
# ----------------------------------------------------------------------------------
# limitations and potential improvement: The website needs more than 1 minutes to get the        
# result. It would be better to have more functions on the website. For example, after user     # get the result, they can download the result as well. Because of the lack of time and 
# knowledge, a lot of ideas cannot be realised.
# ----------------------------------------------------------------------------------
# user guide: upload one csv file at one time
# ----------------------------------------------------------------------------------
# technical user guide: 
# install the shiny package before run the app
# ----------------------------------------------------------------------------------
