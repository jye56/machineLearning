---
title: "Machine Learning Project"
output: html_document
---

# 1. SYNOPSIS

Exercise data were obtained from Groupware Technologies (http://groupware.les.inf.puc-rio.br/har). The dataset has 5 classes  of activity: sitting-down, standing-up, standing, walking, and sitting, desginated as A,B,C,D, and E, collected on 8 hours of activities of 4 healthy subjects, with 159 attributes. After tidying up the data, the data were partitioned into training set and testing set for the purpose of cross validation. Three different models, classification trees model, boosting (gbm) model, and random forest model, were fit. While the classification trees model has the lowest accuracy, only 49.26%, boosting model and random forest model have the accuracy of 98.69% and 99.78%, respectively.

Accuracies calculated using testing data subsets are slightly lower than those calculated using the training data subset (0.4926 vs 0.4965 for classification trees, 0.9869 vs 0.9939 for gbm model, and 0.9978 vs 1.000 for rf model). These numbers are consistent with the expectation that the out of sample error should be higher, i.e, the testing  sample accuracy should be lower.

The above three models predicted 8, 20 and 20 correctly out of 20 test cases in the submission portion of the project.

# 2. Modeling considerations

Classification trees model, boosting model and random forest models were tested because these classification models do not require many assumptions on the underlying data. Aslo, no preprocessing is needed. Model based prediction such as linear discriminaant analysis was not tested. 

Time required for fitting classification trees, gbm model and rf model are approximately 2 minutes, 30 miniutes and 70 minutes, respectively, on my laptop (Intel cpu 2.6GH, Ram 4GB, 64-bit Windows 7).


# 3. Supplemental material/modeling details

The following codes and output are related to data loading, tidying, modeling and accuracy calculations. Exploratory data analysis is not included.


```r
# Here are codes that were used to load the data, but commented out for not re-runining it each time
#url<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
#download.file(url, destfile="mlTrain.csv")

#url2<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
#download.file(url2, destfile="mlTest.csv")
```


### Load libraries


```r
suppressMessages(library(dplyr))
suppressMessages(library(caret))
library(ggplot2)
```
### Load data and tidy up data:

For tidying up the data, columns that are secondary (calculated) in nature and are mostly NA are removed. Also, time stamps and index are not expected to contribute to the final model and are therefore remove. The data frame is reduced from 160 columns to 56 columns.


```r
mytrain<-read.csv("mlTrain.csv")

mytrain<-select(mytrain, -starts_with("max_"),-starts_with("min_"),-starts_with("amplitude"),-starts_with("kurtosis_"),-starts_with("skewness"),-starts_with("var_"),-starts_with("avg_"),-starts_with("stddev"),-X,-c(raw_timestamp_part_1:cvtd_timestamp))

mytest<-read.csv("mlTest.csv")

mytest<-select(mytest, -starts_with("max_"),-starts_with("min_"),-starts_with("amplitude"),-starts_with("kurtosis_"),-starts_with("skewness"),-starts_with("var_"),-starts_with("avg_"),-starts_with("stddev"),-X,-c(raw_timestamp_part_1:cvtd_timestamp))
```

### Partition the data set for cross validation

```r
set.seed(1028)
inTrain<-createDataPartition(y=mytrain$classe,p=0.7, list=F)
training<-mytrain[inTrain,]
testing<-mytrain[-inTrain,]
```
### Fit different models

Using tree model to fit the data

```r
modfittree<-train(classe~.,data=training, method='rpart')
```

```
## Loading required package: rpart
```

Using boosting (gbm) to fit the data

```r
modfitgbm <- train(classe ~ ., method="gbm", data=training,verbose=FALSE)
```

```
## Loading required package: gbm
## Loading required package: survival
## 
## Attaching package: 'survival'
## 
## The following object is masked from 'package:caret':
## 
##     cluster
## 
## Loading required package: splines
## Loading required package: parallel
## Loaded gbm 2.1.1
## Loading required package: plyr
## -------------------------------------------------------------------------
## You have loaded plyr after dplyr - this is likely to cause problems.
## If you need functions from both plyr and dplyr, please load plyr first, then dplyr:
## library(plyr); library(dplyr)
## -------------------------------------------------------------------------
## 
## Attaching package: 'plyr'
## 
## The following objects are masked from 'package:dplyr':
## 
##     arrange, count, desc, failwith, id, mutate, rename, summarise,
##     summarize
```
gbm took 30 min to run on my laptop.

Using random forests to fit the data

```r
set.seed(1028)
modfitrf<-train(classe~.,data=training,method='rf')
```

```
## Loading required package: randomForest
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## 
## The following object is masked from 'package:dplyr':
## 
##     combine
```


### Accuracies of the prediction models


```r
length(training[,1]);length(testing[,1])
```

```
## [1] 13737
```

```
## [1] 5885
```

```r
#predicting on the training data used for model building
predtrain<-predict(modfittree,newdata=training)
table(predtrain,training$classe)
```

```
##          
## predtrain    A    B    C    D    E
##         A 3560 1078 1116 1002  363
##         B   49  902   70  399  341
##         C  286  678 1210  851  672
##         D    0    0    0    0    0
##         E   11    0    0    0 1149
```

```r
#predicting on the test data selected from the train
pred<-predict(modfittree,newdata=testing)
table(pred,testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1515  487  470  434  161
##    B   32  384   38  169  145
##    C  124  268  518  361  294
##    D    0    0    0    0    0
##    E    3    0    0    0  482
```

```r
#predicting on the training data used for model building
predtrain<-predict(modfitgbm,newdata=training)
table(predtrain,training$classe)
```

```
##          
## predtrain    A    B    C    D    E
##         A 3899   10    0    0    0
##         B    7 2633    9    3    3
##         C    0   15 2381   17    3
##         D    0    0    3 2230    9
##         E    0    0    3    2 2510
```

```r
#predicting on the test data selected from the train
pred<-predict(modfitgbm,newdata=testing)
table(pred,testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1665    7    0    1    2
##    B    9 1114    7    3    5
##    C    0   15 1016    9    3
##    D    0    1    2  947    6
##    E    0    2    1    4 1066
```

```r
#predicting on the training data used for model building
predtrain<-predict(modfitrf,newdata=training)
table(predtrain,training$classe)
```

```
##          
## predtrain    A    B    C    D    E
##         A 3906    0    0    0    0
##         B    0 2658    0    0    0
##         C    0    0 2396    0    0
##         D    0    0    0 2252    0
##         E    0    0    0    0 2525
```

```r
#predicting on the test data selected from the train
pred<-predict(modfitrf,newdata=testing)
table(pred,testing$classe)
```

```
##     
## pred    A    B    C    D    E
##    A 1674    1    0    0    0
##    B    0 1136    4    0    0
##    C    0    2 1022    6    0
##    D    0    0    0  958    0
##    E    0    0    0    0 1082
```
### Accuracies:

The above data are used to calculate the prediction accuracy for all three models

Accuracy for classification trees:
training data: 6821/13737=0.4965 testing data: 2899/5885=0.4926

Accuracy for gbm model:
training data: 13653/13737=0.9939 testing data: 5808/5885=0.9869

Accuracy for rf model:
training data: 13737/13737=1.000 testing data: 5872/5885=0.9978


