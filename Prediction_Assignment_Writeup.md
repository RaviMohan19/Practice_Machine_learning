---
title: "Prediction Assignment Writeup"
author: "Ravi M.B"
date: "June 10, 2018"
output: 
  html_document: 
    fig_caption: yes
    keep_md: yes
    toc: yes
---
# Project Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively, these devices are a part of quantified self-movement which consists of group of enthusiasts who take measurements about themselves regularly to improve their health. Usually people tend to quantify how much of an activity they do instead of quantifying how well or good they perform the activity. 

# Project Goal

In this project the goal is to use the data from accelerometers on the belt, forearm, arm, and dumbbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The training data [Training](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv) for this projecte is extracted from * CoursEra * instructions page, similarly test data [Test](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv) was extracted from the same page.

The entire data for the project comes from the [Source]( http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har). It is cited here for future references and reproducibility.

# Analysis of the Data

This section covers the analysis of both the training data and testing data.

## Relevant libraries invoked
At first, loaded the necessary package libraries for analysis, then the data is downloaded in to the working directory

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.4.4
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
library(rpart)
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.4.4
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(e1071)
```

```
## Warning: package 'e1071' was built under R version 3.4.4
```

```r
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.4.4
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.3
```

```r
library(ggplot2)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.4.4
```

```r
setwd("~/R/Practice_Machine_Learning/Project")
```

## Loading the data 



## Data partioning and cleaning for prediction
Now post loading the data, as per instructions listed on [CoursEra](https://www.coursera.org/learn/practical-machine-learning/peer/R43St/prediction-assignment-writeup), I will be using the "*classe*" variable in the training set to predict the manner in which the exercise activity is performed, so the *training data* is split further in to two partitions and use the *testing data * for validation, also the seven variables irrlevant for the analysis and the prediction were excluded. 


```r
set.seed(123) 
training<-training[,colSums(is.na(training)) == 0]
testing <-testing[,colSums(is.na(testing)) == 0]
head(colnames(training))
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"
```

```r
training <- training[,8:dim(training)[2]]
testing <- testing[,8:dim(testing)[2]]
seedData1 <- createDataPartition(y = training$classe, p = 0.8, list = F)
seedData2 <- training[seedData1,]
validation_data <- training[-seedData1,]
training_data1 <- createDataPartition(y = seedData2$classe, p = 0.75, list = F)
training_data2 <-seedData2[training_data1,]
testing_data <- seedData2[-training_data1,]
qplot(classe, fill = "4", data=training_data2, main="Distribution of Classe Variable")
```

![](Prediction_Assignment_Writeup_files/figure-html/Data Partitioning-1.png)<!-- -->

### Predictors and the building model for the predictors Out of sample error

using the names function,the predictors could be listed for the data analysis created and mentioned above, the *error rate* you get on a new data set it is also referred as *generalization error* , as a part of the *CoursEra* week 3 teachings, I have used the techniques taught during week3  * model based predictions * ,  *predicting with trees* and *Random forest* to build the prediction model.


```r
names(training_data2[,-53])
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"
```

```r
model_tree <- rpart(classe ~ ., data=training_data2, method="class")
prediction_tree <- predict(model_tree, testing_data, type="class")
class_tree <- confusionMatrix(prediction_tree, testing_data$classe)
rpart.plot(model_tree)
```

![](Prediction_Assignment_Writeup_files/figure-html/Predictors-1.png)<!-- -->

```r
forest_model <- randomForest(classe ~ ., data=training_data2, method="class")
prediction_forest <- predict(forest_model, testing_data, type="class")
random_forest <- confusionMatrix(prediction_forest, testing_data$classe)
class_tree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 944 111   7  53  42
##          B  53 492  51  59  53
##          C  49  98 447  54  51
##          D  47  33 178 468  70
##          E  23  25   1   9 505
## 
## Overall Statistics
##                                           
##                Accuracy : 0.728           
##                  95% CI : (0.7138, 0.7419)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6559          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8459   0.6482   0.6535   0.7278   0.7004
## Specificity            0.9241   0.9317   0.9222   0.9000   0.9819
## Pos Pred Value         0.8159   0.6949   0.6395   0.5879   0.8970
## Neg Pred Value         0.9378   0.9170   0.9265   0.9440   0.9357
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2406   0.1254   0.1139   0.1193   0.1287
## Detection Prevalence   0.2949   0.1805   0.1782   0.2029   0.1435
## Balanced Accuracy      0.8850   0.7900   0.7879   0.8139   0.8412
```

```r
random_forest
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    2    0    0    0
##          B    0  753    5    0    0
##          C    0    3  678    9    0
##          D    0    1    1  634    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9915, 0.9965)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9921   0.9912   0.9860   0.9986
## Specificity            0.9993   0.9984   0.9963   0.9991   1.0000
## Pos Pred Value         0.9982   0.9934   0.9826   0.9953   1.0000
## Neg Pred Value         1.0000   0.9981   0.9981   0.9973   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1919   0.1728   0.1616   0.1835
## Detection Prevalence   0.2850   0.1932   0.1759   0.1624   0.1835
## Balanced Accuracy      0.9996   0.9953   0.9938   0.9925   0.9993
```

```r
prediction1 <- predict(forest_model, newdata=testing_data)
confusionMatrix(prediction1, testing_data$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    2    0    0    0
##          B    0  753    5    0    0
##          C    0    3  678    9    0
##          D    0    1    1  634    1
##          E    0    0    0    0  720
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9944          
##                  95% CI : (0.9915, 0.9965)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9929          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9921   0.9912   0.9860   0.9986
## Specificity            0.9993   0.9984   0.9963   0.9991   1.0000
## Pos Pred Value         0.9982   0.9934   0.9826   0.9953   1.0000
## Neg Pred Value         1.0000   0.9981   0.9981   0.9973   0.9997
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1919   0.1728   0.1616   0.1835
## Detection Prevalence   0.2850   0.1932   0.1759   0.1624   0.1835
## Balanced Accuracy      0.9996   0.9953   0.9938   0.9925   0.9993
```
Finally comparing *tree based* and *Random Forest* models, it came to the observation that *Random Forest* model is more accurate.

### Out of Sample Error 

As mentioned during the course,the *out of sample error* is the "* error rate you get on new data set*", in this study the *out of sample error* is the error rate that is observed after executing the *predict() function* on the *model_tree* and *forest_model* variables using the partitioned *training data* and the *Cross Validation Testing data : Validation*  


# Summary and Conclusion

A prediction model was created using the *Random Forest* and *Tree Based* principles for the data analysis, in this study for plausible prediction the characteristic s of both *Training Data* and *Testing Data* were considerable reduced.

The characteristics that were reduced are the  *Percentage of NA values*, *low variation* , *correlation * and etc., the *training data* was further partiotioned in to * subtraining * and *Validation* to implement a  predictive model for accuracy. 



*Decision Tree Based* and *Random Forest* pricinples were applied and it was observed that the model with *Random Forest* principled model provided better accuracy than *Tree Based* model





