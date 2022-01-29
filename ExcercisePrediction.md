---
title: "Predicting Exercise Activity Given Accelerometer Data"
date: "1/29/2022" 
output: 
  html_document:
    keep_md: true
---
## Overview
This analysis builds a predictive model given accelerometer data. Specifically, a set of measurements from accelerometers correspond to one of five specific classes of activity. This analysis will first identify which subset of these variables should be used and then build a predictive model to determine the activity corresponding to a given set of measurements from the accelerometers.

## Getting the Data
Since there are no outcome labels (classe variable) for the testing csv file, it will be treated as a validation set. The data in the training csv file will be split into training and testing sets, and so I will be utilizing **cross-validation** via random subsampling.


```r
library(caret)
```

```r
if(!file.exists("./training_testing.csv")) {
      download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
                    "./training_testing.csv", method = "curl")}

if(!file.exists("./validation.csv")) { 
      download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
                    "./validation.csv", method = "curl")}
      
training_testing <- read.csv("./training_testing.csv")
validation <- read.csv("./validation.csv")

set.seed(999)
indices <- createDataPartition(training_testing$classe, p = .7, list = FALSE)
training <- training_testing[indices,]; trainingOutcomes <-  as.factor(training$classe)
testing <- training_testing[-indices,]; testingOutcomes <-  as.factor(testing$classe)
```

## Exploring, Cleaning the Data
Let's look at the structure of the data.

```r
str(training, list.len = 15)
```

```
## 'data.frame':	13737 obs. of  160 variables:
##  $ X                       : int  2 3 4 5 6 7 8 9 11 13 ...
##  $ user_name               : chr  "carlitos" "carlitos" "carlitos" "carlitos" ...
##  $ raw_timestamp_part_1    : int  1323084231 1323084231 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 1323084232 ...
##  $ raw_timestamp_part_2    : int  808298 820366 120339 196328 304277 368296 440390 484323 500302 560359 ...
##  $ cvtd_timestamp          : chr  "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" "05/12/2011 11:23" ...
##  $ new_window              : chr  "no" "no" "no" "no" ...
##  $ num_window              : int  11 11 12 12 12 12 12 12 12 12 ...
##  $ roll_belt               : num  1.41 1.42 1.48 1.48 1.45 1.42 1.42 1.43 1.45 1.42 ...
##  $ pitch_belt              : num  8.07 8.07 8.05 8.07 8.06 8.09 8.13 8.16 8.18 8.2 ...
##  $ yaw_belt                : num  -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 -94.4 ...
##  $ total_accel_belt        : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ kurtosis_roll_belt      : chr  "" "" "" "" ...
##  $ kurtosis_picth_belt     : chr  "" "" "" "" ...
##  $ kurtosis_yaw_belt       : chr  "" "" "" "" ...
##  $ skewness_roll_belt      : chr  "" "" "" "" ...
##   [list output truncated]
```
Looking at the structure of the training data, we can see there are 159 independent variables and so 160 total variables. The first few of these, including user name and some timestamp measurements, shouldn't be used in making the the model because they are irrelevant at best, and could lead to overfitting at worst. 

We can further see a mixture of numeric, integer and character vectors. It seems some of these character vectors are supposed to be numerics/integers; however, coercing these vectors to numerics reveals that most of the values in many of them seem to be NA. Thus, it seems they may not have much predictive value; further, imputing them will likely be computationally expensive and there's also very little information to use to impute all of the missing values with.


```r
slices <- c(sum(is.na(as.numeric(training$kurtosis_roll_belt))), sum(!is.na(as.numeric(training$kurtosis_roll_belt))))
slices2 <- c(sum(is.na(as.numeric(training$skewness_roll_belt))), sum(!is.na(as.numeric(training$skewness_roll_belt))))
par(mfrow = c(1,2))
pie(slices, col = 2:3, labels = c("NA", "Not NA"), main = "kurtosis_roll_belt values")
pie(slices2, col = 2:3, labels = c("NA", "Not NA"), main = "skewness_roll_belt values")
```

![](ExcercisePrediction_files/figure-html/unnamed-chunk-4-1.png)<!-- -->

So, at this point, I will remove the first few predictors and all that are not numerics/integers. 


```r
training <- training[,-(1:5)]
classes <- lapply(training, class)
numericIndices <- which(classes == "integer" | classes == "numeric")
training <- training[,numericIndices]
```

However, there are still NA values in some of our remaining predictors. I am going to take a look at the average proportion of missing values in the predictors that have at least one missing value.

```r
NA.proportions <- numeric()
nonNAIndices <- numeric()
for (i in 1:ncol(training)) {
      if(sum(is.na(training[,i])) == 0) {nonNAIndices <- c(nonNAIndices, i)}
      else {NA.proportions <- c(NA.proportions, mean(is.na(training[,i])))}
}
meanNA <- mean(NA.proportions)
```
The average percentage of observations that are NA for predictors that have at least one NA is 97.85. Being so high, I will just remove all remaining predictors that have any missing values in them. Now we have the full set of predictors that will be used for building our model, the names of which are listed below.

```r
training <- training[,nonNAIndices]
training$classe <- trainingOutcomes
names(training)
```

```
##  [1] "num_window"           "roll_belt"            "pitch_belt"          
##  [4] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
##  [7] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [10] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [13] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [16] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [19] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [22] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [25] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [28] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [31] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [34] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [37] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [40] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [43] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [46] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [49] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [52] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

## Building a Model
Being that **random forests** tend to perform well, I will build a random forest model.


```r
model <- train(classe ~ ., data = training, method = "rf")
```



```r
print(100)
```

```
## [1] 100
```

```r
paste("TEST SET ACCURACY: ", confusionMatrix(testingOutcomes, predict(model, testing))$overall[1])
```

```
## [1] "TEST SET ACCURACY:  0.997790994052676"
```

```r
table(predict(model, testing), testingOutcomes)
```

```
##    testingOutcomes
##        A    B    C    D    E
##   A 1673    5    0    0    0
##   B    1 1133    0    0    0
##   C    0    1 1026    6    0
##   D    0    0    0  958    0
##   E    0    0    0    0 1082
```

The test set accuracy shown above should be a strong indicator of our **out of sample error**. Thus, there is good reason to believe that this random forest model will yield accurate predictions on new data, such as the validation data. The predicted outcomes for the validation set are given below.


```r
paste("VALIDATION PREDICTIONS: ", predict(model, validation))
```

```
##  [1] "VALIDATION PREDICTIONS:  B" "VALIDATION PREDICTIONS:  A"
##  [3] "VALIDATION PREDICTIONS:  B" "VALIDATION PREDICTIONS:  A"
##  [5] "VALIDATION PREDICTIONS:  A" "VALIDATION PREDICTIONS:  E"
##  [7] "VALIDATION PREDICTIONS:  D" "VALIDATION PREDICTIONS:  B"
##  [9] "VALIDATION PREDICTIONS:  A" "VALIDATION PREDICTIONS:  A"
## [11] "VALIDATION PREDICTIONS:  B" "VALIDATION PREDICTIONS:  C"
## [13] "VALIDATION PREDICTIONS:  B" "VALIDATION PREDICTIONS:  A"
## [15] "VALIDATION PREDICTIONS:  E" "VALIDATION PREDICTIONS:  E"
## [17] "VALIDATION PREDICTIONS:  A" "VALIDATION PREDICTIONS:  B"
## [19] "VALIDATION PREDICTIONS:  B" "VALIDATION PREDICTIONS:  B"
```

Let's look at which predictor variables were most important in creating this model.

```r
varImp(model)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 53)
## 
##                      Overall
## num_window           100.000
## roll_belt             66.939
## pitch_forearm         40.801
## yaw_belt              31.814
## magnet_dumbbell_z     29.694
## magnet_dumbbell_y     28.801
## pitch_belt            27.638
## roll_forearm          25.428
## accel_dumbbell_y      13.135
## magnet_dumbbell_x     10.938
## roll_dumbbell         10.815
## accel_forearm_x       10.478
## accel_belt_z           9.532
## total_accel_dumbbell   9.453
## accel_dumbbell_z       8.580
## magnet_belt_z          7.362
## magnet_forearm_z       7.115
## magnet_belt_y          6.974
## magnet_belt_x          6.399
## yaw_dumbbell           5.230
```

## Summary
The above analysis built a random forest predictive model. Given accelerometer data, this model yields a corresponding activity label. Random subsampling **cross-validation** was used to estimate an **out of sample error**.



