# Practical Machine Learning - Prediction Assignment
Vishanta Rayamajhi  
November 21, 2015  

##Background
As part of the course project for **Practical Machine Learning** in Coursera, we were provided with a training and a testing set of activity data of six young health participants, collected from accelerometers on the belt, forearm, arm, and dumbell, where each participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

##Goal
The goal of the project is to predict the manner in which the participants did the exercise. Build a predictive model to be able to predict the outcome 'classe' in the testing set that falls in one of the following classes:

- Exactly according to the specification (Class A)
- Throwing the elbows to the front (Class B)
- Lifting the dumbbell only halfway (Class C)
- Lowering the dumbbell only halfway (Class D)
- Throwing the hips to the front (Class E)

##Loading data
The data is downloaded and stored in the 'data' directory. We load the data by changing the missing values those that are coded as string "#DIV/0!" or "" to "NA" in order to maintain uniformity with missing values.

```r
# Load training set
pml_training <- read.csv("data/pml-training.csv", na.strings=c("NA","#DIV/0!", ""))

# Load testing set
pml_testing <- read.csv("data/pml-testing.csv", na.strings=c("NA","#DIV/0!", ""))
```

##Load library

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

##Data structure

```r
dim(pml_training)
```

```
## [1] 19622   160
```

```r
dim(pml_testing)
```

```
## [1]  20 160
```

```r
#str(pml_training)
#str(pml_testing)

table(pml_training$user_name) # summary(pml_training$user_name)
```

```
## 
##   adelmo carlitos  charles   eurico   jeremy    pedro 
##     3892     3112     3536     3070     3402     2610
```

```r
table(pml_training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

```r
table(pml_training$classe, pml_training$user_name)
```

```
##    
##     adelmo carlitos charles eurico jeremy pedro
##   A   1165      834     899    865   1177   640
##   B    776      690     745    592    489   505
##   C    750      493     539    489    652   499
##   D    515      486     642    582    522   469
##   E    686      609     711    542    562   497
```

##Cleaning Data
The preliminary study of the summary of the training set reveals more insight of the structure of the data which enables us to come up with a number of cleaning actions below:

- Remove not relevant columns for classification (the first 7 columns)
- Remove columns with over 95% of NAs
- Exclude near zero variance predictors
- Convert dependent variable "classe" into factor, if not

Let's first check the columns with NA values before proceeding with cleaning the data.


```r
summary(colSums(is.na(pml_training)) == 0) # table(colSums(is.na(pml_training)) == 0)
```

```
##    Mode   FALSE    TRUE    NA's 
## logical     100      60       0
```

The result shows we have 100 columns with zero variability, which could be excluded in our prediction model.


```r
# Remove the first 7 columns
training.set <- pml_training[,-c(1:7)] ## 153 covariants
#training.set <- pml_training[,8:ncol(pml_training)] # or dim(pml_training)[2]

# Remove missing data
NAs <- apply(training.set, 2, function(x) {sum(is.na(x))}) # colSums(is.na(training.set))
training.set <- training.set[,which(NAs < nrow(training.set)*0.95)] ## 53 covariants
#training.set <- training.set[,colSums(is.na(training.set)) == 0] # deletes columns with all NAs

# Exclude near zero variance predictors
table(nearZeroVar(training.set, saveMetrics = TRUE)$nzv) # FALSE=53, implies no nzv due to data cleaning
```

```
## 
## FALSE 
##    53
```

```r
nzv.columns <- nearZeroVar(training.set, saveMetrics = TRUE)
training.set <- training.set[,nzv.columns$nzv == FALSE] # takes the 53 'FALSE' columns
#nzv.columns <- nearZeroVar(training.set)
#training.set <- training.set[, -nzv.columns] # gives empty set with 19622 rows

# Convert classe into factor
training.set$classe <- factor(training.set$classe)
```

##Feature Selection
The final set of features used for classification are as follows.

```r
names(training.set)
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
## [52] "magnet_forearm_z"     "classe"
```

##Exploratory Data Analysis (EDA)
Plots...

##Cross validation: Splitting data
Split the dataset into 60% training and 40% validation set in order to perform cross validation. This is performed using random sampling without replacement.


```r
inTrain <- createDataPartition(y=training.set$classe, p=0.60, list=FALSE)
training <- training.set[inTrain,]
validation <- training.set[-inTrain,]
```

##Fit a Model
We will be using a couple of model and find out the one that provides high accuracy and low out of sample error. The Principal Component Analysis (PCA) can also be added as a preprocess option of the train function, but at the expense of losing accuracy. Hence, we will omit PCA model.

- Three models are generated: random forest ("rf"), boosted trees ("gbm") and linear discriminant analysis ("lda") model.
- Parallel computing methods are employed to improve efficiency.


```r
# set a seed
set.seed(6789)
```

**Decision Tree**
First, we start with decision tree model.

**Random Forest**
When I first ran the model with default train() function, it took over half an hour and I stopped the exeuction in the middle. The first thing to note is that by default {caret}train() uses bootstrap resampling which is very computationally intensive. Hence, I used optimization method using trControl parameter for tuning model training. So I switched to cross-validation with a low k=3, and played around with it to get a compromise between accuracy and speed. With this the model completed below 6 mins.


```r
# train a model - random forest (rf)
#modelRF <- train(classe ~ ., data=training, method="rf", trControl = trainControl(method = "cv", number = 3))
#modelRF
#modelRF$finalModel

# prediction
#predictions <- predict(modelRF, newdata=testing)
#predictions

# quick checks
#table(testing$classe)
#table(predictions)

# confusion matrix
#confusionMatrix(predictions, testing$classe) # Accuracy: %
```

"Tune with cross validation" using 10 folds to avoid overfitting.

##Accuracy and Out-of-Sample Error
This is acheived through Confusion Matrix.

##Conclusion
Random Forest generates prediction with highest accuracy and lowest out of sample error compared to other models.
