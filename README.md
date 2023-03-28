Titanic Survival Prediction Using Radial Basis Function Support Vector
Machines, Linear Support Vector Machines, Random Forest, Decision Tree and
Linear Regression
================

``` r
library(dplyr)
set.seed(1)
titanic3 <- read.csv("titanic3.csv")
titanic3 <- select(titanic3, -ticket, -boat, -body, -home.dest, -cabin, -embarked) %>%
mutate(sex = factor(sex),pclass=factor(pclass))
titanic3$survived = as.factor(titanic3$survived)
titanic3 <- na.omit(titanic3)
summary(titanic3)
```

Each row in the data is a passenger. Columns are variables:

-   `survived`: the class attribute - 0 if died, 1 if survived
-   `sex`: Gender
-   `sibsp`: Number of Siblings/Spouses Aboard
-   `parch`: Number of Parents/Children Aboard
-   `fare`: Fare Payed
-   `pclass`: Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
-   `age`: Age

1.  Randomly select 80% of the observations in the titanic3 dataset to
    create a training dataset, then use the remaining observations to
    create a testing dataset.

``` r
# sample 20% or 1/5 of the whole set
testing_index <- sample(1:nrow(titanic3), nrow(titanic3)/5)
# testing set is therefore 20%
testing <- titanic3[testing_index,]
# training set is 100-20 = 80%
training <- titanic3[-testing_index,]
```

2.  Use the created training dataset to conduct a 10-fold cross
    validation to evaluate the predictive accuracy of 5 different
    classification algorithms introduced in this module, i.e. logistic
    regression, decision tree, random forests, linear kernel-based svm
    and rbf kernel-based svm. Define set.seed(1) and use default
    parameters of those corresponding r functions.

``` r
library(caret)
library(tree)
library(randomForest)
library(e1071)
set.seed(1)

folds <- createFolds(y=training[,2],k=10)
acc_value_glm<-as.numeric()
acc_value_tree<-as.numeric()
acc_value_rf<-as.numeric()
acc_value_svm_linear<-as.numeric()
acc_value_svm_rbf<-as.numeric()

for(i in 1:10){
    fold_cv_test <- training[folds[[i]],]
    fold_cv_train <- training[-folds[[i]],]
    
    # linear regression
    train.glm <- glm(survived ~ ., data = fold_cv_train, family = "binomial")
    pred.prob.glm <- predict(train.glm, newdata = fold_cv_test, type='response')
    pred.glm <- rep(1,length(pred.prob.glm))
    pred.glm[pred.prob.glm<0.5] <- 0 # 0.5 is threshold, otherwise assigned to 0
    acc.glm.fold <- mean(pred.glm == fold_cv_test$survived)
    acc_value_glm <- append(acc_value_glm, acc.glm.fold)
    
    # decision tree
    train.tree <- tree(survived ~ ., data = fold_cv_train)
    pred.tree <- predict(train.tree, newdata = fold_cv_test, type="class")
    acc.tree.fold <- mean(pred.tree == fold_cv_test$survived)
    acc_value_tree <- append(acc_value_tree, acc.tree.fold)
    
    # random forest
    train.rf <- randomForest(survived ~ ., data = fold_cv_train)
    pred.rf <- predict(train.rf, newdata = fold_cv_test, type="class")
    acc.rf.fold <- mean(pred.rf == fold_cv_test$survived)
    acc_value_rf <- append(acc_value_rf, acc.rf.fold)
    
    # linear Support Vector Machines
    train.svm.linear <- svm(survived ~ ., data = fold_cv_train, kernel = "linear")
    pred.svm.linear <- predict(train.svm.linear, newdata = fold_cv_test)
    acc.svm.linear.fold <- mean(pred.svm.linear == fold_cv_test$survived)
    acc_value_svm_linear <- append(acc_value_svm_linear, acc.svm.linear.fold)
    
    # Radial Basis Function Support Vector Machines
    train.svm.rbf <- svm(survived ~ ., data = fold_cv_train, kernel = "radial")
    pred.svm.rbf <- predict(train.svm.rbf, newdata = fold_cv_test)
    acc.svm.rbf.fold <- mean(pred.svm.rbf == fold_cv_test$survived)
    acc_value_svm_rbf <- append(acc_value_svm_rbf, acc.svm.rbf.fold)
}

print(mean(acc_value_glm))
print(mean(acc_value_tree))
print(mean(acc_value_rf))
print(mean(acc_value_svm_linear))
print(mean(acc_value_svm_rbf))
```

#The rbf kernel-based svm is the optimal classification algorihtm since
it obtains the highest predictive accuracy (i.e. 81.1%) over the 10-fold
cross validation.

3.  Use the entire training dataset to retrain the best classification
    algorithm found by the cross-validation, then report the predictive
    accuracy of the retrained model on the testing dataset.

``` r
svm.rbf <- svm(survived ~ ., data = training, kernel = "radial")
pred.test.svm.rbf <- predict(svm.rbf, newdata = testing)
acc.testing.svm.rbf <- mean(pred.test.svm.rbf == testing$survived)
print(acc.testing.svm.rbf)
```

#The predictive accuracy of the retrained rbf kernel-based svm is 78.9%.
