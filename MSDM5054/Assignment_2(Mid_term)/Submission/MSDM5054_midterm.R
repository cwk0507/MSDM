#####Q1
cars=read.csv("cars.csv", header=TRUE,sep=",")
library(boot)
cv.error=rep(0,5)
for (i in 1:5){
  glm.fit=glm(dist~poly(speed,i),data=cars)  
  cv.error[i]=cv.glm(cars,glm.fit)$delta[1]}
cv.error
plot(c(1:5),cv.error,type='o',xlab='degree of polynomial',ylab='cv error',main='LOOCV')

cv.error.5=rep(0,5)
for (i in 1:5){
  glm.fit=glm(dist~poly(speed,i),data=cars)
  cv.error.5[i]=cv.glm(cars,glm.fit,K=5)$delta[1] 
}
cv.error.5
plot(c(1:5),cv.error.5,type='o',xlab='degree of polynomial',ylab='cv error',main='5-fold cross validation')

cv.error.mat=matrix(rep(0,50),5,10)
for (i in 1:10){
  for (j in 1:5){
    glm.fit=glm(dist~poly(speed,j),data=cars)
    cv.error.mat[j,i]=cv.glm(cars,glm.fit,K=5)$delta[1] 
  }
}
df = data.frame(c(1:5),cv.error.mat)
colnames(df) = c('Degree','T1','T2','T3','T4','T5','T6','T7','T8','T9','T10')
df
rowMeans(df[,c(2:11)])
library(ggplot2)
library(reshape2)
df <- melt(df,id.vars = 'Degree', variable.name = 'Test')
p = ggplot(df, aes(Degree,value)) + geom_line(aes(colour = Test))
p + ggtitle("CV with 5 folds") + xlab("Polynomial Degree") + ylab("CV Error")

####Q2
titanic=read.csv("titanic.csv", header=TRUE,sep=",")
summary(titanic)
library(mice)
# perform mice imputation, based on predictive mean matching
t = mice(titanic,maxit=10,meth='pmm',)
titanic.data = complete(t,1)
titanic.data$Pclass = as.character(titanic.data$Pclass)
titanic.model = glm(Survived~Pclass+Sex+Age+SibSp+Fare,data=titanic.data,family=binomial)
summary(titanic.model)
confint(titanic.model,c('Sexmale','Pclass3'),level=0.95)

boot.fn=function(data,index){                             
  return(coef(glm(Survived~Pclass+Sex+Age+SibSp+Fare,data=data,family=binomial,subset=index)))}
library(boot)
boot.obj = boot(titanic.data ,boot.fn ,1000)
boot.ci(boot.obj,conf=0.95,type='perc',index=3)  #for Pclass3
boot.ci(boot.obj,conf=0.95,type='perc',index=4)  #for Sexmale

titanic.prob = predict(titanic.model_1,titanic.data,type='response')
titanic.pred = rep(1,nrow(titanic.data))
titanic.pred = rep(0,nrow(titanic.data))
titanic.pred[titanic.prob>.5]=1
table(titanic.pred,titanic.data$Survived)

####Q3
gpa = read.csv("FirstYearGPA.csv", header=TRUE,sep=",")
library(leaps)
regfit.full=regsubsets(GPA~.,data=gpa)         
reg.summary=summary(regfit.full)
plot(c(1:8),summary(regfit.full)$rsq,
     main='R-square vs model size with best subset selection',xlab='model size', 
     ylab='R-square',type='o')
which.max(reg.summary$adjr2) 

library(magrittr)
library(purrr)
get_model_formula = function(id, object, response){
  models = summary(object)$which[id,-1]
  predictors = names(which(models == TRUE))
  predictors = paste(predictors, collapse = "+")
  as.formula(paste0(response, "~", predictors))
}

get_cv_error = function(model.formula, data){
  glm.fit = glm(model.formula,data=data)
  cv.glm(data,glm.fit,K=5)$delta[1]
}

model.ids = c(1:8)
cv.errors = matrix(nrow=10,ncol=8,dimnames = list(paste("Test",1:10),paste("Size",1:8)))
for (i in 1:10){
  cv.errors[i,] =  map(model.ids, get_model_formula, regfit.full, "GPA") %>%
    map(get_cv_error, data = gpa) %>%
    unlist()
}
cv.errors.avg = colMeans(cv.errors)
cv.errors.avg
names(which.min(cv.errors.avg))
summary(glm(get_model_formula(4,regfit.full,'GPA'),data=gpa))

regfit.fwd=regsubsets(GPA~.,data=gpa,nvmax=8, method ="forward")  
reg.fwd.summary = summary(regfit.fwd)
plot(c(1:8),summary(regfit.fwd)$adjr2,
     main='Adjusted R-square vs model size with forward stepwise selection',xlab='model size', 
     ylab='R-square',type='o')
which.min(reg.fwd.summary$bic) 

model.ids = c(1:8)
cv.fwd.errors = matrix(nrow=10,ncol=8,dimnames = list(paste("Test",1:10),paste("Size",1:8)))
for (i in 1:10){
  cv.fwd.errors[i,] =  map(model.ids, get_model_formula, regfit.fwd, "GPA") %>%
    map(get_cv_error, data = gpa) %>%
    unlist()
}
cv.fwd.errors.avg = colMeans(cv.fwd.errors)
cv.fwd.errors.avg
names(which.min(cv.fwd.errors.avg))
summary(glm(get_model_formula(4,regfit.fwd,'GPA'),data=gpa))

####Q4
q4train=read.csv("diabetes_train.csv", header=TRUE,sep=",")
q4test=read.csv("diabetes_test.csv", header=TRUE,sep=",")
xtrain=model.matrix(Y~.,q4train)[,-1]
ytrain=q4train$Y
xtest=model.matrix(Y~.,q4test)[,-1]
ytest=q4test$Y
library(glmnet)
grid=10^seq(4,-2,length=100)
lasso.mod=glmnet(xtrain,ytrain,alpha=1,lambda=grid)
plot(lasso.mod)

cv.out = cv.glmnet(xtrain,ytrain,alpha=1,K=10)
lambdas = cv.out$lambda
cv.lasso.errors = matrix(nrow=10,ncol=100,dimnames = list(paste("CV",1:10),lambdas))
lambda_min = matrix(nrow=10,ncol=1)
for (i in 1:10){
   cv.out = cv.glmnet(xtrain,ytrain,alpha=1,K=10)
    cv.lasso.errors[i,] = cv.out$cvm
    lambda_min[i] = cv.out$lambda.min
}
cv.lasso.errors.avg = colMeans(cv.lasso.errors)
lambda_min
bestlam=as.numeric(names(which.min(cv.lasso.errors.avg)))
bestlam

cv.lasso.errors.t = t(cv.lasso.errors)
df = data.frame(as.numeric(rownames(cv.lasso.errors.t)),cv.lasso.errors.t)
colnames(df) = c('Lambda',colnames(cv.lasso.errors.t))
library(ggplot2)
library(reshape2)
df = melt(df,id.vars = 'Lambda', variable.name = 'CV')
p = ggplot(df, aes(Lambda,value)) + geom_line(aes(colour = CV))
p + ggtitle("CV with 10 folds") + xlab("Lambda") + ylab("CV Error")


lasso.best=glmnet(xtrain,ytrain,alpha=1,lambda=bestlam)
sum(lasso.best$beta[,1]!=0)
names(lasso.best$beta[,1][lasso.best$beta[,1]!=0])

lasso.pred=predict(lasso.mod,s=bestlam ,newx=xtest)
mean((lasso.pred-ytest)^2)
