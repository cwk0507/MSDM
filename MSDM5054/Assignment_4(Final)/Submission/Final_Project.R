#Part 1
#random forest
library(randomForest)
doc = read.delim("documents.txt",header=FALSE,sep='\t')
groupnames = read.delim("groupnames.txt",header=FALSE,sep='\t')
doc.group = read.delim("newsgroups.txt",header=FALSE,sep='\t')
wordlist = read.delim("wordlist.txt",header=FALSE,sep='\t')
doc.table = matrix(0, nrow = 16242, ncol = 100)
for (i in 1:nrow(doc)){
  doc.table[doc[i,1],doc[i,2]]=doc[i,3]
}
colnames(doc.table)=t(wordlist)
doc.data = cbind(doc.table,doc.group)
colnames(doc.data)=c(t(wordlist),'newsgroup')
doc.data$newsgroup = as.factor(doc.data$newsgroup)
k=5
folds=sample(1:k,nrow(doc.data),replace=TRUE)   
cv.errors=matrix(0,nrow=5,ncol=25)
for (n in 1:25){
  for (j in 1:k){
    train=as.numeric(rownames(doc.data[folds!=j,]))
    test=doc.data[folds==j,]
    doc.rf=randomForest(doc.table,doc.data$newsgroup,data=doc.data,subset=train,mtry=10,ntree=2*n,importance=TRUE)
    yhat.rf=predict(doc.rf ,newdata=test)
    cv.errors[j,n]=sum(1-(yhat.rf==test$newsgroup))/nrow(test)
  }
}
cv.errors.mean=colMeans(cv.errors)
plot(seq(2,50,2),cv.errors.mean,type='o',xlab='number of trees',ylab='misclassfication cv error',main='5-fold cv of random forest with p=10')
cv.errors.mean.best = min(cv.errors.mean)
ntree.best = which(cv.errors.mean==cv.errors.mean.best)*2
print(c(ntree.best,cv.errors.mean.best))

cv.errors.mtry=matrix(0,nrow=5,ncol=10)
for (n in 1:10){
  for (j in 1:k){
    train=as.numeric(rownames(doc.data[folds!=j,]))
    test=doc.data[folds==j,]
    doc.rf=randomForest(doc.table,doc.data$newsgroup,data=doc.data,subset=train,mtry=10*n,ntree=20,importance=TRUE)
    yhat.rf=predict(doc.rf ,newdata=test)
    cv.errors.mtry[j,n]=sum(1-(yhat.rf==test$newsgroup))/nrow(test)
  }
}
cv.errors.mtry.mean=colMeans(cv.errors.mtry)
cv.errors.mtry.mean.best = min(cv.errors.mtry.mean)
plot(seq(10,100,10),cv.errors.mtry.mean,type='o',xlab='number of preditors',ylab='misclassfication cv error',main='5-fold cv of random forest with ntree=20')
cv.errors.mtry.mean.best = min(cv.errors.mtry.mean)
mtry.best = which(cv.errors.mtry.mean==cv.errors.mtry.mean.best)*10
print(c(mtry.best,cv.errors.mtry.mean.best))

doc.rf.best=randomForest(doc.table,doc.data$newsgroup,data=doc.data,mtry=90,ntree=20,importance=TRUE)
doc.rf.best.predict=predict(doc.rf.best,doc.data)
doc.rf.best.error=sum(1-(doc.rf.best.predict==doc.data$newsgroup))/nrow(doc.data)
doc.rf.best.error
table(predicted_newsgroup=doc.rf.best.predict,newsgroup=doc.data$newsgroup)  #confusion matrix

importance=doc.rf.best$importance
importance_sorted=importance[order(importance[,6],decreasing=TRUE),]
importance_sorted[c(1:10),]

# boosting
library(gbm)
cv.errors.boost=matrix(0,nrow=5,ncol=10)
cv.errors.boost.mean=matrix(0,nrow=5,ncol=10)
for (dep in 1:5){
  for (j in 1:k){
      train=as.numeric(rownames(doc.data[folds!=j,]))
      test=doc.data[folds==j,]
      doc.boost=gbm(newsgroup~.,data=doc.data[train,],distribution="multinomial",n.trees=2000, interaction.depth=dep)
      for (n in 1:10){
        doc.boost.predict=predict(doc.boost ,newdata=test,n.trees=n*200,type='response')
        yhat.boost=apply(doc.boost.predict, 1, which.max)
        cv.errors.boost[j,n]=sum(1-(yhat.boost==test$newsgroup))/nrow(test)
      }
  }
  cv.errors.boost.mean[dep,]=colMeans(cv.errors.boost)
}
cv.errors.boost.mean
matplot(seq(200,2000,200),t(cv.errors.boost.mean), type = c("b"),pch=1,xlab='number of trees',ylab='misclassfication error',main='boosting tree with different depth',ylim=c(min(cv.errors.boost.mean),max(cv.errors.boost.mean)*1.05))
legend('topright', legend = 1:5, col=1:5, pch=16,cex=0.6,lty=1,title='depth',h=1)
boost.best = c(which(apply(cv.errors.boost.mean,1,min)==min(cv.errors.boost.mean)),which(apply(cv.errors.boost.mean,2,min)==min(cv.errors.boost.mean))*200)
print(c(boost.best,min(cv.errors.boost.mean)))

doc.boost.best=gbm(newsgroup~.,data=doc.data,distribution="multinomial",n.trees=600, interaction.depth=4)
doc.boost.predict=predict(doc.boost.best ,newdata=doc.data,n.trees=600,type='response')
yhat.boost=apply(doc.boost.predict, 1, which.max)
doc.boost.error=sum(1-(yhat.boost==doc.data$newsgroup))/nrow(doc.data)
table(predicted_newsgroup=yhat.boost,newsgroup=doc.data$newsgroup)

#LDA and QDA
library(MASS)
cv.errors.lda=rep(0,5)
cv.errors.qda=rep(0,5)
word_not_use = c()
for (j in 1:5){
  train=as.numeric(rownames(doc.data[folds!=j,]))
  test=doc.data[folds==j,]
  lda.fit=lda(newsgroup~.,data=doc.data ,subset=train)
  yhat.lda=predict(lda.fit,test)
  cv.errors.lda[j]=sum(1-(yhat.lda$class==test$newsgroup))/nrow(test)
  # check keywords which should not be used in qda
  for (g in 1:4){
    temp=doc.data[train,][doc.data[train,]$newsgroup==g,]
    word_not_use=c(word_not_use,names(which(colSums(temp[,1:100])==0)))
  }
}

for (j in 1:5){
  train=as.numeric(rownames(doc.data[folds!=j,]))
  test=doc.data[folds==j,]
  qda.fit=qda(newsgroup~.-bmw-hockey-jews-lunar-nhl-puck-vitamin-aids-bible-dos-israel-orbit-patients-scsi-shuttle-honda-jesus-mars-dealer-season,data=doc.data ,subset=train)
  yhat.qda=predict(qda.fit,test)
  cv.errors.qda[j]=sum(1-(yhat.qda$class==test$newsgroup))/nrow(test)
}
cv.errors.lda.mean=mean(cv.errors.lda)
lda.fit=lda(newsgroup~.,data=doc.data)
yhat.lda=predict(lda.fit,doc.data)
table(predicted_newsgroup=yhat.lda$class,newsgroup=doc.data$newsgroup)

cv.errors.qda.mean=mean(cv.errors.qda)
qda.fit=qda(newsgroup~.-bmw-hockey-jews-lunar-nhl-puck-vitamin-aids-bible-dos-israel-orbit-patients-scsi-shuttle-honda-jesus-mars-dealer-season,data=doc.data)
yhat.qda=predict(qda.fit,doc.data)
table(predicted_newsgroup=yhat.qda$class,newsgroup=doc.data$newsgroup)

#Part 2
#PCA
doc.pc=prcomp(doc.table, scale=TRUE)
set.seed(1)
doc.km4=kmeans(doc.pc$x[,1:4],4,nstart=20)
require(plyr)
doc.km4$cluster=mapvalues(doc.km4$cluster,from=c(1,2,3,4),to=c(2,1,4,3))
table(doc.km4$cluster,doc.group[,1])
doc.km5=kmeans(doc.pc$x[,1:5],4,nstart=20)
doc.km5$cluster=mapvalues(doc.km5$cluster,from=c(1,2,3,4),to=c(4,3,2,1))
table(doc.km5$cluster,doc.group[,1])

#Part 3
mnist.train=read.csv("train_resized.csv",header=TRUE)
mnist.test=read.csv("test_resized.csv",header=TRUE)
mnist.train$label=as.factor(mnist.train$label)
mnist.test$label=as.factor(mnist.test$label)
mnist.train.i36=mnist.train[mnist.train[,1]==3 | mnist.train[,1]==6,]
mnist.test.i36=mnist.test[mnist.test[,1]==3 | mnist.test[,1]==6,]
library(e1071)
set.seed(1)
tc=tune.control(cross = 5)
#linear kernel
start_time=Sys.time()
tune.out=tune(svm,label~.,data=mnist.train.i36,kernel="linear",tunecontrol=tc,ranges=list(cost=c(0.001,0.01,0.1,1,5,10,100)))
end_time=Sys.time()
end_time-start_time
summary(tune.out)
bestmod=tune.out$best.model
ypred=predict(bestmod ,mnist.test.i36)
table(predict=ypred, truth=mnist.test.i36$label)

#radial kernel
start_time=Sys.time()
tune.out_1=tune(svm,label~.,data=mnist.train.i36,kernel="radial",tunecontrol=tc,ranges=list(cost=c(1,10,100,1000),gamma=c(0.0001,0.001,0.01,0.1)))
end_time=Sys.time()
end_time-start_time
summary(tune.out_1)
bestmod_r1=tune.out_1$best.model
ypred=predict(bestmod_r1 ,mnist.test.i36)
table(predict=ypred, truth=mnist.test.i36$label)

#1258
mnist.train.i1258=mnist.train[mnist.train[,1]==1 | mnist.train[,1]==2 | mnist.train[,1]==5 | mnist.train[,1]==8,]
mnist.test.i1258=mnist.test[mnist.test[,1]==1 | mnist.test[,1]==2 | mnist.test[,1]==5 | mnist.test[,1]==8,]
start_time=Sys.time()
tune.out.1258=tune(svm,label~.,data=mnist.train.i1258,kernel="linear",tunecontrol=tc,ranges=list(cost=c(0.01,0.1,1,10,100)))
end_time=Sys.time()
end_time-start_time
summary(tune.out.1258)
bestmod_1258=tune.out.1258$best.model
ypred=predict(bestmod_1258 ,mnist.test.i1258)
table(predict=ypred, truth=mnist.test.i1258$label)

#all
start_time=Sys.time()
options(warn=-1)
tune.out.all=tune(svm,label~.,data=mnist.train,kernel="linear",tunecontrol=tc,ranges=list(cost=c(0.01,0.1,1,10,100)))
end_time=Sys.time()
end_time-start_time
summary(tune.out.all)
bestmod_all=tune.out.all$best.model
ypred=predict(bestmod_all ,mnist.test)
table(predict=ypred, truth=mnist.test$label)

#PART 4 
#MLP
library(dplyr)
library(keras)
library(tensorflow)
library(yardstick)

data_train=read.csv("train_resized.csv", header=TRUE)
data_test=read.csv("test_resized.csv", header=TRUE)

train_x=as.matrix(data_train[,-1])
test_x=as.matrix(data_test[,-1])
train_y=to_categorical(data_train$label, num_classes = 10)
test_y=data_test$label

tf$random$set_seed(1)
start_time=Sys.time()
model=keras_model_sequential(name = "MLP_MNIST")
layer_dense(model,units = 512, activation = "relu", input_shape = ncol(train_x), name = "Hidden_1")
layer_dense(model,units = 256, activation = "relu", name = "Hidden_2")
layer_dense(model,units = 128, activation = "relu", name = "Hidden_3")
layer_dense(model,units = 64, activation = "relu", name = "Hidden_4")
layer_dense(model,units = 32, activation = "relu", name = "Hidden_5")
layer_dense(model,units = 16, activation = "relu", name = "Hidden_6")
layer_dense(model,units = 10, activation = "softmax", name = "Out")
model
compile(model, loss="categorical_crossentropy", optimizer=optimizer_adam(lr=0.001), metrics="accuracy")
history=fit(model, x=train_x, y=train_y, epochs=15, batch_size=32, validation_split=0.2, verbose=1)
end_time=Sys.time()
end_time-start_time
plot(history)
pred_test=predict_classes(model, test_x)
error_mlp=1-accuracy_vec(truth = as.factor(data_test$label), estimate = as.factor(pred_test))
error_mlp
table(prediction = as.factor(pred_test), truth = as.factor(data_test$label))

#CNN
train_x_cnn=array_reshape(train_x, dim=c(nrow(train_x), 12, 12, 1))
test_x_cnn=array_reshape(test_x, dim=c(nrow(test_x), 12, 12, 1))

tf$random$set_seed(1)
model1=keras_model_sequential(name = "CNN_Mnist")
layer_conv_2d(model1,filters=32, kernel_size=c(2,2), padding="same", activation="relu", input_shape=c(12,12,1))
layer_max_pooling_2d(model1, pool_size=c(2,2))
layer_conv_2d(model1, filters=32, kernel_size=c(2,2), padding="same", activation="relu", input_shape=c(12,12,1))
layer_max_pooling_2d(model1, pool_size=c(2,2))  
layer_conv_2d(model1, filters=32, kernel_size=c(2,2), padding="same", activation="relu", input_shape=c(12,12,1))
layer_max_pooling_2d(model1, pool_size=c(2,2))    
layer_flatten(model1) 
layer_dense(model1,units=16, activation="relu")
layer_dense(model1,units=10, activation="softmax", name="Output")
model1
compile(model1, loss="categorical_crossentropy", optimizer=optimizer_adam(lr=0.001), metrics="accuracy")
history=fit(model1, x=train_x_cnn, y=train_y, epochs=15, batch_size=32, validation_split=0.2, verbose=1)
plot(history)
pred_test=predict_classes(model1, test_x_cnn)
error_cnn=1-accuracy_vec(truth = as.factor(data_test$label), estimate = as.factor(pred_test))
error_cnn
table(prediction = as.factor(pred_test), truth = as.factor(data_test$label))
