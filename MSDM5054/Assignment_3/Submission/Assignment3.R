########Q1(1)##########
trees=read.csv("trees.csv", header=TRUE,sep=",")
fit.1=lm(Volume~Girth,data=trees)
fit.2=lm(Volume~poly(Girth,2),data=trees)
fit.3=lm(Volume~poly(Girth,3),data=trees)
fit.4=lm(Volume~poly(Girth,4),data=trees)
summary(fit.1)$adj.r.squared
summary(fit.2)$adj.r.squared
summary(fit.3)$adj.r.squared
summary(fit.4)$adj.r.squared

girthlims=range(trees$Girth)
girth.grid=seq(from=girthlims[1],to=girthlims[2],by=0.1)
preds=predict(fit.2,newdata=list(Girth=girth.grid),se=TRUE)
se.bands=cbind(preds$fit+2*preds$se.fit,preds$fit-2*preds$se.fit)

plot(girth.grid,preds$fit,xlim=girthlims ,type="n",ylim=c(10,80),
     xlab='Diameter',ylab='Volume')
title('Degree 2 model of Volume by Diameter')
lines(girth.grid,preds$fit,lwd=2, col="red")
matlines(girth.grid,se.bands,lwd=1,col="blue",lty=3)

########Q1(2)##########
fit.2.logit=glm(I(Volume>30)~poly(Girth,2),data=trees,family=binomial)
preds.logit=predict(fit.2.logit,newdata=list(Girth=girth.grid),se=T)
probs=exp(preds.logit$fit)/(1+exp(preds.logit$fit))
se.bands.logit = cbind(preds.logit$fit+2*preds.logit$se.fit, preds.logit$fit-2*preds.logit$se.fit)
se.bands.probs = exp(se.bands.logit)/(1+exp(se.bands.logit))

plot(girth.grid,preds.logit$fit,xlim=girthlims ,type="n",ylim=c(0,1),
     xlab='Diameter',ylab='P(Volume>30)')
title('Degree 2 model of P(Volume>30) by Diameter')
lines(girth.grid,probs,lwd=2, col="red")
matlines(girth.grid,se.bands.probs,lwd=1,col="blue",lty=3)

########Q1(3)##########
library(splines)
fit.spline=lm(Volume~bs(Girth,df=2,knots=c(10,14,18)),data=trees)
preds.spline=predict(fit.spline,newdata=list(Girth=girth.grid),se=T)
plot(girth.grid,preds$fit,xlim=girthlims ,type="n",ylim=c(10,80),
     xlab='Diameter',ylab='Volume')
title('Volume by Diameter with regression spline')
lines(girth.grid,preds.spline$fit,lwd=2,col="red")
lines(girth.grid,preds.spline$fit+2*preds.spline$se ,lty="dashed",col="green")
lines(girth.grid,preds.spline$fit-2*preds.spline$se ,lty="dashed",col="green")

########Q1(4)##########
fit.smooth.cv=smooth.spline(trees$Girth,trees$Volume,cv=TRUE)
fit.smooth.cv$df
plot(fit.smooth.cv$x,fit.smooth.cv$y ,type="n",ylim=c(10,80),
     xlab='Diameter',ylab='Volume')
title ("Volume by Diameter with Smoothing Spline")
lines(fit.smooth.cv,col="green",lwd=2)

########Q1(5)##########
library(gam)
fit.gam=gam(Volume~s(Girth,4)+s(Height,5) ,data=trees)
par(mfrow=c(1,2))
plot(fit.gam, se=TRUE,col="red")

########Q2(1)##########
options(warn=-1)
library(tree)
audit_train=read.csv("audit_train.csv", header=TRUE,sep=",")
audit_test=read.csv("audit_test.csv", header=TRUE,sep=",")
audit_train$Risk=as.factor(audit_train$Risk)  
fit.tree=tree(Risk~.,audit_train)
summary(fit.tree)
par(mfrow=c(1,1))
plot(fit.tree)
text(fit.tree,pretty=0)
title("Risk Classification Tree")
preds=predict(fit.tree,audit_test,type="class")
table(preds,audit_test$Risk)

########Q2(2)##########
set.seed(1)
fit.tree.cv =cv.tree(fit.tree ,FUN=prune.misclass )
fit.tree.cv
plot(fit.tree.cv$size ,fit.tree.cv$dev ,type="b",xlab="Tree size",ylab="Error")
title("Training error vs tree size")
prune.tree=prune.misclass(fit.tree,best=5)
plot(prune.tree)
text(prune.tree,pretty=0)
title("Pruned Tree of size=5")
preds.prune=predict(prune.tree,audit_test,type="class")
table(preds.prune,audit_test$Risk)

########Q2(3)##########
library(randomForest)
fit.rf=randomForest(Risk~.,data=audit_train,ntree=25,mtry=13,importance =TRUE)
fit.rf

########Q2(4)##########
m=c(8,12,14,16,18)
errors=rep(0,length(m))
for (i in seq_along(m)){
  fit.rf=randomForest(Risk~.,data=audit_train,ntree=25,mtry=m[i],importance =TRUE)
  errors[i]=fit.rf.i$err.rate[25,1]
  }
m.min=m[which.min(errors)]
m.min
fit.rf.min=randomForest(Risk~.,data=audit_train,ntree=25,mtry=m.min,importance =TRUE)
preds.rf.min=predict(fit.rf.min,audit_test,type="class")
table(preds.rf.min,audit_test$Risk)
