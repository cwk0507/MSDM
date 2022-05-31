defaultW=getOption("warn")
options(warn = -1)

#Q1
sbux=read.table("d-sbuxsp0106.txt", header=F)[,2]
lr_sbux=log(sbux+1)
Box.test(lr_sbux, lag=10, type="Ljung")
par(mfrow=c(1,2))
acf(lr_sbux, main='ACF of log returns of SBUX', lag=10)
pacf(lr_sbux, main='PACF of log returns of SBUX', lag=10)
par(mfrow=c(1,1))
(m=arima(lr_sbux, order=c(1,0,0), fixed=c(NA,0)))
bt=Box.test(m$residuals, lag=10, type="Ljung")
(pv=1-pchisq(bt$statistic,9))

Box.test(m$residuals^2, lag=10, type="Ljung")

library(fGarch)
modelq1=garchFit(~garch(1,1), data=lr_sbux, trace=F)
summary(modelq1)

#modelq11=garchFit(~arma(1,0)+garch(1,1), data=lr_sbux, trace=F)
#summary(modelq11)


#Q2
sp500=read.table("d-sbuxsp0106.txt",header=F)[,3]
lr_sp500=log(sp500+1)
Box.test(lr_sp500, lag=10, type="Ljung")

at=lr_sp500-mean(lr_sp500)
Box.test(at^2, lag=10, type="Ljung")

library(rugarch)
specq2=ugarchspec(variance.model=list(model="iGARCH"),mean.model=list(armaOrder=c(0,0)))
(modelq2=ugarchfit(spec=specq2,data=lr_sp500))
(foreq2=ugarchforecast(modelq2, n.ahead=4))

stresi=residuals(modelq2,standardize=T)
Box.test(stresi,10,type="Ljung")
Box.test(stresi^2,10,type="Ljung")


#Q3
specq3a=ugarchspec(variance.model=list(model="sGARCH", garchOrder=c(1,1)),
                   mean.model=list(armaOrder=c(0,0),include.mean=TRUE,archm=TRUE,archpow=2),
                   distribution.model="norm")
(modelq3a=ugarchfit(specq3a, data=lr_sbux))

specq3c=ugarchspec(variance.model=list(model="eGARCH", garchOrder=c(1,1)),
                   mean.model=list(armaOrder=c(0,0)),
                   distribution.model="norm")
(modelq3c=ugarchfit(specq3c, data=lr_sbux))

stresi=residuals(modelq3c,standardize=T)
Box.test(stresi,10,type="Ljung")
Box.test(stresi^2,10,type="Ljung")

#Q4
pg=read.table("m-pg5606.txt", header=F)[,2]
lr_pg=log(pg+1)
Box.test(lr_pg, lag=10, type="Ljung")

modelq4=garchFit(~garch(1,1), data=lr_pg, trace=F)
summary(modelq4)
predict(modelq4, 5)

#Q5
exuseu=read.table("d-exuseu.txt", header=F)[,4]
Rt=diff(exuseu)
rt=log(1+Rt)
Box.test(rt, lag=10, type="Ljung")

a5t=rt-mean(rt)
Box.test(a5t^2, lag=10, type="Ljung")

specq5=ugarchspec(variance.model=list(model="iGARCH"),mean.model=list(armaOrder=c(0,0)))
(modelq5=ugarchfit(spec=specq5,data=rt))

stresi=residuals(modelq5,standardize=T)
Box.test(stresi,10,type="Ljung")
Box.test(stresi^2,10,type="Ljung")

(foreq5=ugarchforecast(modelq5, n.ahead=4))

options(warn = defaultW)
