hsbc=read.csv("hsbc.csv", header=T)
r_open = tail(hsbc$r_Open, -1)
r_high = tail(hsbc$r_high, -1)
r_low = tail(hsbc$r_low, -1)
r_close = tail(hsbc$r_close, -1)

Box.test(r_open, lag=12, type="Ljung")
Box.test(r_high, lag=12, type="Ljung")
Box.test(r_low, lag=12, type="Ljung")
Box.test(r_close, lag=12, type="Ljung")


ra_ho = hsbc$High/hsbc$Open
ra_lo = hsbc$Low/hsbc$Open
ra_co = hsbc$Close/hsbc$Open

library(fUnitRoots)
adfTest(ra_ho, lag=12, type=c('c'))
adfTest(ra_lo, lag=12, type=c('c'))
adfTest(ra_co, lag=12, type=c('c'))

acf(r_open, main='ACF of return Open', lag=24)
pacf(r_open, main='PACF of return Open', lag=24)
(m_ropen=arima(r_open, order=c(5,0,0)))
acf(m_ropen$residuals, lag=24)
bt=Box.test(m_ropen$residuals, lag=12, type="Ljung")
(pv=1-pchisq(bt$statistic,7))

acf(r_close, main='ACF of return Close', lag=24)
pacf(r_close, main='PACF of return Close', lag=24)
Box.test(r_close, lag=12, type="Ljung")
Box.test((r_close-mean(r_close))^2, lag=10, type="Ljung")

library(fGarch)
model_close=garchFit(~garch(1,1), data=r_close, trace=F)
summary(model_close)

Box.test(ra_co, lag=12, type="Ljung")
acf(ra_co, main='ACF of ration Close/Open', lag=24)
pacf(ra_co, main='PACF of ration Close/Open', lag=24)
(m_raco=arima(ra_co, order=c(11,0,0), ))
pacf(m_raco$residuals, lag=24)
bt=Box.test(m_raco$residuals, lag=12, type="Ljung")
(pv=1-pchisq(bt$statistic,7))
