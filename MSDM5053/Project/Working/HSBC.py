import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numpy import exp
from arch import arch_model
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
from datetime import datetime
import warnings
import yfinance as yf

warnings.filterwarnings("ignore")

def LB(ts, lags=[12], nparams=0):
    result = sm.stats.diagnostic.acorr_ljungbox(ts, lags=lags, return_df=True)
    result['Adjusted'] = 1 - stats.chi2.cdf(result['lb_stat'], result.index-nparams)
    print(result)


def ADFT(ts, alpha=0.05, l=12, caption='The time series'):
    from statsmodels.tsa.stattools import adfuller
    result = adfuller(ts, maxlag=l)
    print('*************************')
    print('ADF Test on stationarity')
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    if result[1] <= alpha:
        print(f'{caption} is stationary.')
    else:
        print(f'{caption} is not stationary.')
    return 

def check_series(rt, alpha=0.05, caption=None):
    ADFT(rt)
    LB = sm.stats.diagnostic.acorr_ljungbox(rt, lags=[12,24], return_df=True)
    print('*************************')
    print(f'Ljung-Box Test on Serial Correlation - {caption}')
    print(LB)
    if (LB['lb_pvalue']>alpha).sum() > 0:
        print('H0 is not rejected.')
        print('There is no serial correlation.')
    else:
        print('H0 is rejected')
        print('There is serial correlation.')
    at = rt-np.mean(rt)
    print('*************************')
    print(f'Ljung-Box Test on ARCH effect - {caption}')
    LB2 = sm.stats.diagnostic.acorr_ljungbox(at**2, lags=[12,24], return_df=True)
    print(LB2)
    if (LB2['lb_pvalue']>alpha).sum() > 0:
        print('H0 is not rejected.')
        print('There is no ARCH effect.')
    else:
        print('H0 is rejected')
        print('There is ARCH effect.')
    return
    
# evaluate an ARMA model for a given order (p,q)
def arma_mse(X, p, q):
    size = int(len(X) * 0.7)
    train, test = X[:size], X[size:]
    rolling_train = list(train)
    predictions = []
    for i in range(len(test)):
        model = ARIMA(rolling_train, order=(p,0,q))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        rolling_train.append(test[i])
    return mean_squared_error(test, predictions)

def arma_grid_mse(X, max_p=5, max_q=5):
    best_pq = (0,0)
    best_mse = float('Inf')
    for i in range(max_p):
        for j in range(max_q):
            print(f'Calculating...p={i},q={j}')
            s = arma_mse(X, i, j)
            if s < best_mse:
                best_pq = (i,j)
                best_mse = s
    print(f'{best_pq} is the best combination')
    return best_pq
    
def arma_grid_aic(X, p0=0, mp=5, q0=0, mq=5):
    best_pq = (0,0)
    best_aic = float('Inf')
    for i in range(p0, mp):
        for j in range(q0, mq):
            print(f'Calculating...p={i},q={j}')
            model = ARIMA(X, order=(i,0,j))
            model_fit = model.fit()
            s = model_fit.aic
            if s < best_aic:
                best_pq = (i,j)
                best_aic = s
    print(f'{best_pq} is the best combination')
    return best_pq

# path = "C:\\Users\\aauser\\Downloads\\d-0005.HK.csv"
# path_m = "C:\\Users\\aauser\\Downloads\\m-0005.HK.csv"

# fig, ax = plt.subplots(3,figsize=(10,10))
# df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
# dfm = pd.read_csv(path_m, parse_dates=['Date']).set_index('Date')
df = yf.Ticker('0005.hk').history(period='max')
dfm = yf.Ticker('0005.hk').history(period='max', interval='1mo').dropna()
df = df[df.Volume!=0].dropna()
df['R_Open'] = df['Open'].pct_change()
df['R_High'] = df['High'].pct_change()
df['R_Low'] = df['Low'].pct_change()
df['R_Close'] = df['Close'].pct_change()
df['r_Open'] = np.log(df['R_Open']+1)
df['r_High'] = np.log(df['R_High']+1)
df['r_Low'] = np.log(df['R_Low']+1)
df['r_Close'] = np.log(df['R_Close']+1)

# check_series(df['r_Open'][1:], caption='r_Open')
# check_series(df['r_High'][1:], caption='r_High')
# check_series(df['r_Low'][1:], caption='r_Low')
# check_series(df['r_Close'][1:], caption='r_Close')

# best_high = arma_grid_aic(df['r_High'][1:], 7, 7)
# best_low = arma_grid_aic(df['r_Low'][1:], 7, 7)
# best_open = arma_grid_aic(df['r_Open'][1:], 7, 7)
best_close = arma_grid_aic(df['r_Close'][1:], 10, 10)

# best_high = (6,5)
# best_low = (0,2)

# c = 4000
# ph = []
# pl = []
# while c < len(df['r_High']):
#     # predicting High
#     dh = df['r_High'][1:c]
#     model_high = ARIMA(dh, order=(6,0,[3,4,5])).fit()
#     r_high_hat = model_high.forecast().iloc[0]
#     p_high_hat = df['Adj_High'][c-1]*exp(r_high_hat)
#     ph.append(p_high_hat)
#     # predicting Low
#     dl = df['r_Low'][1:c]
#     model_low = ARIMA(dl, order=(0,0,2)).fit()
#     r_low_hat = model_low.forecast().iloc[0]
#     p_low_hat = df['Adj_Low'][c-1]*exp(r_low_hat)
#     pl.append(p_low_hat)
#     c += 1

# df_sim = df.iloc[c:]
# df_sim['Predict_High(t-1)'] = ph
# df_sim['Predict_Low(t-1)'] = pl

# col = ['Adj_High', 'Adj_Low', 'Predict_High(t-1)', 'Predict_Low(t-1)']
# df_sim['Buy']=((df_sim['Predict_Low(t-1)']>=df_sim['Adj_Low'])).astype('int')
# df_sim['Sold']=((df_sim['Predict_High(t-1)']<=df_sim['Adj_High'])).astype('int')

# 1 - stats.chi2.cdf(1.888286, 3) # p-value adjustment for LB Test on ARMA residual


# Pt = df['Adj Close'].dropna()
# # Plotting Pt
# Pt.plot(title='Stock price over time', ax=ax[0])
# pt = np.log(Pt)
# Rt = Pt.pct_change().dropna()
# Rt.plot(title='Return over time', ax=ax[1])
# rt = pt.diff().dropna()
# rt.plot(title='r(t) over time', ax=ax[2])
# Plotting

# Unit Root Test for stationarity
# rt = df['r_Close']
# ADF(rt, 0.05)
# LB = sm.stats.diagnostic.acorr_ljungbox(rt, lags=[12,24], return_df=True)
# print('Ljung-Box Test on Serial Correlation')
# print(LB)
# at = rt-np.mean(rt)
# LB2 = sm.stats.diagnostic.acorr_ljungbox(at**2, lags=np.arange(10,31,5), return_df=True)
# print(LB2)

# # Ljung-Box Test for serial correlation
# LB = sm.stats.diagnostic.acorr_ljungbox(rt, lags=[12,24], return_df=True)
# print('Ljung-Box Test on Serial Correlation')
# print(LB)
# # sm.graphics.tsa.plot_acf(rt, zero=False)
# # sm.graphics.tsa.plot_acf(rt, zero=False, auto_ylims=True)
# # sm.graphics.tsa.plot_pacf(rt, zero=False, auto_ylims=True, method='ywm')

# # Ljung-Box Test for ARCH effect
# at = rt-np.mean(rt)
# LB2 = sm.stats.diagnostic.acorr_ljungbox(at**2, lags=np.arange(10,31,5), return_df=True)
# print(LB2)

# n_test = 0
# train, test = rt[:-n_test]*100, rt[-n_test:]*100

# train = df['r_High'].dropna()

# GARCH model
# model = arch_model(train, mean='Zero', vol='GARCH', p=1, q=1)
model = arch_model(train, vol='GARCH', p=1, q=1, dist='Normal')
model_fit = model.fit()
print(model_fit)
# yhat = model_fit.forecast(horizon=n_test, reindex=True)

# # # ARIMA model
# # model = ARIMA(r, order=(5,0,0))
# # model_fit = model.fit()
# # print(model_fit.summary())