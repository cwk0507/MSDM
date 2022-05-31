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

def LB(ts, lags=[12], g=0):
    result = sm.stats.diagnostic.acorr_ljungbox(ts, lags=lags, model_df=g, return_df=True)
    return result.iloc[-1]['lb_pvalue']

def LBdf(ts, lags=[12], g=0):
    result = sm.stats.diagnostic.acorr_ljungbox(ts, lags=lags, model_df=g, return_df=True)
    return result

def ACF(ts, lags=24, title='ACF'):
    sm.graphics.tsa.plot_acf(ts, auto_ylims=True, lags=lags, title=title)
    
def PACF(ts, lags=24, title='PACF'):
    sm.graphics.tsa.plot_pacf(ts, auto_ylims=True, lags=lags, title=title)

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
    
def arma_grid_aic(X, mp=5, d=0, mq=5, caption='rt'):
    target = []
    for k in range(d+1):
        for i in range(mp):
            for j in range(mq):
                print(f'Calculating...p={i},d={k},q={j}')
                model = ARIMA(X, order=(i,k,j))
                model_fit = model.fit()
                QM = LB(model_fit.resid, g=i+j)
                if QM > 0.05:
                    target.append([(i,k,j), model_fit.aic, model_fit.bic])
    tdf = pd.DataFrame(target, columns=['order','aic','bic']).sort_values(by='aic')
    try:
        print(f'{tdf.order.iloc[0]} is the best combination for {caption}')
        return tdf, tdf.order.iloc[0]
    except:
        print('Cannot find appropriate model')
        return tdf, 'N/A'

df = yf.Ticker('0005.hk').history(period='max')
# dfw = yf.Ticker('0005.hk').history(period='max', interval='1wk').dropna()
# dfm = yf.Ticker('0005.hk').history(period='max', interval='1mo').dropna()
df = df[df.Volume!=0].dropna()
# df['R_Open'] = df['Open'].pct_change()
df['R_High'] = df['High'].pct_change()
df['R_Low'] = df['Low'].pct_change()
df['R_Close'] = df['Close'].pct_change()
# df['r_Open'] = np.log(df['R_Open']+1)
df['r_High'] = np.log(df['R_High']+1)
df['r_Low'] = np.log(df['R_Low']+1)
df['r_Close'] = np.log(df['R_Close']+1)

n = len(df)
c = df.index.get_loc('2016').start

train_high, test_high = df['r_High'][1:c], df['r_High'][c:]
order_high = (6,0,0)
mh = ARIMA(train_high, order=order_high).fit()
print(mh.summary())
print(LB(mh.resid, [10], g=sum(order_high)))
print(LB(mh.resid**2, [10], g=sum(order_high)))
at_h = mh.resid
mh_garch = arch_model(at_h, mean='Zero', vol='GARCH', p=1, q=1, dist='Normal').fit()
print(mh_garch.summary())
print(LB(mh_garch.std_resid, [20]))
print(LB(mh_garch.std_resid**2, [20]))


train_low, test_low = df['r_Low'][1:c], df['r_Low'][c:]
order_low = (2,0,0)
ml = ARIMA(train_low, order=order_low).fit()
print(ml.summary())
print(LB(ml.resid, [10], g=sum(order_low)))
print(LB(ml.resid**2, [10], g=sum(order_low)))
at_l = ml.resid
ml_garch = arch_model(at_l, mean='Zero', vol='GARCH', p=1, q=1, dist='Normal').fit()
print(ml_garch.summary())
print(LB(ml_garch.std_resid, [20]))
print(LB(ml_garch.std_resid**2, [20]))


train_close, test_close = df['r_Close'][1:c], df['r_Close'][c:]
order_close = ([1,4,9,11],0,0)
mc = ARIMA(train_close, order=order_close).fit()
print(mc.summary())
print(LB(mc.resid, [10], g=4))
print(LB(mc.resid**2, [10], g=4))
at_c = mc.resid
mc_garch = arch_model(at_c, mean='Zero', vol='GARCH', p=1, q=1, dist='Normal').fit()
print(mc_garch.summary())
print(LB(mc_garch.std_resid, [20]))
print(LB(mc_garch.std_resid**2, [20]))

rh = []
rh_se = []
rl = []
rl_se = []
rc = []
rc_se = []
c1 = c
while c1 < n:
    # predicting High
    print(f'Forecasting {df.index[c1]}')
    dh = df['r_High'][1:c1]
    model_high = ARIMA(dh, order=order_high).fit()
    rh.append(model_high.get_forecast(1).predicted_mean.iloc[0])
    model_high_garch = arch_model(model_high.resid, mean='Zero', vol='GARCH', p=1, q=1, dist='Normal').fit(update_freq=0, disp='off')
    rh_se.append(np.sqrt(model_high_garch.forecast().variance['h.1'].iloc[-1]))

    # predicting Low
    dl = df['r_Low'][1:c1]
    model_low = ARIMA(dl, order=order_low).fit()
    rl.append(model_low.get_forecast(1).predicted_mean.iloc[0])
    model_low_garch = arch_model(model_low.resid, mean='Zero', vol='GARCH', p=1, q=1, dist='Normal').fit(update_freq=0, disp='off')
    rl_se.append(np.sqrt(model_low_garch.forecast().variance['h.1'].iloc[-1]))

    # predicting Close
    dc = df['r_Close'][1:c1]
    model_close = ARIMA(dc, order=order_close).fit()
    rc.append(model_close.get_forecast(1).predicted_mean.iloc[0])
    model_close_garch = arch_model(model_close.resid, mean='Zero', vol='GARCH', p=1, q=1, dist='Normal').fit(update_freq=0, disp='off')
    rc_se.append(np.sqrt(model_close_garch.forecast().variance['h.1'].iloc[-1]))

    c1 += 1

df_sim = df[df.index.get_loc('2016').start:]
df_sim['rhat_High'] = rh
df_sim['rhat_High_se'] = rh_se
df_sim['rhat_Low'] = rl
df_sim['rhat_Low_se'] = rl_se
df_sim['rhat_Close'] = rc
df_sim['rhat_Close_se'] = rc_se
df_sim = pd.concat([df_sim, df.loc['2015-12-31'].to_frame().T]).sort_index()
df_sim['Phat_High'] = df_sim['High'].shift(1)*np.exp(df_sim['rhat_High'])
df_sim['Phat_Low'] = df_sim['Low'].shift(1)*np.exp(df_sim['rhat_Low'])
df_sim['Phat_Close'] = df_sim['Close'].shift(1)*np.exp(df_sim['rhat_Close'])

df_sim['Phat_High_upper'] = df_sim['High'].shift(1)*np.exp(df_sim['rhat_High']+1.96*df_sim['rhat_High_se'])
df_sim['Phat_High_lower'] = df_sim['High'].shift(1)*np.exp(df_sim['rhat_High']-1.96*df_sim['rhat_High_se'])
df_sim['Phat_Low_upper'] = df_sim['Low'].shift(1)*np.exp(df_sim['rhat_Low']+1.96*df_sim['rhat_Low_se'])
df_sim['Phat_Low_lower'] = df_sim['Low'].shift(1)*np.exp(df_sim['rhat_Low']-1.96*df_sim['rhat_Low_se'])
df_sim['Phat_Close_upper'] = df_sim['Close'].shift(1)*np.exp(df_sim['rhat_Close']+1.96*df_sim['rhat_Close_se'])
df_sim['Phat_Close_lower'] = df_sim['Close'].shift(1)*np.exp(df_sim['rhat_Close']-1.96*df_sim['rhat_Close_se'])

d = df_sim[1:]
P_High_percent = sum((d.High<=d.Phat_High_upper) & (d.High>=d.Phat_High_lower))/len(d)*100
P_Low_percent = sum((d.Low<=d.Phat_Low_upper) & (d.Low>=d.Phat_Low_lower))/len(d)*100
P_Close_percent = sum((d.Close<=d.Phat_Close_upper) & (d.Close>=d.Phat_Close_lower))/len(d)*100

# Strategy A
sa = df_sim[['High','Low','Close','r_Close','rhat_Close']][1:]
sa['Order'] = ((sa['rhat_Close']>0).astype(int)-(sa['rhat_Close']<0).astype(int)).shift(1)
sa['Order_Price']=sa['Close'].shift(1)
sa['Order_Success']=0
bdf = sa[sa.Order==1]
sa.loc[sa.Order==1, 'Order_Success'] = (bdf['Order_Price']>=bdf['Low']).astype(int)
sdf = sa[sa.Order==-1]
sa.loc[sa.Order==-1, 'Order_Success'] = (sdf['Order_Price']<=sdf['High']).astype(int)
sa['Trade_Price']=0
sa.loc[sa.Order_Success==1, 'Trade_Price'] = sa.loc[sa.Order_Success==1, ['High','Order_Price']].min(axis=1)
sa['Transaction'] = (-sa.Order*sa.Trade_Price).fillna(0)
sa['Cash'] = sa.Transaction.cumsum().fillna(0)
sa['Position'] = (sa.Order*sa.Order_Success).cumsum().fillna(0)
sa['Balance'] = sa.Cash+sa.Position*sa.Close

# fig, ax = plt.subplots()
# ax3 = ax.twinx()
# sa.Close.plot(ax=ax, style='b-')
# sa.Balance.plot(ax=ax3, style='r-')


# # Plotting Pt
# Pt.plot(title='Stock price over time', ax=ax[0])
# pt = np.log(Pt)
# Rt = Pt.pct_change().dropna()
# Rt.plot(title='Return over time', ax=ax[1])
# rt = pt.diff().dropna()
# rt.plot(title='r(t) over time', ax=ax[2])
# Plotting

# ra_HO = df['High']/df['Open']
# ra_LO = df['Low']/df['Open']
# ra_CO = df['Close']/df['Open']

# check_series(ra_HO[c1:c], caption='Ratio High/Open')
# check_series(ra_LO[c1:c], caption='Ratio Low/Open')
# check_series(ra_CO[c1:c], caption='Ratio Close/Open')

# ACF(ra_HO.iloc[:c])
# PACF(ra_HO.iloc[:c])
# ACF(ra_HO.diff().dropna().iloc[:c])
# PACF(ra_HO.diff().dropna().iloc[:c])
# ACF(ra_LO.iloc[:c])
# PACF(ra_LO.iloc[:c])
# ACF(ra_LO.diff().dropna().iloc[:c])
# PACF(ra_LO.diff().dropna().iloc[:c])
# ACF(ra_CO.iloc[:c])
# PACF(ra_CO.iloc[:c])

# model_rho = ARIMA(ra_HO, order=(0,1,1)).fit()
# # model_rho = ARIMA(ra_HO, order=(1,1,3)).fit()
# # LB(model_rho.resid, g=4)
# LB(model_rho.resid, g=1)
# LB(model_rho.resid**2)
# model_rlo = ARIMA(ra_LO, order=([1,3,4],1,1)).fit()
# LB(model_rlo.resid, g=4)
# LB(model_rlo.resid**2)

# model_rco = ARIMA(ra_CO[:c], order=([1,4,5,9,11],0,0)).fit()
# LB(model_rco.resid, g=5)
# LB(model_rco.resid**2)

# check_series(df['r_Open'][1:], caption='r_Open')
# check_series(df['r_High'][1:], caption='r_High')
# check_series(df['r_Low'][1:], caption='r_Low')
# check_series(df['r_Close'][1:], caption='r_Close')

# best_high = arma_grid_aic(df['r_High'][1:c], 7, 7, caption='r_High')
# best_low = arma_grid_aic(df['r_Low'][1:c], 7, 7, caption='r_Low')
# best_open = arma_grid_aic(df['r_Open'][1:c], 7, 7, caption='r_Open')
# best_close = arma_grid_aic(df['r_Close'][1:c], 7, 7, caption='r_Close')

# model_BH = ARIMA(df['r_High'][1:c], order=best_high).fit()
# LB(model_BH.resid, [15], g=sum(best_high))
# LB(model_BH.resid**2, [15], g=sum(best_high))
# model_BL = ARIMA(df['r_Low'][1:c], order=best_low).fit()
# LB(model_BL.resid, [15], g=sum(best_low))
# LB(model_BL.resid**2, [15], g=sum(best_low))
# model_BO = ARIMA(df['r_Open'][1:c], order=best_open).fit()
# LB(model_BO.resid, [15], g=sum(best_open))
# LB(model_BO.resid**2, [15], g=sum(best_open))
# model_BC = ARIMA(df['r_Close'][1:c], order=best_close).fit()
# LB(model_BC.resid, [15], g=sum(best_close))

# rh = []
# rh_se = []
# rl = []
# rl_se = []
# c1 = c
# while c1 < n:
#     # predicting High
#     print(f'Forecasing {df.index[c]}')
#     dh = ra_HO[1:c1]
#     model_high = ARIMA(dh, order=(0,1,1)).fit()
#     rh.append(model_high.get_forecast(1).predicted_mean.iloc[0])
#     rh_se.append(model_high.get_forecast(1).se_mean.iloc[0])
#     # p_high_hat = df['High'][c-1]*exp(r_high_hat)
#     # ph.append(p_high_hat)
#     # predicting Low
#     dl = ra_LO[1:c1]
#     model_low = ARIMA(dl, order=(0,1,1)).fit()
#     rl.append(model_low.get_forecast(1).predicted_mean.iloc[0])
#     rl_se.append(model_low.get_forecast(1).se_mean.iloc[0])
#     # p_low_hat = df['Low'][c-1]*exp(r_low_hat)
#     # pl.append(p_low_hat)
#     c += 1

# df_sim = df[df.index.get_loc('2016').start:]
# df_sim['Predicted_raHO'] = rh
# df_sim['Predicted_raHO_se'] = rh_se
# df_sim['Predicted_raLO'] = rl
# df_sim['Predicted_raLO_se'] = rl_se
# df_sim = pd.concat([df_sim, df.loc['2015-12-31'].to_frame().T]).sort_index()