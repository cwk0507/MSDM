import numpy as np
import pandas as pd

file_path = "C:\\Users\\aauser\\Desktop\\MscDDM\\5003\\Wk7\\HSI.csv"
df = pd.read_csv(file_path).dropna()

ret = np.abs(np.log(df['Adj Close'].shift(-1)/df['Adj Close'])).dropna()
mu = ret.mean()
cs = (ret-mu).cumsum()

T = len(cs)
logtaumax = int((round(np.log10(T-7, ), 1) - round(np.log10(7, ), 1)) / 0.1) + 2
x = np.arange(logtaumax)*0.1 + round(np.log10(7, ), 1)

epsilons = []
taus = []
for i in x:
    tau = int(10 ** i)
    epsilon = 0
    xx = np.arange(tau)
    xx_mean = (tau-1)/2
    xx2_mean = (tau-1)*(2*tau-1)/6
    for j in range(T-tau+1):
        yy = cs.iloc[j:j+tau]
        yy_mean = yy.mean()
        yy2_mean = (yy**2).mean()
        xy_mean = np.dot(xx, yy) / tau
        epsilon += np.sqrt(yy2_mean - yy_mean**2 - (xy_mean - xx_mean * yy_mean)**2 / (xx2_mean - xx_mean**2))
    epsilons.append(epsilon/(T-tau+1))
    taus.append(tau)
    
import matplotlib.pyplot as plt
plt.plot(np.log10(taus),np.log10(epsilons))
plt.xlabel('log(\u03C4)')
plt.ylabel('log(F(\u03C4))')

from scipy.optimize import curve_fit

def linear_function(x, m, b):
    return m*x+b

m = 1
b = 0
m_fit, b_fit = curve_fit(linear_function, np.log10(taus)[10:20], np.log10(epsilons)[10:20], p0=[m, b])[0]

plt.plot(np.log10(taus), linear_function(np.log10(taus), m_fit, b_fit), 'b--', label='Fitted line')