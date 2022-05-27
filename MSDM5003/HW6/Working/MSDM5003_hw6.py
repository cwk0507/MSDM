from random import random, sample, seed
import numpy as np
import matplotlib.pyplot as plt

seed(5003)

ex = -0.7 # Exponent of Saving Factor
N = 1500 # Number of Agents
n = 5000 # Number of iterations
nsample = 6 # Number of samples

wealth = np.array([])
for i in range(nsample):
    savings = np.random.random(N)**(1/(1+ex)) # savings
    m = np.ones(N) # wealth
    for j in range(n):
        pair = sample(range(N), N)
        for k in range(0, N, 2):
            p1 = pair[k] # player 1
            p2 = pair[k+1] # player 2
            trade = random()
            m[p1], m[p2] = savings[p1]*m[p1]+trade*((1-savings[p1])*m[p1]+(1-savings[p2])*m[p2]), savings[p2]*m[p2]+(1-trade)*((1-savings[p1])*m[p1]+(1-savings[p2])*m[p2])
    wealth = np.concatenate((wealth, m))

count, bins, bar = plt.hist(np.log10(wealth), density=1, cumulative=-1, histtype='step', log=True, bins=1000)
plt.title('Histogram of log(Wealth)')
plt.xlabel('log(wealth)')
plt.ylabel('Tail distribution of wealth')

bins = np.array([(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)])

from scipy.optimize import curve_fit

def linear_function(x, m, b):
    return m*x+b

m = 1
b = 0
x1 = 600
x2 = 850
m_fit, b_fit = curve_fit(linear_function, bins[x1:x2], np.log10(count[x1:x2]), p0=[m, b])[0]

# Calculate R^2
residuals = np.log10(count[x1:x2]) - linear_function(bins[x1:x2], m_fit, b_fit)
ss_residuals = np.sum(residuals**2)
ss_total = np.sum((np.log10(count[x1:x2])-np.mean(np.log10(count[x1:x2])))**2)
r_squared = 1 - (ss_residuals / ss_total)


plt.figure()
plt.plot(bins[x1:x2], np.log10(count[x1:x2]), 'r-', label='From Simulation')
plt.plot(bins[x1:x2], linear_function(bins[x1:x2], m_fit, b_fit), 'b--', label='Fitted line')
plt.xlabel('log Wealth (log(w))')
plt.ylabel('log Density of Agents (log(d))')
plt.legend()
plt.title('Tail Distribution of Asset Exchange Model')
plt.text(1.1,-1.7,'log(d)=\u03B1log(w)', fontsize=15)
plt.text(1.1,-1.8,f'\u03B1 = {m_fit:.3f}', fontsize=15)
plt.text(1.1,-1.9,f'R^2 = {r_squared:.3f}', fontsize=15)
plt.show()

