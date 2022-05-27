import numpy as np
import matplotlib.pyplot as plt

def white_noise(gamma, kBT, N):
    return np.random.normal(0, np.sqrt(2*gamma*kBT), N)

t1 = 1000
N = 1000000  # number of steps
dt = t1/N  # step size
gamma = 1
kBT = 1
interval = 1000

noise = white_noise(gamma, kBT, N)

xa = np.array([0]*(N+1), dtype='float32')
t = np.linspace(0, N*dt, N+1)
for idx, _ in enumerate(t):
    if idx > 0:
        xa[idx] = xa[idx-1]+dt*noise[idx-1]  # Euler's method

msd = np.array([0]*interval, dtype='float32')  # container of msd
x0 = xa[0]
for i in range(int(N/interval)):
    msd = msd + (xa[i*interval+1:(i+1)*interval+1]-x0)**2
    x0 = xa[(i+1)*interval]

msd = msd / (N/interval)
plt.figure()
plt.plot(xa)
# plt.plot(t[1:interval+1],msd)

diffusivity = msd / (2*t[1:interval+1])
    
        