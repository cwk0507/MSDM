# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 15:44:26 2020

@author: aauser
"""

from multiprocessing import Process, Value, Lock
import numpy as np
import time
import os

def cal_pi(array,sum_i,lock):
  for j in array:  
      with lock:
          sum_i.value += f(j)*width

def f(x):
    return 4 / (1 + x**2)

n_processes = os.cpu_count()
sep = 100000
N = sep // n_processes
width = 1 / sep
data = np.linspace(0,1,sep)
p_data = [[] for i in range(n_processes)]
for i in range(n_processes-1):
    p_data[i] = data[i*N:(i+1)*N]
p_data[-1] = data[(n_processes-1)*N:]

if __name__ == '__main__':
   x = Value('d',0)
   lock = Lock()
   processes = []

   for i in range(n_processes):
       p = Process(target=cal_pi, args=(p_data[i],x,lock))
       processes.append(p)
       
   t0 = time.time()
   
   for p in processes:
       p.start()
    
   for p in processes:
       p.join()

   print(f'pi = {x.value*width}')
   print(f'Processing time with {n_processes} processes is {time.time()-t0} seconds.')
