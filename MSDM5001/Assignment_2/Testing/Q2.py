from multiprocessing import Process, Value, Lock
import numpy as np
import time

def cal_pi(array,sum_i,lock):
  for j in array:  
      with lock:
          sum_i.value += f(j)

def f(x):
    return 4 / (1 + x**2)

n_process = 4
sep = 100000
N = sep // n_process
data = np.linspace(0,1,sep)
p_data = [[] for i in range(n_process)]
for i in range(n_process-1):
    p_data[i] = data[i*N:(i+1)*N]
p_data[-1] = data[(n_process-1)*N:]

if __name__ == '__main__':
   x = Value('d',0)
   lock = Lock()
   processes = []
   for i in range(n_process):
       p = Process(target=cal_pi, args=(p_data[i],x,lock))
       processes.append(p)
   t0 = time.time()
   for p in processes:
       p.start()
   for p in processes:
       p.join()
   print(f'pi = {x.value/sep}')
   print(f'Process time with {n_process} processes is {time.time()-t0} seconds.')
