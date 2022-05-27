from multiprocessing import Pool
import numpy as np
import time

def count_in(data):
    a = data[0]
    b = data[1]
    if a*a + b*b < 1:
        return 1
    return 0
    
if __name__ == '__main__':
    n=100000
    m=4
    xx = np.random.random(n)
    yy = np.random.random(n)
    data = [(xx[i],yy[i]) for i in range(len(xx))]
    p = Pool(m)
    t0 = time.time()
    result = p.map(count_in,data)
    p.close()
    p.join()
    print(f'pi = {sum(result)*4/n}')
    print(f'Process time using Pool with n={m}: {time.time()-t0} seconds.')
