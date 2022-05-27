def cal_pi(array,sum_i,lock):
    for j in array:  
        with lock:
            sum_i.value += f(j)

def f(x):
    return 4 / (1 + x**2)