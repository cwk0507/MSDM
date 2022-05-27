import numpy as np
from numpy.random import randint
from random import seed, choice
import matplotlib.pyplot as plt

seed(5003)

m = 3 # memory size
states = 2**m # number of states
ns = 2 # number of stragy of each agent
N = 101 # number of agents
iterations = 1000 # number of interations

sequence = 2*randint(0, 2, N*ns*states) - 1
darray = sequence.reshape((N, ns, states)) # array of strategies
varray = np.zeros((N, ns)) # array of virtual scores
rarray = np.zeros((N, iterations+1)) # array of real scores
active = 0 # initialize input state at t=0
buyers = [] # array of number of buyers

def update_scores(d, v, r, jchoice, actions, active):
    score = sum(actions)/N
    for i in range(N):
        for j in range(ns):
            v[i,j] -= d[i,j,active]*score
        r[i] += v[i,jchoice[i]]
    return v, r
        
def update_state(current, new):
    return (2 * current + new) % states
            
for k in range(1, iterations+1):
    actions = [] # array of agents' action in each iteration
    jchoice = [] # array of strategy chosen in each iteration
    for i in range(N):
        max_v = varray[i,0]
        max_vi = [0]
        for j in range(1,ns):
            if varray[i,j] > max_v:
                max_v = varray[i,j]
                max_vi = [j]
            elif varray[i,j] == max_v:
                max_vi.append(j)
        if len(max_vi)==1:
            jchoice.append(max_vi[0])
        else:
            jchoice.append(choice(max_vi))
        actions.append(darray[i,jchoice[-1],active])
    buyers.append(sum([n==1 for n in actions]))
    if buyers[-1] > (N-1)/2:
        winning = 1
    else:
        winning = 0
    varray, rarray[:,k] = update_scores(darray, varray, rarray[:,k-1], jchoice, actions, active)
    active = update_state(active, winning)

best3 = rarray[:,-1].argsort()[-3:][::-1]
worst3 = rarray[:,-1].argsort()[:3]
plt.plot(rarray[best3[0],:], c='r', label='Best 3 agent')
plt.plot(rarray[best3[1],:], c='r')
plt.plot(rarray[best3[2],:], c='r')
plt.plot(rarray[worst3[0],:], c='g', label='Worst 3 agents')
plt.plot(rarray[worst3[1],:], c='g')
plt.plot(rarray[worst3[2],:], c='g')
plt.legend()
plt.title('Real scores of the 3 best and 3 worst agents')
plt.xlabel('t')
plt.ylabel('Real scores')

