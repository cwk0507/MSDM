import numpy as np
from numpy.random import randint
from random import seed, choice, sample
import matplotlib.pyplot as plt
import time

seed(5003)

def theta(xt, xM):
    return 1 if xt>xM else 0

def cross_over(p1, p2):
    pos = randint(1, len(p1))
    c1 = np.concatenate((p1[:pos],p2[pos:]))
    c2 = np.concatenate((p2[:pos],p1[pos:]))
    return c1, c2

def replacement_co(d, v, parent='Random', replace='Same'):
    d1 = d.copy()
    ns = len(v)
    if parent == 'Random':
        parents_id = sample(range(ns),2)
    elif parent == 'Best':
        h1 = v.max()
        h2 = np.sort(v)[::-1][1]
        if sum(v>=h2) == 2:
            parents_id = [i for i in range(len(v)) if v[i]>=h2]
        elif h1>h2:
            parents_id = [v.argmax()]
            parents_id.append(choice([i for i in range(len(v)) if (v[i]!=max(v)) & (v[i]==h2)]))
        else:
            parents_id = sample([i for i in range(len(v)) if v[i]==h2], 2)
    child1, child2 = cross_over(d1[parents_id[0],:], d1[parents_id[1],:])
    if replace == 'Same':
        replace_id = parents_id
    elif replace == 'Worst':
        l1 = v.min()
        l2 = np.sort(v)[1]
        if sum(v<=l2) == 2:
            replace_id = [i for i in range(len(v)) if v[i]<=l2]
        elif l1<l2:
            replace_id = [v.argmin()]
            replace_id.append(choice([i for i in range(len(v)) if (v[i]!=min(v)) & (v[i]==l2)]))
        else:
            replace_id = sample([i for i in range(len(v)) if v[i]==l2], 2)
    elif replace == 'Random':
        replace_id = sample(range(ns),2)
    d1[replace_id[0],:] = child1
    d1[replace_id[1],:] = child2
    v[replace_id[0]] = v[parents_id[0]]
    v[replace_id[1]] = v[parents_id[1]]
    return d1, v

def mutation(parent, mode='Tail'):
    pos = choice(range(len(parent)))
    if (mode == 'Tail') or (pos <= (len(parent)+1)/2):
        flipped = abs(parent[pos:]-1)
        child = np.concatenate((parent[:pos],flipped))
    else:
        flipped = abs(parent[:pos]-1)
        child = np.concatenate((flipped, parent[pos:]))
    return child

def replacement_m(d, v, n=2, mode='Tail'):
    d1 = d.copy()
    l = np.sort(v)[n-1]
    if sum(v<=l) == n:
        mutate_id = [i for i in range(len(v)) if v[i]<=l]
    else:
        mutate_id = [i for i in range(len(v)) if v[i]<l]
        k = len(mutate_id)
        smid = sample([i for i in range(len(v)) if v[i]==l], n-k)
        for m in smid:
            mutate_id.append(m)
    for mid in mutate_id:
        d1[mid,:] = mutation(d1[mid,:], mode)
        v[mid] = v.max()
    return d1, v

def crossover_mutation(b, w, mode='Long'):
    pos = choice(range(len(b)))
    if mode=='Long':
        if pos > (len(b)+1)/2:
            flipped = abs(w[pos:]-1)
            child = np.concatenate((b[:pos],flipped))
        else:
            flipped = abs(w[:pos]-1)
            child = np.concatenate((flipped,b[pos:]))
    else:
        if pos > (len(b)+1)/2:
            flipped = abs(w[:pos]-1)
            child = np.concatenate((flipped,b[pos:]))
        else:
            flipped = abs(w[pos:]-1)
            child = np.concatenate((b[:pos],flipped))            
    return child

def replacement_cm(d, v, n=2, mode='Long'):
    d1 = d.copy()
    l = np.sort(v)[n-1]
    bestid = v.argmax()
    if sum(v<=l) == n:
        mutate_id = [i for i in range(len(v)) if v[i]<=l]
    else:
        mutate_id = [i for i in range(len(v)) if v[i]<l]
        k = len(mutate_id)
        smid = sample([i for i in range(len(v)) if v[i]==l], n-k)
        for m in smid:
            mutate_id.append(m)
    for mid in mutate_id:
        d1[mid,:] = crossover_mutation(d1[bestid,:], d1[mid,:], mode=mode)
        v[mid] = v.max()
    return d1, v

def adaptive_minority(m=3, N=101, ns=2, iterations=1000, n=0.6, tau=10, p='Random', r='Same', form='co', mode='Long'):
    
    def update_scores(d, v, r, jchoice, actions, active, xtt, win):
        r1 = r.copy()
        Ux = ((1-theta(xtt,xM))*xtt+theta(xtt,xM)*(N-xtt))/xM
        for i in range(N):
            for j in range(ns):
                if d[i,j,active] == win:
                    v[i,j] += 1
            if d[i,jchoice[i],active] ==  win:
                r1[i] = r[i] + 1
        return v, r1, Ux
    
    def update_state(current, new):
        return (2 * current + new) % states
    
    # Initialization
    states = 2**m # number of states
    sequence = randint(0, 2, N*ns*states)
    darray = sequence.reshape((N, ns, states)) # array of strategies
    varray = np.zeros((N, ns)) # array of virtual scores
    rarray = np.zeros((N, iterations+1)) # array of real scores
    active = 0 # initialize input state at t=0
    states_array = [0]
    xt = [] # array of number of agents choosing action 1
    xM = int((N-1)/2) # max number of winning agents  per interation
    Uxt = []
    
    # Simulation
    for k in range(1, iterations+1):        
        if k % tau == 0:
            adapt = True
            rsorted = np.sort(rarray[:,k-1])[::-1]
            bound = rsorted[int(N*(1-n))-1]
        else:
            adapt = False
        actions = [] # array of agents' action in each iteration
        jchoice = [] # array of strategy chosen in each iteration
        for i in range(N):
            if adapt:
                if rarray[i,k-1]<=bound:
                    if form == 'co':
                        darray[i,:,:], varray[i,:] = replacement_co(darray[i,:,:], varray[i,:], parent=p, replace=r)
                    elif form == 'm':
                        darray[i,:,:], varray[i,:] = replacement_m(darray[i,:,:], varray[i,:], n=2, mode=mode)
                    elif form == 'cm':
                        darray[i,:,:], varray[i,:] = replacement_cm(darray[i,:,:], varray[i,:], n=2, mode=mode)
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
        xt.append(sum(np.array(actions)==1))
        if xt[-1] > xM:
            winning = 0
        else:
            winning = 1
        varray, rarray[:,k], Ux = update_scores(darray, varray, rarray[:,k-1], jchoice, actions, active, xt[-1], winning)
        active = update_state(active, winning)
        states_array.append(active)
        Uxt.append(Ux)
        
    return xt, rarray, Uxt, states_array

def amg_collective_sim(samples=30, plot_ind=False):
    m = 4
    N = 501
    ns = 10
    iterations = 10000
    n = 0.3
    tau = 40
    window = 50
    U = [[],[],[],[],[],[],[],[],[],[],[]]
    parent = ['Random', 'Best']
    children = ['Same', 'Random', 'Worst']
    mode_m = ['Tail','Long']
    mode_cm = ['Long','Short']
    c = 0
    # Crossover
    for p in parent:
        for r in children:
            for s in range(samples):
                st = time.time()
                xt, rarray, Uxt, states_array = adaptive_minority(m, N, ns, iterations, n, tau, p, r, form='co')
                print(f'Simulation Time: {(time.time()-st)/60:2f} mins')
                if plot_ind:
                    plt.figure(figsize=(15,15))
                    plt.plot(xt)
                    plt.ylim(0,N)
                    plt.xlim(0,iterations)
                    plt.xlabel('Time')
                    plt.ylabel('x_t')
                    plt.title(f'Number of agents choosing action 1 with genetic crossover: Parents:{p}, Offsprings:{r}')
                u=[np.mean(Uxt[i*window:(i+1)*window]) for i in range(int(iterations/window))]
                U[c].append(u)
            c += 1
    # Mutation 
    for mo in mode_m:
        for s in range(samples):
                st = time.time()
                xt, rarray, Uxt, states_array = adaptive_minority(m, N, ns, iterations, n, tau, form='m', mode=mo)
                print(f'Simulation Time: {(time.time()-st)/60:2f} mins')
                if plot_ind:
                    plt.figure(figsize=(15,15))
                    plt.plot(xt)
                    plt.ylim(0,N)
                    plt.xlim(0,iterations)
                    plt.xlabel('Time')
                    plt.ylabel('x_t')
                    plt.title(f'Number of agents choosing action 1 with genetic mutation: Mode:{m}')
                u=[np.mean(Uxt[i*window:(i+1)*window]) for i in range(int(iterations/window))]
                U[c].append(u)
        c += 1
    # Mutation with crossover
    for mo in mode_cm:
        for s in range(samples):
                st = time.time()
                xt, rarray, Uxt, states_array = adaptive_minority(m, N, ns, iterations, n, tau, form='cm', mode=mo)
                print(f'Simulation Time: {(time.time()-st)/60:2f} mins')
                if plot_ind:
                    plt.figure(figsize=(15,15))
                    plt.plot(xt)
                    plt.ylim(0,N)
                    plt.xlim(0,iterations)
                    plt.xlabel('Time')
                    plt.ylabel('x_t')
                    plt.title(f'Number of agents choosing action 1 with genetic mutation with crossover: Mode:{m}')
                u=[np.mean(Uxt[i*window:(i+1)*window]) for i in range(int(iterations/window))]
                U[c].append(u)
        c += 1
    # Basic MG
    for s in range(samples):
        st = time.time()
        xt, rarray, Uxt, states_array = adaptive_minority(m, N, ns, iterations, n=0)
        print(f'Simulation Time: {(time.time()-st)/60:2f} mins')
        if plot_ind:
            plt.figure(figsize=(15,15))
            plt.plot(xt)
            plt.ylim(0,N)
            plt.xlim(0,iterations)
            plt.xlabel('Time')
            plt.ylabel('x_t')
            plt.title(f'Number of agents choosing action 1 with genetic mutation with crossover: Mode:{m}')
        u=[np.mean(Uxt[i*window:(i+1)*window]) for i in range(int(iterations/window))]
        U[c].append(u)    
    # U vs Time
    plt.figure(figsize=(15,15))
    marks = ['s','o','D','v','^','p','+','*','x','h','<']
    label = ['A: Crossover - Parents: Random, Offsprings: Same',
              'B: Crossover - Parents: Random, Offsprings: Random',
              'C: Crossover - Parents: Random, Offsprings: Worst',
              'D: Crossover - Parents: Best, Offsprings: Same',
              'E: Crossover - Parents: Best, Offsprings: Random',
              'F: Crossover - Parents: Best, Offsprings: Worst',
              'G: Mutation - Mode: Tail',
              'H: Mutation - Mode: Long',
              'I: Mutation with Crossover - Mode: Long',
              'J: Mutation with Crossover - Mode: Short',
              'K: No adaptive scheme',]
    for i, u in enumerate(U):
        plt.plot(np.array(u).mean(axis=0), marker=marks[i], linestyle='None', markerfacecolor='None', markersize=5, label=label[i])
    plt.title('Scaled Utility of different adaptive mechanism', fontsize=20)
    plt.xlabel('Scaled Time', fontsize=20)
    plt.ylabel('U', fontsize=20)
    plt.legend(loc=4)
    # log(1-U) vs log(Time)
    plt.figure(figsize=(15,15))
    for i, u in enumerate(U):
        plt.loglog(1-np.array(u).mean(axis=0), marker=marks[i], linestyle='None', markerfacecolor='None', markersize=5, label=label[i])
    plt.title('Scaled Utility deviation from U_max = 1 of different adaptive mechanism', fontsize=20)
    plt.xlabel('Scaled Time', fontsize=20)
    plt.ylabel('1-U', fontsize=20)
    plt.ylim(0.00001,1)
    plt.legend(loc=3)
    return U

def mg_revenge(form='co', p1='Random', p2='Same'):     
    def update_scores(d, v, r, jchoice, actions, active, win):
        r1 = r.copy()
        for i in range(N):
            for j in range(ns):
                if d[i,j,active] == win:
                    v[i,j] += 1
            if d[i,jchoice[i],active] ==  win:
                r1[i] = r[i] + 1
        return v, r1
    def update_state(current, new):
        return (2 * current + new) % states
    m = 4
    N = 501
    ns = 10
    iterations = 10000
    tt = 3000
    n = 0.3
    tau = 40
    xM = int((N-1)/2)
    states = 2**m # number of states
    sequence = randint(0, 2, N*ns*states)
    darray = sequence.reshape((N, ns, states)) # array of strategies
    varray = np.zeros((N, ns)) # array of virtual scores
    rarray = np.zeros((N, iterations+1)) # array of real scores
    active = 0 # initialize input state at t=0
    states_array = [0]
    xt = [] # array of number of agents choosing action 1
    target_agent = [] # array of the worst 5 agents at t=tt
    adapt = False
    # Simulation
    for k in range(1, iterations+1):
        if k == tt:
            target_agent = rarray[:,tt-1].argsort()[:5]
        if k>= tt:
            if k % tau == 0:
                adapt = True
                rsorted = np.sort(rarray[:,k-1])[::-1]
                bound = rsorted[int(N*(1-n))-1]
            else:
                adapt = False
        actions = [] # array of agents' action in each iteration
        jchoice = [] # array of strategy chosen in each iteration
        for i in range(N):
            if adapt and (i in target_agent):
                if rarray[i,k-1]<=bound:
                    if form == 'co':
                        darray[i,:,:], varray[i,:] = replacement_co(darray[i,:,:], varray[i,:], p1, p2)
                    elif form == 'm':
                        darray[i,:,:], varray[i,:] = replacement_m(darray[i,:,:], varray[i,:], 2, p1)
                    else:
                        darray[i,:,:], varray[i,:] = replacement_cm(darray[i,:,:], varray[i,:], 2, p1)
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
        xt.append(sum(np.array(actions)==1))
        if xt[-1] > xM:
            winning = 0
        else:
            winning = 1
        varray, rarray[:,k] = update_scores(darray, varray, rarray[:,k-1], jchoice, actions, active, winning)
        active = update_state(active, winning)
        states_array.append(active)
    r_norm = rarray-rarray.mean(axis=0)
    normal_agents = [i for i in range(N) if (i not in target_agent)]    
    plt.figure(figsize=(10,10))
    for idx, i in enumerate(normal_agents):
        if idx == 0:
            plt.plot(r_norm[i,:], c='g', label='Other agents playing basic minority game')
        else:
            plt.plot(r_norm[i,:], c='g')
    for idx, i in enumerate(target_agent):
        if idx == 0:
            plt.plot(r_norm[i,:], c='b', label=f'Agents with adaptive Scheme J after t={tt}')
        else:
            plt.plot(r_norm[i,:], c='b')
    plt.legend()
    return xt, rarray, target_agent, states_array

def mg_competition(n=0.4,tau=10):     
    def update_scores(d, v, r, jchoice, actions, active, win):
        r1 = r.copy()
        for i in range(N):
            for j in range(ns):
                if d[i,j,active] == win:
                    v[i,j] += 1
            if d[i,jchoice[i],active] ==  win:
                r1[i] = r[i] + 1
        return v, r1
    
    def update_state(current, new):
        return (2 * current + new) % states
    
    # Initialization
    m = 4
    N = 49*11
    ns = 10
    iterations = 10000
    tt = 3000
    xM = int((N-1)/2)
    states = 2**m # number of states
    sequence = randint(0, 2, N*ns*states)
    darray = sequence.reshape((N, ns, states)) # array of strategies
    varray = np.zeros((N, ns)) # array of virtual scores
    rarray = np.zeros((N, iterations+1)) # array of real scores
    active = 0 # initialize input state at t=0
    states_array = [0]
    xt = [] # array of number of agents choosing action 1
    groups = list(np.split(np.random.permutation(N), 11))
    adapt = False
    # Simulation
    print(f'Simulating competition with m={m},N={N},S={ns},T={iterations},tau={tau},n={n}')
    for k in range(1, iterations+1):
        if (k % tau == 0) and (k>=tt):
            adapt = True
            rsorted = np.sort(rarray[:,k-1])[::-1]
            bound = rsorted[int(N*(1-n))-1]
        else:
            adapt = False
            bound = 0
        actions = [] # array of agents' action in each iteration
        jchoice = [] # array of strategy chosen in each iteration
        for i in range(N):
            if adapt and (rarray[i,k-1]<=bound):
                if  i in groups[0]:
                    darray[i,:,:], varray[i,:] = replacement_co(darray[i,:,:], varray[i,:], 'Random', 'Same')
                elif i in groups[1]:
                    darray[i,:,:], varray[i,:] = replacement_co(darray[i,:,:], varray[i,:], 'Random', 'Random')
                elif i in groups[2]:
                    darray[i,:,:], varray[i,:] = replacement_co(darray[i,:,:], varray[i,:], 'Random', 'Worst')
                elif i in groups[3]:
                    darray[i,:,:], varray[i,:] = replacement_co(darray[i,:,:], varray[i,:], 'Best', 'Same')
                elif i in groups[4]:
                    darray[i,:,:], varray[i,:] = replacement_co(darray[i,:,:], varray[i,:], 'Best', 'Random')
                elif i in groups[5]:
                    darray[i,:,:], varray[i,:] = replacement_co(darray[i,:,:], varray[i,:], 'Best', 'Worst')
                elif i in groups[6]:
                    darray[i,:,:], varray[i,:] = replacement_m(darray[i,:,:], varray[i,:], 2, mode='Tail')
                elif i in groups[7]:
                    darray[i,:,:], varray[i,:] = replacement_m(darray[i,:,:], varray[i,:], 2, mode='Long')
                elif i in groups[8]:
                    darray[i,:,:], varray[i,:] = replacement_cm(darray[i,:,:], varray[i,:], 2, mode='Long')
                elif i in groups[9]:
                    darray[i,:,:], varray[i,:] = replacement_cm(darray[i,:,:], varray[i,:], 2, mode='Short')
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
        xt.append(sum(np.array(actions)==1))
        if xt[-1] > xM:
            winning = 0
        else:
            winning = 1
        varray, rarray[:,k] = update_scores(darray, varray, rarray[:,k-1], jchoice, actions, active, winning)
        active = update_state(active, winning)
        states_array.append(active)
    return xt, rarray, groups

# n_array = [0.1,0.2,0.3,0.4,0.5,0.6]
n_array = [0.1]
# tau_array = [10,20,30,40,50,60]
tau_array = [60]
samples = 10
path = 'C:\\Users\\aauser\\Desktop\\MscDDM\\5003\\HW\\Final Project\\Figures\\Project_Parameters\\Competition'
for nnn in n_array:
    for ttt in tau_array:
        performance_array = []
        for s in range(samples):
            print(f'n={nnn}, tau={ttt}, Sample {s+1}')
            st = time.time()
            b, r, g = mg_competition(n=nnn, tau=ttt)
            print(f'Simulation Time: {(time.time()-st)/60:2f} mins')
            r_norm = r-r.mean(axis=0)
            performance_average = []
            for i in range(11):
                gr = r_norm[g[i],:]
                performance_average.append(gr.mean(axis=0))
            performance_array.append(performance_average)
            
        color = ['b','g','r','c','k','y','m','orange','grey','lime','pink']
        schemes = ['A','B','C','D','E','F']
        mode_m = ['Tail','Long']
        mode_cm = ['Long','Short']
        
        fig = plt.figure(figsize=(10,10))
        for i in range(11):
            average_performance = np.zeros(len(performance_array[0][0]))
            for j in range(samples):
                average_performance += performance_array[j][i]
            average_performance = average_performance/samples
            if i<6:
                plt.plot(average_performance, c=color[i], label=f'Agents with genetic crossover scheme {schemes[i]}')
            elif i<8:
                plt.plot(average_performance, c=color[i], label=f'Agents with genetic mutation mode {mode_m[i-6]}')
            elif i<10:
                plt.plot(average_performance, c=color[i], label=f'Agents with genetic mutation with crossover mode {mode_cm[i-8]}')
            else:
                plt.plot(average_performance, c=color[i], label='Agents without adaptive scheme')
        plt.legend()
        plt.ylabel('Performance')
        plt.title('Average performance of different adaptive schemes')
        fig.savefig(f'{path}\\Average_Performance_m4_N441_S10_T10000_n{nnn}_tau{ttt}.png')
        plt.close()
