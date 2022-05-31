import pandas as pd
import time
import matplotlib.pyplot as plt

file_path = 'C:\\Users\\aauser\\Desktop\\MscDDM\\5051\\Assignment\\Assignment1\\data\\TDCS_M06A_20190830_080000.csv'

df = pd.read_csv(file_path,header=None)
d1_vtype = list(df.iloc[:,0])
d2_dtime = list(df.iloc[:,1])
d3_gan = list(df.iloc[:,2])

n = [1/5,1/4,1/3,1/2,1] #portion of data

def counting_sort(d):
    max_item = max(d)
    counts = [0 for i in range(max_item + 1)]
    for i in d:
        counts[i]+=1
    return [i for i in range(len(counts)) for j in range(counts[i])]

def quick_sort(d):
    if len(d)<2:
        return d
    pivot = d[0]
    left = [i for i in d[1:] if i <= pivot]
    right = [i for i in d[1:] if i > pivot]
    return quick_sort(left)+[pivot]+quick_sort(right)

def counting_sort_by(array,max_rank=None,rank=lambda x: x):
    if max_rank==None:
        max_rank=0
        for item in array:
            if rank(item)>max_rank:
                max_rank=rank(item)
    counts = [[] for i in range(max_rank+1)]
    for item in array:
        counts[rank(item)].append(item)
    return [item for sublist in counts for item in sublist]     

def dig(rd,d):
    return 0 if d>=len(rd) else rd[-(d+1)]
    
def radix_LSD_sort(array):
    rd_array=[]
    max_length=0
    for item in array:
        list_item=list(item)
        rd_array.append(list_item)
        if max_length<len(list_item):
            max_length = len(list_item)
    for d in range(max_length):
        rd_array = counting_sort_by(rd_array,90,lambda rd: ord(dig(rd,d)))
    return [''.join(list_item) for list_item in rd_array]
    


timing_1 = []
for p in n:
    d1 = d1_vtype[:int(p*len(d1_vtype))]
    t0 = time.time()
    sorted_d1 = counting_sort(d1)
    timing_1.append(time.time()-t0)
    
fig = plt.subplots(3,1,figsize=(10,30))
ax1 = plt.subplot(311)
ax1.plot([int(n[i]*len(d1_vtype)) for i in range(len(n))],timing_1)
ax1.set_title('Sort "Vechicle Type" using Counting Sort')
ax1.set_xlim(0,len(d1_vtype))
ax1.set_xlabel('Size of array (n)')
ax1.set_ylabel('Sorting time',rotation=90)

timing_2 = []
for p in n:
    d2 = d2_dtime[:int(p*len(d2_dtime))]
    t0 = time.time()
    sorted_d2 = quick_sort(d2)
    timing_2.append(time.time()-t0)
    
ax2 = plt.subplot(312)
ax2.plot([int(n[i]*len(d2_dtime)) for i in range(len(n))],timing_2)
ax2.set_title('Sort "DerectionTime_O" using Quick Sort')
ax2.set_xlim(0,len(d2_dtime))
ax2.set_xlabel('Size of array (n)')
ax2.set_ylabel('Sorting time',rotation=90)

timing_3 = []
for p in n:
    d3 = d3_gan[:int(p*len(d3_gan))]
    t0 = time.time()
    sorted_d3 = radix_LSD_sort(d3)
    timing_3.append(time.time()-t0)
    
ax3 = plt.subplot(313)
ax3.plot([int(n[i]*len(d3_gan)) for i in range(len(n))],timing_3)
ax3.set_title('Sort "GantryID_O" using Radix_LSD_Sort')
ax3.set_xlim(0,len(d3_gan))
ax3.set_xlabel('Size of array (n)')
ax3.set_ylabel('Sorting time',rotation=90)