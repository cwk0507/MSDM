def count_sort_by(array,max_rank=None,rank=lambda x: x):
    if max_rank==None:
        max_rank=0
        for item in array:
            if rank(item)>max_rank:
                max_rank=rank(item)
    counts = [[] for i in range(max_rank+1)]
    for item in array:
        counts[rank(item)].append(item)
    return [item for sublist in counts for item in sublist]

            
def dig(x,d):
    return 0 if d>=len(x) else x[-(d+1)]

def radix_LSD_sort(array):
    rd_array = [list(str(i)) for i in array]
    for i in range(max([len(i) for i in rd_array])):
        rd_array = count_sort_by(rd_array,90,lambda x: ord(dig(x,i)))
    rd_array = [int(''.join(i)) for i in rd_array]
    return rd_array


array = [123,232,322,321,323,132,133,242,241]
print(radix_LSD_sort(array))