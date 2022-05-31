def radix_MSD_sort(array,i):
    # base case
    if len(array)<=1:
        return array
    
    rd_array = [list(str(j)) for j in array]
    
    # divide
    done_bucket = []
    buckets=[[] for j in range(10)]
    
    for elem in rd_array:
        if len(elem) < i:
            done_bucket.append(elem)
        else:
            buckets[int(elem[i])].append(elem)
    
    # conquer
    buckets = [radix_MSD_sort([int(''.join(j)) for j in b],i+1) for b in buckets]
        
    # join
    return done_bucket + [b for sublist in buckets for b in sublist]
        
array = [123,232,322,321,323,132,133,242,241,24]
print(radix_MSD_sort(array,0))