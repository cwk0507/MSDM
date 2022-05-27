from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n=50

t0 = MPI.Wtime()
for i in range(n):
    if rank==0:
        comm.send(f'Message {i}',dest=1,tag=i)
        print(f'rank {rank} sends {i} to rank {rank+1}')
        sys.stdout.flush()
        rn = comm.recv(source=size-1,tag=(size-1)*size+i)
        print(f'rank {rank} receives {rn} from rank {size-1}')
        sys.stdout.flush()
    
    elif rank < size - 1:
        rn = comm.recv(source=rank-1,tag=(rank-1)*size+i)
        print(f'rank {rank} receives {rn} from rank {rank-1}')
        sys.stdout.flush()
        comm.send(rn,dest=rank+1,tag=rank*size+i)
        print(f'rank {rank} sends {rn} to rank {rank+1}')
        sys.stdout.flush()
        
    else:
        rn = comm.recv(source=rank-1,tag=(rank-1)*size+i)
        print(f'rank {rank} receives {rn} from rank {rank-1}')
        sys.stdout.flush()
        comm.send(rn,dest=0,tag=rank*size+i)
        print(f'rank {rank} sends {rn} to rank 0')
        sys.stdout.flush()

dt = MPI.Wtime() - t0
print(f'Elapsed time for process {rank} out of {size}: {dt} seconds')