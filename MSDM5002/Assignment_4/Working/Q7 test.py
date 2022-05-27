import matplotlib.pyplot as plt
import numpy as np

n=100 # number of grid points
max_height = 1000
main_grid_portion = 1/5
main_grid_number = int(n*main_grid_portion)
step = int(n/main_grid_number)
levels = np.insert(np.linspace(0,max_height,7),0,-max_height)

xlist = np.linspace(0, n, n+1)
ylist = np.linspace(0, n, n+1)
X, Y = np.meshgrid(xlist, ylist)
Z = np.zeros(X.shape)

# generate main grid points level
for i in range(main_grid_number+1):
    for j in range(main_grid_number+1):
        Z[i*step,j*step]=np.random.uniform(-max_height,max_height)

# interplotion on main grid row
for i in range(main_grid_number+1):
    pointer = 0
    while pointer<n:
        level_1 = Z[i*step,pointer]
        level_2 = Z[i*step,pointer+step]
        interpolation = np.linspace(level_1,level_2,step+1)
        for idx, j in enumerate(range(pointer+1,pointer+step)):
            Z[i*step,j]=interpolation[idx+1]
        pointer += step

# interplotion on columns
for j in range(n+1):
    pointer = 0
    while pointer<n:
        level_1 = Z[pointer,j]
        level_2 = Z[pointer+step,j]
        interpolation = np.linspace(level_1,level_2,step+1)
        for idx, i in enumerate(range(pointer+1,pointer+step)):
            Z[i,j]=interpolation[idx+1]
        pointer += step


fig , ax = plt.subplots(figsize=(20,10))
ax.contour(X, Y, Z,levels=levels,colors='k')
ax.contourf(X, Y, Z,cmap='jet',levels=levels)
ax.set_xlim(0,n)
ax.set_ylim(0,n)
ax.tick_params(axis='both',
                which='both',
                bottom=False, 
                left=False, 
                labelleft=False, 
                labelbottom=False)