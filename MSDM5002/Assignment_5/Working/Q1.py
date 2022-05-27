import numpy as np
from skimage import measure

# Define the heart function 
def F(x,y,z):
    return (x**2 + 9*y**2/4 + z**2-1)**3 - x**2 * z**3 - 9*y**2 * z**3/80
    
# Set up mesh
n = 100

x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
z = np.linspace(-3,3,n)
X, Y, Z =  np.meshgrid(x, y, z, indexing='ij')

vol = F(X,Y,Z)

# Extract a 2D surface mesh from a 3D volume (F=0)
verts, faces, _, _ = measure.marching_cubes_lewiner(vol, 0, spacing=(6/n, 6/n, 6/n))


# Your Code Here
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')

# 3D heart
XX = verts[:,0]-3
YY = verts[:,1]-3
ZZ = verts[:,2]-3
ax.plot_trisurf(XX,YY,faces,ZZ,cmap='Spectral',alpha=0.8)

# heart contour
X, Z = np.meshgrid(np.linspace(-3,3,n*10), np.linspace(-3,3,n*10))
Y=np.zeros(X.shape)
F0=F(X, Y, Z) 
ax.contour3D(X, F0, Z, levels=[0], zdir='y',colors='r', offset=2)

# plot control
ax.set_xlim(-2,2)
ax.set_ylim(-2,2)
ax.set_zlim(-2,2)
ax.set_xlabel('x',labelpad=10)
ax.set_ylabel('y',labelpad=10)
ax.set_zlabel('z',labelpad=10)
ax.set_title('Love from Cheng Wing Kit')
ax.view_init(15,-70)