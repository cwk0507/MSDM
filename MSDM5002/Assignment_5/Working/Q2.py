import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig, ax = plt.subplots(figsize=(5,5))

#prepare something you want to update
ln1, = plt.plot ([],[],'g.')
ln2, = plt.plot ([],[],'g-')
ln3, = plt.plot ([],[],'r.')
ln4, = plt.plot ([],[],'r-')
ln5, = plt.plot ([],[],'b-')

text_step = plt.text(2,2.5,'t=',color ='g',fontsize = 20)
#the initializaion function
def init():
    ax.set_xlim(-1,np.pi)
    ax.set_ylim(-1,np.pi)
    ax.set_xticks(np.linspace(-1,3,5))
    return ln1,ln2,ln3,ln4,ln5,text_step

def updates(t):
    # set straight lines
    dtheta = 2*np.pi/100
    xm = np.cos(dtheta*t)
    ym = -np.sin(dtheta*t)
    ln2.set_data([xm,xm],[0,ym])
    ln4.set_data([0,xm],[ym,ym])
    ln5.set_data([0,xm],[0,ym])

    # set curve lines
    xc = np.linspace(xm,2*np.pi,700)
    yc = -(50>=t)*np.sin((xc-xc[0])*np.pi*1.25+t/50*np.pi)+(50<t)*np.sin(-(xc-xc[0])*np.pi*1.25-t/50*np.pi)
    ln1.set_data(xc,yc)
    yc1 = np.linspace(ym,2*np.pi,700)
    xc1 = (50>=t)*np.cos(-(yc1-yc1[0])*np.pi*1.25+t/50*np.pi)+(50<t)*np.cos((yc1-yc1[0])*np.pi*1.25-t/50*np.pi)
    ln3.set_data(xc1,yc1)
    text_step.set_text('t='+str(t))
    return ln1,ln2,ln3,ln4,ln5,text_step

# plot static circle and lines
theta = np.linspace(0,2*np.pi,100)
r = 1
xc = r*np.sin(theta)
yc = r*np.cos(theta)
plt.plot(xc,yc,c='b')
plt.plot(np.zeros(100),np.linspace(-1,1,100),c='b')
plt.plot(np.linspace(-1,1,100),np.zeros(100),c='b')

# plot lines animation
ani = FuncAnimation(fig=fig,
                    init_func=init,
                    func=updates,
                    frames=100, 
                    interval=100,
                    blit=True)
# ani.save('sin_cos_movie.gif')