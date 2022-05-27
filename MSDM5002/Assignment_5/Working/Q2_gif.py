import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

tt = [i for i in range(100)]
dtheta = 2*np.pi/100

# text_step = plt.text(2,2.5,'t=',color ='g',fontsize = 20)
for t in tt:
    fig, ax = plt.subplots(figsize=(5,5))
    ax.set_xlim(-1,np.pi)
    ax.set_ylim(-1,np.pi)
    ax.set_xticks(np.linspace(-1,3,5))
    theta = np.linspace(0,2*np.pi,100)
    r = 1
    xc = r*np.sin(theta)
    yc = r*np.cos(theta)
    plt.plot(xc,yc,c='b')
    plt.plot(np.zeros(100),np.linspace(-1,1,100),c='b')
    plt.plot(np.linspace(-1,1,100),np.zeros(100),c='b')
    xm = np.cos(dtheta*t)
    ym = -np.sin(dtheta*t)
    plt.plot([xm,xm],[0,ym],'g-')
    plt.plot([0,xm],[ym,ym],'r-')
    plt.plot([0,xm],[0,ym],'b-')
    xc = np.linspace(xm,2*np.pi,700)
    yc = -(50>=t)*np.sin((xc-xc[0])*np.pi*1.25+t/50*np.pi)+(50<t)*np.sin(-(xc-xc[0])*np.pi*1.25-t/50*np.pi)
    plt.plot(xc,yc,'g.')
    yc1 = np.linspace(ym,2*np.pi,700)
    xc1 = (50>=t)*np.cos(-(yc1-yc1[0])*np.pi*1.25+t/50*np.pi)+(50<t)*np.cos((yc1-yc1[0])*np.pi*1.25-t/50*np.pi)
    plt.plot(xc1,yc1,'r.')
    plt.text(2,2.5,'t='+str(t),color ='g',fontsize = 20)
    plt.savefig(str(t)+'.png')
    plt.pause(0.1)
    plt.close()
    




