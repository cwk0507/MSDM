import matplotlib.pyplot as plt
import numpy as np

m = ['Dec','Jan','Feb','Mar','Apr','May','Jun',
     'Jul','Aug','Sep','Oct','Nov','Dec']
csat_s = np.array([71,62,68,55,68,54,45,66,66,98,75,94,71])
csat_g = np.array([58,72,83,67,79,89,85,96,50,56,31,41,58])
csfont = {'fontname':'Calibri'}

angles = np.linspace(0,2*np.pi,13)
angles_degree = np.linspace(0,360,13)

fig = plt.figure(figsize=(10,10),dpi=100) 
ax = plt.subplot(polar='True')
ax.set_theta_zero_location("N")  
ax.set_theta_direction(-1)
ax.plot(angles,csat_s,marker='.',markersize=20,c='b',linewidth=5,label='Simpson Ltd')
ax.plot(angles,csat_g,marker='.',markersize=20,c='g',linewidth=5,label='Griffin Ltd')
ax.set_thetagrids(angles_degree,m,fontsize=15)
ax.set_rgrids(np.linspace(0,100,11))
ax.tick_params(axis='y', which='both', labelleft=False)
ax.legend(loc=8,ncol=2,bbox_to_anchor=(0.5,-0.12),frameon=False)
fig.text(0.1, 0.9, "Customer Satisfication Score in 2019: ",
         fontsize=20,va="bottom", c='k',weight='bold',**csfont)
fig.text(0.53, 0.9, "Simpson Ltd", va="bottom", fontsize=20,
         c='b',weight='bold',**csfont)
fig.text(0.68, 0.9, "and", va="bottom", fontsize=20,
         c='k',weight='bold',**csfont)
fig.text(0.73, 0.9, "Griffin Ltd", va="bottom", fontsize=20,
         c='g',weight='bold',**csfont)
fig.subplots_adjust(top=0.8,bottom=0.2)
                