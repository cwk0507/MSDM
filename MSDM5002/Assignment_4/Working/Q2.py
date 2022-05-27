import matplotlib.pyplot as plt
import numpy as np

#define data
labels = ['Coats','Jeans','Jackets','Trousers','Joggers','Suits','Hoodies','T-Shirts',
          'Shorts','Polo Shirts']
IR = [75,68,20,18,12,11,9,6,4,2]
CP = [0.33,0.64,0.72,0.8,0.86,0.91,0.95,0.97,0.99,1]

c1='#5B9BD5'
c2='#ED7D31'
csfont = {'fontname':'Calibri'}

width = 0.99
id_label = np.arange(len(labels))
yy = np.linspace(0,1,11)
f = plt.figure(figsize=(10,8), dpi=100)
ax = plt.subplot()


ax.bar(id_label + width/2, IR, width, color=c1)
ax.set_xlim(0,10)
ax.set_xticks(id_label + width/2)
ax.set_xticklabels(labels,style='italic')
ax.set_xlabel('Product',weight='bold',fontsize=12)

ax.set_ylim(0,80)
ax.set_ylabel('Items Returned',rotation=90,weight='bold',fontsize=12)
ax.set_title('Returns & Refunds',color='k',fontsize=20,weight='bold',**csfont)
for idx,val in enumerate(IR):
    ax.text(idx+width/2,val+1,val,color='k',ha='center',weight='bold')
    
ax1 = ax.twinx()
ax.tick_params(axis='both',
                which='both',
                top=False, 
                bottom=False, 
                left=False, 
                right=False, 
                labelleft=True, 
                labelbottom=True)
ax1.plot(id_label + width/2,CP,color=c2,linewidth=4,marker='o',label='Cummulative Return')
ax1.set_ylim(0,1.01)
ax1.set_yticks(yy)
ax1.set_yticklabels(['{:.0%}'.format(val) for val in yy])
for idx,val in enumerate(CP):
     ax1.text(idx+width/2,val+0.03,'{:.0%}'.format(val),color='r',ha='center',weight='bold')
    
for spine in ax.spines.keys():
    ax.spines[spine].set_visible(False)
    ax1.spines[spine].set_visible(False)
    
ax1.tick_params(axis='both',
                which='both',
                top=False, 
                bottom=False, 
                left=False, 
                right=False, 
                labelright=True, 
                labelbottom=False)