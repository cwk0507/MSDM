import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#define data
labels = ['Jan','Feb','Mar','Apr','May']
US = [20000,22000,19000,23000,24000]
Europe = [14500,14600,14500,14450,16000]
Asia = [26000,17000,18000,28000,30000]
SA = [5000,3000,3500,6000,2000]


 
fig , ax = plt.subplots(1,2,figsize=(20,8))
ax1=ax[0]
ax2=ax[1]

id_label = np.arange(len(labels))
width = 0.2 
ax1.bar(id_label - 3/2*width, US, width, color='b', label='US')
ax1.bar(id_label - width/2, Europe, width, color='orange', label='Europe')
ax1.bar(id_label + width/2, Asia, width, color='grey', label='Asia')
ax1.bar(id_label + +3/2*width, SA, width, color='y', label='South America')

ax1.set_title('Clustered Column',color='k',fontsize=30,pad=30)
ax1.yaxis.set_major_locator(MultipleLocator(5))
# plt.xticks(id_label,labels=labels,fontsize=15)
# plt.yticks(np.linspace(0,40000,5),fontsize=15)
# plt.ylim(0,40000)
# plt.legend(loc='upper center',frameon=False,ncol=4,bbox_to_anchor=(0.5, 1.1),fontsize=15)
# plt.tick_params(axis='y',which='minor',length=7)

# for spine in plt.gca().spines.values():
#     spine.set_visible(False)

# plt.tick_params(axis='both',
#                 which='both',
#                 top=False, 
#                 bottom=False, 
#                 left=True, 
#                 right=False, 
#                 labelleft=True, 
#                 labelbottom=True)
# plt.grid(b=True, axis = 'y', which='major', color='k', linestyle='-')

# plt.subplot(122)
# p1 = plt.barh(id_label, US, width,color='b', label='US')
# p2 = plt.barh(id_label, Europe, width,left=US,color='orange', label='Europe')
# p3 = plt.barh(id_label, Asia, width,left=[US[i]+Europe[i] for i in range(5)],color='grey', label='Asia')
# p4 = plt.barh(id_label, SA, width,left=[US[i]+Europe[i]+Asia[i] for i in range(5)],color='y', label='South America')

# plt.title('Stacked Bar',color='k',fontsize=40,pad=30)
# plt.xlim(0,80000)
# plt.xticks(np.linspace(0,80000,5),fontsize=15)
# plt.yticks(id_label,labels=labels,fontsize=15)
# plt.legend(loc='lower center',frameon=False,ncol=4,bbox_to_anchor=(0.5, -0.15),fontsize=15)

# for spine in plt.gca().spines.values():
#     spine.set_visible(False)


# plt.tick_params(axis='both',
#                 which='both',
#                 top=False, 
#                 bottom=True, 
#                 left=False, 
#                 right=False, 
#                 labelleft=True, 
#                 labelbottom=True)
# plt.grid(b=True, axis = 'x', which='major', color='k', linestyle='-')
