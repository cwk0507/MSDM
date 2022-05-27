import matplotlib.pyplot as plt
import numpy as np

#define data
labels = ['Jan','Feb','Mar','Apr','May']
US = [20000,22000,19000,23000,24000]
Europe = [14500,14600,14500,14450,16000]
Asia = [26000,17000,18000,28000,30000]
SA = [5000,3000,3500,6000,2000]

c1='#4472C4'
c2='#ED7D31'
c3='#A5A5A5'
c4='#FFC000'
csfont = {'fontname':'Calibri'}

f = plt.figure(figsize=(20,4), dpi=100)

ax = plt.subplot(121)
id_label = np.arange(len(labels))
width = 0.2 
ax.bar(id_label - 3/2*width, US, width, color=c1, label='US')
ax.bar(id_label - width/2, Europe, width, color=c2, label='Europe')
ax.bar(id_label + width/2, Asia, width, color=c3, label='Asia')
ax.bar(id_label + +3/2*width, SA, width, color=c4, label='South America')

plt.title('Clustered Column',color=c3,fontsize=40,pad=60,**csfont)
plt.xticks(id_label,labels=labels,fontsize=15,**csfont)
plt.yticks(np.linspace(0,40000,5),fontsize=15,**csfont)
plt.ylim(-100,40000)
plt.legend(loc='upper center',frameon=False,ncol=4,bbox_to_anchor=(0.5, 1.2),fontsize=12)
plt.tick_params(axis='y',which='minor',length=7)

for spine in plt.gca().spines.values():
    spine.set_visible(False)

ax.set_axisbelow(True)
ax.minorticks_on()
plt.tick_params(axis='both',
                which='both',
                top=False, 
                bottom=False, 
                left=False, 
                right=False, 
                labelleft=True, 
                labelbottom=True)
plt.grid(b=True, axis = 'y', which='major', color='k', linestyle='-')
plt.grid(b=True, axis = 'y', which='minor', color='grey', linestyle=':')

ax1=plt.subplot(122)
width1 = 0.4
p1 = plt.barh(id_label, US, width1,color=c1, label='US')
p2 = plt.barh(id_label, Europe, width1,left=US,color=c2, label='Europe')
p3 = plt.barh(id_label, Asia, width1,left=[US[i]+Europe[i] for i in range(5)],color=c3, label='Asia')
p4 = plt.barh(id_label, SA, width1,left=[US[i]+Europe[i]+Asia[i] for i in range(5)],color=c4, label='South America')

plt.title('Stacked Bar',color=c3,fontsize=30,pad=30,**csfont)
plt.xlim(0,80100)
plt.xticks(np.linspace(0,80000,5),fontsize=15,**csfont)
plt.ylim(-0.5,4.6)
plt.yticks(id_label,labels=labels,fontsize=15,**csfont)
plt.legend(loc='lower center',frameon=False,ncol=4,bbox_to_anchor=(0.5, -0.3),fontsize=12)

for spine in plt.gca().spines.values():
    spine.set_visible(False)

ax1.set_axisbelow(True)
ax1.minorticks_on()
plt.tick_params(axis='both',
                which='both',
                top=False, 
                bottom=False, 
                left=False, 
                right=False, 
                labelleft=True, 
                labelbottom=True)
plt.grid(b=True, axis = 'x', which='major', color='k', linestyle='-')
