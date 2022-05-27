import matplotlib.pyplot as plt
import numpy as np

colors = ['#4472C4','#ED7D31','#A5A5A5','#FFC000',
          '#5B9BD5','#70AD47','#002060','#833C0C']
labels = ['Computers','Furniture','Office Supplies','Electrical Supplies',
          'Software','Day Planners','Health & Safety','Productivity Tools']
csfont = {'fontname':'Calibri'}

l_order = [0,4,1,5,2,6,3,7]

years = ['2017','2018','2019','2020']
cpu = [1400,2200,2100,3000]
f = [1200,1100,500,750]
os = [300,590,125,123]
es = [1800,1700,1900,2200]

sw = [1200,500,900,1250]
dp = [900,800,800,900]
hs = [400,300,200,200]
pt = [32,25,75,80]

data = [cpu,f,os,es,sw,dp,hs,pt]

fig, ax = plt.subplots(figsize=(12,6))
ax.margins(0)
for idx, val in enumerate(labels):
    ax.stackplot(years,data[idx],color=colors[idx],labels=[val])
ax.set_title('Too Many Items?',fontsize=20,pad=20)
ax.set_ylim(0,3500)
h,l = ax.get_legend_handles_labels()
h = [h[i] for i in l_order]
l = [l[i] for i in l_order]

ax.legend(h,l,ncol=4,loc='lower center',bbox_to_anchor=(0.5,-0.25),frameon=False)
for spine in ax.spines.keys():
    ax.spines[spine].set_visible(False)
ax.set_axisbelow(True)
plt.grid(b=True, axis = 'y', which='major', color='grey', linestyle='-')
ax.tick_params(axis='y',
                which='both',
                top=False, 
                bottom=False, 
                left=False, 
                right=False, 
                labelleft=True, 
                labelbottom=True)
fig.subplots_adjust(top=0.8,bottom=0.2)