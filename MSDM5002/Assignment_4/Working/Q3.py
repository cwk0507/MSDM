import matplotlib.pyplot as plt
import numpy as np

def get_color(x,y,h1=70000,v1=15000):
    red='#FF0000'
    orange='#FFC000'
    cyan='#92D050'
    green='#00B050'
    if x<=v1:
        if y<=h1:
            return orange
        else:
            return green
    else:
        if y<=h1:
            return red
        else:
            return cyan

labels = ['SEO','Email marketing','Google AdWords','Facebook Ads','Instagram Ads',
          'Content marketing','LinkedIn Ads','Twitter Ads']
expenses = [29000,8673,11004,19543,8943,24000,24500,4000]
revenue = [95523,124045,94245,18543,24345,85434,35000,11453]
colors = [get_color(expenses[i],revenue[i]) for i in range(len(expenses))]

h = 70000
v = 15000

c='#5B9BD5'
csfont = {'fontname':'Calibri'}

xx = np.linspace(0,v*2,7)
yy = np.linspace(0,h*2,8)
x1 = [v for i in range(len(yy))]
y1 = [h for i in range(len(xx))]
f = plt.figure(figsize=(9,6), dpi=100)
ax = plt.subplot()

ax.set_title('Overall Marketing Performance',fontsize=20,pad=20,**csfont)
ax.set_xlim(0,v*2)
ax.set_xticklabels(['${:,.0f}'.format(x) for x in xx],fontsize=12,**csfont)
ax.set_xlabel('Expenses',fontsize=16,**csfont)
ax.set_ylim(0,h*2)
ax.set_yticklabels(['${:,.0f}'.format(y) for y in yy],fontsize=12,**csfont)
ax.set_ylabel('Revenue',fontsize=16,rotation=90,**csfont)
ax.tick_params(axis='both',
                which='both',
                top=False, 
                bottom=False, 
                left=False, 
                right=False, 
                labelleft=True, 
                labelbottom=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('grey')
ax.spines['left'].set_color('grey')

ax.plot(x1,yy,color='k')
ax.plot(xx,y1,color='k')
ax.scatter(expenses,revenue,s=80,color=c)
for i in range(len(expenses)):
    ax.text(expenses[i],revenue[i]+5000,labels[i],color=colors[i],
            ha='center',fontsize=14,weight='bold',**csfont)