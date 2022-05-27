import matplotlib.pyplot as plt

labels = ['Poor','Fair','Good','Great','Excellent','']
levels = [20,20,20,20,20,100]
colors = ['#FF0000','#F4B183','#FFD966','#A9D18E','#70AD47','w']
pie_lv = [20,1]
pie_lv.append(200-pie_lv[0]-pie_lv[1])
colors_pie = ['#00000000','#000000','#00000000']
csfont = {'fontname':'Calibri'}

fig, ax = plt.subplots(figsize=(10,10))
ax.pie(levels,radius=1,colors=colors,startangle=180,counterclock=False)
ax.pie(levels,radius=0.5,colors=['w']*6,wedgeprops=dict(edgecolor='w'))
for idx, val in enumerate(labels):
    ax.text((idx-2)*3/5+idx%2*(idx-2)*0.1,1.1-abs(idx-2)*0.4+idx%2*0.2,val,
            fontsize=18,ha='center',weight='bold',**csfont)
ax.pie(pie_lv,radius=1,colors=colors_pie,startangle=180,counterclock=False)
ax.text(0,-0.2,pie_lv[0],fontsize=30,weight='bold',ha='center',**csfont)
ax.set_title('Arizona',fontsize=20)