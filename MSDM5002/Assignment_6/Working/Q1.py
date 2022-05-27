# Your Code Here

import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import datetime

alldate=[]
today = datetime.date.today()

for n in range(13,-1,-1):
    alldate.append((today-datetime.timedelta(days=n)).strftime('%Y%m%d'))

alltime=[]
alltemp=[]
allwind=[]
allhumidity=[]
allbar=[]
allvis=[]

ndt=0
for dt in alldate:
    
    url = 'https://www.timeanddate.com/weather/hong-kong/hong-kong/historic' + '?hd='+dt
    r = requests.get(url)
    html_contents = r.text
    html_soup = BeautifulSoup(html_contents, 'html.parser')
    
    table_body = html_soup.find('table', id="wt-his").find('tbody')
    
    for row in table_body.find_all('tr'):
        tmp=row.find('th').text
        alltime.append(float(tmp[0:2]+'.'+str(int(int(tmp[3:5])/6)))+24*ndt)
        values = row.find_all('td')
        alltemp.append(float(values[1].text[0:2]))
        allwind.append(float(values[3].text[0:2]))
        allhumidity.append(float(values[5].text[0:2]))
        allbar.append(float(values[6].text[0:4]))
        try:
            allvis.append(float(values[7].text[0:2]))
        except:
            allvis.append(0)
    ndt=ndt+1
    
f = plt.figure(figsize=(20,10), dpi=100)
f.suptitle('Weather information from {} to {}'.format((today-datetime.timedelta(days=13)).strftime('%Y/%m/%d'),today.strftime('%Y/%m/%d')), fontsize=16)

# temp vs time
ax1 = plt.subplot(221)
ax1.plot(alltime,alltemp)
ax1.set_xticks(np.linspace(0,ndt*24,ndt,endpoint=False))
ax1.set_xticklabels(alldate,rotation=-90,fontsize=10)
ax1.set_xlabel('Date',fontsize=10)
ax1.set_ylabel('Temperature ($^o$C)',fontsize=10)

# wind vs time
ax2 = plt.subplot(222)
ax2.plot(alltime,allwind)
ax2.set_xticks(np.linspace(0,ndt*24,ndt,endpoint=False))
ax2.set_xticklabels(alldate,rotation=-90,fontsize=10)
ax2.set_xlabel('Date',fontsize=10)
ax2.set_ylabel('Wind (km/h)',fontsize=10)

# humidity vs time
ax3 = plt.subplot(234)
ax3.plot(alltime,allhumidity)
ax3.set_xticks(np.linspace(0,ndt*24,ndt,endpoint=False))
ax3.set_xticklabels(alldate,rotation=-90,fontsize=10)
ax3.set_xlabel('Date',fontsize=10)
ax3.set_ylabel('Humidity (%)',fontsize=10)

# Barometer vs time
ax4 = plt.subplot(235)
ax4.plot(alltime,allbar)
ax4.set_xticks(np.linspace(0,ndt*24,ndt,endpoint=False))
ax4.set_xticklabels(alldate,rotation=-90,fontsize=10)
ax4.set_xlabel('Date',fontsize=10)
ax4.set_ylabel('Barometer (mbar)',fontsize=10)

# visibilty vs time
ax5 = plt.subplot(236)
ax5.plot(alltime,allvis)
ax5.set_xticks(np.linspace(0,ndt*24,ndt,endpoint=False))
ax5.set_xticklabels(alldate,rotation=-90,fontsize=10)
ax5.set_xlabel('Date',fontsize=10)
ax5.set_ylabel('Visibility (km)',fontsize=10)

plt.tight_layout(pad=0.5,h_pad=0.5,w_pad=0.5)