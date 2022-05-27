import requests
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import datetime
import math
from numpy import *

eu_all = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 
      'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 
      'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy',
      'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
      'Netherlands', 'Poland', 'Portugal', 'Romania',
      'Slovakia', 'Slovenia', 'Spain', 'Sweden']
    
url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
url1 = 'https://www.worldometers.info/coronavirus/'

r = requests.get(url)
data = r.text.split('\n')
data1 = [data[i].split(',') for i in range(len(data))]

r1 = requests.get(url1)
Soup = BeautifulSoup(r1.text, 'html.parser')
temp = Soup.find_all('div',class_='maincounter-number')
total_deaths = int((temp[1].text[1:-1]).replace(',',''))

# Plot 1
ncols = len(data1[0])
for i in range(len(data1)-1):
    if len(data1[i]) > ncols:
        c=0
        while '"' not in data1[i][c]:
            c+=1
        temp = data1[i].pop(c)
        data1[i][c]=temp+","+data1[i][c]

df = pd.DataFrame(data1[1:len(data1)-1],columns=data1[0])
for c in df.columns:
    df[c] = df[c].apply(pd.to_numeric,errors='ignore')

hk_index = list(df.index[df['Province/State']=='Hong Kong'])[0]
hk = df.iloc[hk_index][5:]
df = df.drop(index = hk_index)
df_cum = df.groupby('Country/Region').sum()
df_cum_log = df_cum.T.apply(np.log10).replace(-inf,0).iloc[3:,:]
hk_log = hk.apply(lambda x: np.log10(x)).replace(-inf,0)
eu = df_cum.T[eu_all].sum(axis=1)[3:]
eu_log = eu.apply(np.log10).replace(-inf,0)
dates = list(df_cum_log.index)
day1 = datetime.datetime.strptime(dates[0], '%m/%d/%y')
dayn = datetime.datetime.strptime(dates[-1], '%m/%d/%y')
ndays = (dayn-day1).days+1
label_date = [dates[i] for i in range(ndays) if i%7==0]
x = np.linspace(0,ndays-1,ndays)
target = {'US':('#0070C0','United States'),
          'European Union':('#C00000','European Union'),
          'United Kingdom':('#FF66CC','United Kingdom'),
          'China':('#92D050','China'),
          'Singapore':('#00B0F0','Singapore'),
          'Taiwan*':('#FF0000','Taiwan'),
          'Hong Kong':('#FFFF00','Hong Kong')}
f = plt.figure(figsize=(20,10))
ax = plt.subplot()
for c in df_cum_log.columns:
    if c in target:
        ax.plot(x,df_cum_log[c],c=target[c][0],linewidth=2)
        ax.plot(x[-1],df_cum_log[c][-1],c=target[c][0],marker='.',markersize=10)
        ax.text(ndays+1,df_cum_log[c][-1],target[c][1],c=target[c][0])
    else:
        ax.plot(x,df_cum_log[c],c='grey',alpha=0.3)
# add plot HK
ax.plot(x,hk_log,c=target['Hong Kong'][0],linewidth=2)
ax.plot(x[-1],hk_log[-1],c=target['Hong Kong'][0],marker='.',markersize=10)
ax.text(ndays+1,hk_log[-1],'Hong Kong',c=target['Hong Kong'][0])
# add plot EU
ax.plot(x,eu_log,c=target['European Union'][0],linewidth=2)
ax.plot(x[-1],eu_log[-1],c=target['European Union'][0],marker='.',markersize=10)
ax.text(ndays+1,eu_log[-1],'European Union',c=target['European Union'][0])
# fig control
ax.set_xlim(0,ndays)
ax.set_ylim(0,df_cum_log.max().max()+0.2)
levels = [1,2,5]
ylabels=[]
y = np.zeros((int(df_cum_log.max().max()))*3+1)
for i in range(int(df_cum_log.max().max())):
    for j in range(3):
        y[3*i+j]=i+np.log10(levels[j])
        ylabels.append(levels[j]*10**i)
y[-1] = int(df_cum_log.max().max())
ylabels.append(10**int(df_cum_log.max().max()))
ax.set_title('Cumulative confirmed cases of Covid-19 in United States, United Kingdom, China, Singapore, European Union, Taiwan and Hong Kong',fontsize=14,loc='left')
ax.set_xticks(np.linspace(0,int(ndays/7)*7,int(ndays/7)+1))
ax.set_xticklabels(label_date,rotation=90)
ax.set_yticks(y)
ax.set_yticklabels(ylabels)
ax.set_facecolor('#FFF1E5')
f.set_facecolor('#FFF1E5')
td = np.ones(len(x))*np.log10(total_deaths)
ax.plot(x,td,c='k',linewidth=2,label='Up-to-date Global Total Deaths')
ax.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
plt.legend(frameon=False,fontsize=15)

#Plot 2
f1 = plt.figure(figsize=(20,10))
ax1 = plt.subplot()
for idx, c in enumerate(df_cum_log.columns):
    tdf = df_cum.T.iloc[3:,idx]
    if tdf[-1]<1000:
        continue
    day1000 = datetime.datetime.strptime(tdf[tdf>1000].index[0], '%m/%d/%y')
    day_delta = (dayn-day1000).days+1
    day_shift = (day1000-day1).days
    if c in target:
        ax1.plot(x[:day_delta],df_cum_log[c][day_shift:],c=target[c][0],linewidth=2)
        ax1.plot(x[day_delta-1],df_cum_log[c][-1],c=target[c][0],marker='.',markersize=10)
        ax1.text(day_delta+1,df_cum_log[c][-1],target[c][1],c=target[c][0])
    else:
        ax1.plot(x[:day_delta],df_cum_log[c][day_shift:],c='grey',alpha=0.3)
# add plot HK
day1000hk = datetime.datetime.strptime(hk[hk>1000].index[0], '%m/%d/%y')
day_delta = (dayn-day1000hk).days+1
day_shift = (day1000hk-day1).days
ax1.plot(x[:day_delta],hk_log[day_shift:],c=target['Hong Kong'][0],linewidth=2)
ax1.plot(x[day_delta-1],hk_log[-1],c=target['Hong Kong'][0],marker='.',markersize=10)
ax1.text(day_delta+1,hk_log[-1],'Hong Kong',c=target['Hong Kong'][0])
# add plot EU
day1000eu = datetime.datetime.strptime(eu[eu>1000].index[0], '%m/%d/%y')
day_delta = (dayn-day1000eu).days+1
day_shift = (day1000eu-day1).days
ax1.plot(x[:day_delta],eu_log[day_shift:],c=target['European Union'][0],linewidth=2)
ax1.plot(x[day_delta-1],eu_log[-1],c=target['European Union'][0],marker='.',markersize=10)
ax1.text(day_delta+1,eu_log[-1],'European Union',c=target['European Union'][0])
#fig control
ax1.set_xlim(0,ndays+20)
ax1.set_ylim(3,df_cum_log.max().max()+0.2)

ax1.set_title('Cumulative confirmed cases of Covid-19 in United States, United Kingdom, China, Singapore, European Union, Taiwan and Hong Kong',fontsize=14,loc='left')
ax1.set_xticks(np.array([i*20 for i in range(1,int((ndays+20)/20))]))
# ax1.set_xticklabels(label_date,rotation=90)
ax1.set_yticks(y[9:])
ax1.set_yticklabels(ylabels[9:])
ax1.set_facecolor('#FFF1E5')
f1.set_facecolor('#FFF1E5')
ax1.grid()
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.set_xlabel('Number of days since 1000 total cases first recorded',fontsize=12)
ax1.xaxis.set_label_coords(0.15, -0.03)