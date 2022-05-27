import requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import math

def is_valid_date(d):
    try:
        d1 = datetime.datetime.strptime(d, '%d/%m/%Y')
        return True
    except:
        return False

url = 'http://www.chp.gov.hk/files/misc/enhanced_sur_covid_19_eng.csv'

r = requests.get(url)
data = r.text.split('\r\n')
data1 = [data[i].split(',') for i in range(len(data))]
df = pd.DataFrame(data1[1:len(data1)-1],columns=data1[0])
report_date = max(df['Report date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y')))
df_date = df[df['Date of onset'].apply(lambda x: is_valid_date(x))]
dates=df_date['Date of onset'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))
day1 = min(dates)
dayn = max(dates)
delta = (dayn - day1).days+1
all_dates = [(day1+datetime.timedelta(days=i-1)).strftime('%d/%m/%Y') for i in range(delta+2)]
labels = list(df_date['Case classification*'].unique())
counts = np.zeros([len(labels),len(all_dates)])
for i in range(len(all_dates)):
    for j in range(len(labels)):
        counts[j,i]=df_date['Case no.'][df_date['Date of onset']==all_dates[i]][df_date['Case classification*']==labels[j]].count()
# Plot 1
f = plt.figure(figsize=(20,5))
ax = plt.subplot()
xx = np.linspace(0,len(all_dates)-1,len(all_dates))
for i in range(len(labels)):
    if i==0:
        ax.bar(xx,counts[i],label=labels[i])
        bottoms = counts[i].copy()
    else:
        ax.bar(xx,counts[i],bottom=bottoms,label=labels[i])
        bottoms += counts[i]
ax.text(0,180,'Epidemic curve of confirmed and probable cases of COVID-19 in Hong Kong (as of {})'.format(report_date.strftime('%d %b %Y')),fontsize=20)
ax.text(0,160,'Number of confirmed and probable cases  = {}'.format(df.shape[0]),fontsize=20)
ax.set_xlim(0,delta)
ax.set_ylim(0,max(counts.sum(axis=0))+10)
label_date = [all_dates[i] for i in range(delta) if i%7==0]
ax.set_xticks(np.linspace(0,int(delta/7)*7,int(delta/7)+1))
ax.set_xticklabels(label_date,rotation=90)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tick_params(axis='y',
                which='both',
                left=False,  
                labelleft=True, 
                )
ax.set_xlabel('Date of onset',fontsize=20)
ax.set_ylabel('Number of cases',fontsize=20)
plt.legend(loc='lower center',ncol=2,bbox_to_anchor=(0.5, -0.6),fontsize=15)

# Plot 2
# report_dates=df['Report date'].apply(lambda x: datetime.datetime.strptime(x, '%d/%m/%Y'))
# dayr1 = min(report_dates)
# dayrn = max(report_dates)
# deltar = (dayrn - dayr1).days +1
# all_report_dates = [(dayr1+datetime.timedelta(days=i-1)).strftime('%d/%m/%Y') for i in range(deltar+2)]
# rcounts = np.zeros([len(labels),len(all_report_dates)])
# for i in range(len(all_report_dates)):
#     for j in range(len(labels)):
#         rcounts[j,i]=df['Case no.'][df['Report date']==all_report_dates[i]][df['Case classification*']==labels[j]].count()
# ic = rcounts[0]+rcounts[1]
# lc = rcounts.sum(axis=0) - ic
# cic = 0
# clc = 0
# cumulative_cases = np.zeros([2,len(all_report_dates)])
# rlabel_date = [all_report_dates[i] for i in range(deltar) if i%7==0]
# for i in range(len(all_report_dates)):
#     cumulative_cases[0,i] = ic[i]+cic
#     cic += ic[i]
#     cumulative_cases[1,i] = lc[i]+clc
#     clc += lc[i]
# xxr = np.linspace(0,len(all_report_dates)-1,len(all_report_dates))
import_index = [i for i in range(len(labels)) if labels[i] in ['Imported case','Epidemiologically linked with imported case' ]]
ic = np.zeros(len(all_dates))
for i in import_index:
    ic+=counts[i]
lc = counts.sum(axis=0) - ic
cic = 0
clc = 0
cumulative_cases = np.zeros([2,len(all_dates)])
for i in range(len(all_dates)):
    cumulative_cases[0,i] = ic[i]+cic
    cic += ic[i]
    cumulative_cases[1,i] = lc[i]+clc
    clc += lc[i]
f1 = plt.figure(figsize=(10,5))
ax1 = plt.subplot()
ax1.set_title('Cumulative cases in Hong Kong',fontsize=20)
ax1.set_xlim(0,delta)
ax1.set_ylim(0,cumulative_cases.max()+100)
# ax1.plot(xxr,cumulative_cases[1],label='Local')
# ax1.plot(xxr,cumulative_cases[0],label='Import')
ax1.plot(xx,cumulative_cases[1],label='Local')
ax1.plot(xx,cumulative_cases[0],label='Import')
ax1.set_xticks(np.linspace(0,int(delta/7)*7,int(delta/7)+1))
# ax1.set_xticklabels(rlabel_date,rotation=90)
ax1.set_xticklabels(label_date,rotation=90)
# ax1.set_xlabel('Report Date')
ax1.set_xlabel('Date of onset')
ax1.set_ylabel('Cumulative number of cases')
ax1.spines['top'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(b=True, axis = 'y', which='major', color='grey', linestyle='-')
plt.tick_params(axis='both',
                which='both',
                left=False, 
                bottom=False,
                labelleft=True,
                labelbottom=True
                )
plt.legend(loc='lower center',ncol=2,bbox_to_anchor=(0.5, -0.4),fontsize=12)

# Plot 3
f2 = plt.figure(figsize=(5,5))
df_deceased = df[df['Hospitalised/Discharged/Deceased']=='Deceased']
age_max = df_deceased['Age'].apply(int).max()
bin_list = list(range(0,math.ceil(age_max/10)*10+1,math.ceil(age_max/10)))
ax2 = plt.subplot()
all_ages = list(df['Age'].apply(int))
deceassed_ages = list(df_deceased['Age'].apply(int))
result = plt.hist(deceassed_ages,bins=bin_list,alpha=0.5)
ax2.set_title('Deceased cases')
ax2.set_xticks(bin_list)
ax2.set_xlabel('Age')
ax2.set_ylabel('Number of deceased cases')
dead_age_count = result[0]
all_age_count = np.histogram(all_ages,bins=bin_list)[0]
death_rate = (dead_age_count/all_age_count*100).round(2)
ax2t = ax2.twinx()
ax2t.plot([i+5 for i in bin_list[:-1]],death_rate,c='r',label='Death Rate (%)')
ax2t.set_ylim(0,100)
ax2t.set_ylabel('Death Rate (%)')
plt.legend(loc='upper left')
cols = ['Case no.','Report date','Date of onset','Gender','Age','HK/Non-HK resident','Case classification*','Confirmed/probable']
print(df[cols][df['Report date']==report_date.strftime('%d/%m/%Y')])
