def google_map_api(ori,dest,st,mode,path,my_key):
    from datetime import datetime 
    import googlemaps
    gmaps = googlemaps.Client(key=my_key[0])
    std= datetime.strptime(st[0], '%Y-%m-%dT%H:%M:%SZ')
    if mode[0]=="Driving":
        parameters = {
            "units": "metric",
            "origin": ori[0],
            "destination": dest[0],
            "mode": "driving"
            }
    elif mode[0]=="Walking":
        parameters = {
            "units": "metric",
            "origin": ori[0],
            "destination": dest[0],
            "mode": "walking"
            }
    else:
         parameters = {
            "units": "metric",
            "origin": ori[0],
            "destination": dest[0],
            "mode": "transit", 
            "transit_mode": "bus",
            "transit_routing_preference": "less_walking",
            "departure_time": int(datetime.timestamp(std)),
            }
    response = gmaps.directions(**parameters)
    resp = response[0]
    return path_detail(resp,path[0])

def path_detail(res,path):
    import pandas as pd
    import polyline
    import re
    import warnings
    from pandas.core.common import SettingWithCopyWarning
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    leg = res['legs'][0]
    duration = leg['duration']['text']
    steps = leg['steps']
    header = ['Step','Substep','Substep_seq','Lat','Long','Instruction','Travel_Mode','Size']
    df = pd.DataFrame(columns=header)
    tm = ''
    for i in range(len(steps)):
        step_instruction = re.sub(r'<[^>]*>', ' ',steps[i]['html_instructions'])
        tm = steps[i]['travel_mode']
        if 'steps' in steps[i].keys():
            substeps = steps[i]['steps']
            for j in range(len(substeps)):
                substep_instruction = re.sub(r'<[^>]*>', ' ',substeps[j]['html_instructions'])
                path_cor = [list(k) for k in polyline.decode(substeps[j]['polyline']['points'])]
                temp_df = pd.DataFrame(path_cor,columns=['Lat','Long'])
                temp_df['Step']=i+1
                temp_df['Substep']=j+1
                temp_df['Substep_seq']=list(temp_df.index+1)
                temp_df['Size']=1
                temp_df['Instruction']=''
                if j==0:
                    temp_df['Instruction'].iloc[0]=step_instruction
                    temp_df['Size'].iloc[0]=5
                else: 
                    temp_df['Instruction'].iloc[0]=substep_instruction
                temp_df['Travel_Mode']=tm
                df = df.append(temp_df,ignore_index=True)
        else:
            path_cor = [list(k) for k in polyline.decode(steps[i]['polyline']['points'])]
            temp_df = pd.DataFrame(path_cor,columns=['Lat','Long'])
            temp_df['Step']=i+1
            temp_df['Substep']=1
            temp_df['Substep_seq']=list(temp_df.index+1)
            temp_df['Instruction']=''
            temp_df['Size']=1
            temp_df['Travel_Mode']=tm
            if tm=='TRANSIT':
                route = steps[i]['transit_details']['line']['short_name']
                rtype = steps[i]['transit_details']['line']['vehicle']['name']
                step_instruction += ' ({})'.format(route)
                temp_df['Travel_Mode']=rtype+' ({})'.format(route)
            temp_df['Instruction'].iloc[0]=step_instruction
            temp_df['Size'].iloc[0]=5
            df = df.append(temp_df,ignore_index=True)
    df['Trip_seq']=list(df.index+1)
    df['Size'].iloc[0]=10
    df['Size'].iloc[-1]=10
    df['Remark']=''
    df['Remark'].iloc[0]='Origin'
    df['Remark'].iloc[-1]='Destination'
    df.to_csv(path,index=False)
    return duration

from tabpy.tabpy_tools.client import Client
client = Client('http://localhost:9004/')
client.deploy('Gen_path', google_map_api, 'get the detail path from orign to destination from google map API',override=True)