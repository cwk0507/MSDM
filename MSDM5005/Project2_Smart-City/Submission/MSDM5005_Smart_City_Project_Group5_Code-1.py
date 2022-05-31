def smart_journey(ori,dest,st,path,my_key,bus_path,mode):
    import pandas as pd
    import polyline
    import warnings
    from pandas.core.common import SettingWithCopyWarning
    warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
    
    def google_map_api_jp(ori,dest,st,mode,my_key):
        from datetime import datetime 
        import googlemaps
        import polyline
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
        elif mode[0]=="Bus":
            parameters = {
                "units": "metric",
                "origin": ori[0],
                "destination": dest[0],
                "mode": "transit", 
                "transit_mode": "bus",
                "transit_routing_preference": "fewer_transfers",
                "departure_time": int(datetime.timestamp(std)),
                }
        else:
             parameters = {
                "units": "metric",
                "origin": ori[0],
                "destination": dest[0],
                "mode": "transit", 
                "transit_routing_preference": "fewer_transfers",
                "departure_time": int(datetime.timestamp(std)),
                }
        response = gmaps.directions(**parameters)
        full_path = [list(k) for k in polyline.decode(response[0]['overview_polyline']['points'])]
        distance = response[0]['legs'][0]['distance']['value']
        duration = response[0]['legs'][0]['duration']['value']
        return response[0], full_path, distance, duration
    r0,p0,d0,t0 = google_map_api_jp(ori,dest,st,['Walking'],my_key)
    O = [r0['legs'][0]['start_location']['lat'],r0['legs'][0]['start_location']['lng']]
    D = [r0['legs'][0]['end_location']['lat'],r0['legs'][0]['end_location']['lng']]
    s_add = r0['legs'][0]['start_address']
    e_add = r0['legs'][0]['end_address'] 
    header = ['Route','Step','Lat','Long','Travel_Mode','distance','duration','Remark']   
    if mode[0]=='Scooting only':
        substeps = r0['legs'][0]['steps']
        s=1
        df = pd.DataFrame(columns=header)
        for j in range(len(substeps)):
            sub_path = [list(k) for k in polyline.decode(substeps[j]['polyline']['points'])]
            sub_distance = substeps[j]['distance']['value']
            sub_duration = substeps[j]['duration']['value']
            try:
                maneuver = substeps[j]['maneuver']
            except:
                maneuver = 'others'
            temp_sub_df = pd.DataFrame(sub_path,columns=['Lat','Long'])
            temp_sub_df['Step']=s
            temp_sub_df['distance'] = 0
            temp_sub_df['distance'][temp_sub_df.index==0] = sub_distance
            temp_sub_df['duration'] = 0
            temp_sub_df['Remark'] = ''  
            if maneuver == 'ferry':
                s+=0.5
                temp_sub_df['Step']=s
                temp_sub_df['Travel_Mode']='Ferry'
                temp_sub_df['duration'][temp_sub_df.index==0] = sub_duration
            else:
                temp_sub_df['Travel_Mode']='Scooting'
                temp_sub_df['duration'][temp_sub_df.index==0] = sub_duration/4
            df = df.append(temp_sub_df,ignore_index=True)
        df.index+=1
        df.reset_index(inplace=True)
        df['Remark'][df.index==0]='Origin: '+s_add
        df['Remark'][df.index==max(df.index)]='Destination: '+e_add             
        df.to_csv(path[0],index=False)
        return 'Done'

    r0,p0,d0,t0 = google_map_api_jp(ori,dest,st,['Transit'],my_key)
    steps = r0['legs'][0]['steps']
    df = pd.DataFrame(columns=header)
    for i in range(len(steps)):
        tm = steps[i]['travel_mode'] 
        path_cor = [list(k) for k in polyline.decode(steps[i]['polyline']['points'])]
        temp_df = pd.DataFrame(path_cor,columns=['Lat','Long'])
        temp_df['Route'] = 'Google'
        temp_df['Step']=i+1
        temp_df['distance'] = 0
        temp_df['distance'][temp_df.index==0] = steps[i]['distance']['value']
        temp_df['duration'] = 0
        temp_df['Remark'] = ''
        if tm=='WALKING':
            temp_df['duration'][temp_df.index==0] = steps[i]['duration']['value']/4
            temp_df['Travel_Mode']='Scooting'
        else:
            rtype = steps[i]['transit_details']['line']['vehicle']['name']
            if rtype=='Bus':
                route = steps[i]['transit_details']['line']['short_name']
            else:
                route = steps[i]['transit_details']['line']['name']
            temp_df['duration'][temp_df.index==0] = steps[i]['duration']['value']
            temp_df['Travel_Mode']=rtype+' ({})'.format(route)
            temp_df['Remark'][temp_df.index==0]=steps[i]['transit_details']['departure_stop']['name']
            temp_df['Remark'][temp_df.index==max(temp_df.index)]=steps[i]['transit_details']['arrival_stop']['name']
        df = df.append(temp_df,ignore_index=True)
    df.index+=1
    df.reset_index(inplace=True)
    df['Remark'][df.index==0]='Origin: '+s_add
    df['Remark'][df.index==max(df.index)]='Destination: '+e_add
    # stops = pd.read_csv('https://static.data.gov.hk/td/pt-headway-tc/stops.txt')
    # routes = pd.read_csv('https://static.data.gov.hk/td/pt-headway-tc/routes.txt')
    # stop_times = pd.read_csv('https://static.data.gov.hk/td/pt-headway-tc/stop_times.txt')
    stops = pd.read_csv(bus_path[0]+'\\stops.csv')
    routes = pd.read_csv(bus_path[0]+'\\routes.csv')
    stop_times = pd.read_csv(bus_path[0]+'\\stop_times.csv')
    on = min(3,d0/4000) / 106.8
    off = min(3,d0/4000) / 106.8

    # check if stops are within the defined radius from Origin 
    stop = (((stops.stop_lon-O[1])**2+(stops.stop_lat-O[0])**2)**0.5 < on)
    # select all records in stop_times with stop that are within the defined radius from Origin
    select = stop_times.stop_id.isin(stops.stop_id[stop])
    # select all trip_ids of routes that have at least one stop within the defined radius from Origin
    # and at least one stop within the definded radius from Destination
    Select = stop_times.trip_id[select].isin(stop_times.trip_id[stop_times.stop_id.isin(stops.stop_id[
              ((stops.stop_lon-D[1])**2+(stops.stop_lat-D[0])**2)**0.5 < off])])
    
    route_id = [] 
    route_name = []
    target = {}
    for trip_id in stop_times.trip_id[select][Select].unique():
        tid = trip_id[0:trip_id.find('-',trip_id.find('-')+1)]
        if tid not in route_id:
            temp_df = pd.DataFrame(columns=header)
            route_id += [tid]
            trip = stop_times[stop_times.trip_id == trip_id][['stop_id','stop_sequence']].merge(stops) # generate a detail list of stops of the route
            d = ((trip.stop_lon-D[1])**2+(trip.stop_lat-D[0])**2)**0.5 < off # check if stops of the route within the defined radius from Destination
            o = ((trip.stop_lon-O[1])**2+(trip.stop_lat-O[0])**2)**0.5 < on # check if stops of the route within the defined radius from Origin
            dmin = min(d[d].index) # the first stop of the route within the defined radius from the Destination
            omax = max(o[o].index) # the last stop of the route within the defined radius from the Origin
            r_name = routes.route_short_name[routes.route_id == int(trip_id[0:trip_id.find('-')])].values[0]
            if (dmin>=omax) and r_name not in route_name:
                target[tid] = {'routes': r_name+' ('+str(dmin-omax)+' stops)',
                               'details': trip,
                               'stops' : dmin-omax,
                               'origin_stop': (trip.stop_lat.loc[omax],trip.stop_lon.loc[omax]),
                               'destination_stop': (trip.stop_lat.loc[dmin],trip.stop_lon.loc[dmin]),
                               'origin_stop_name':trip.stop_name.loc[omax],
                               'destination_stop_name':trip.stop_name.loc[dmin]
                               }
                target[tid]['steps']=[]
                # 1st step: from origin to target departure bus stop, change walking to scooting and travelling time divided by 4
                r1,p1,d1,t1 = google_map_api_jp([str(O[0])+','+str(O[1])],[target[tid]['origin_stop']],st,['Walking'],my_key)
                sub_df = pd.DataFrame(columns=header)
                substeps = r1['legs'][0]['steps']
                s=1
                for j in range(len(substeps)):
                    sub_path = [list(k) for k in polyline.decode(substeps[j]['polyline']['points'])]
                    sub_distance = substeps[j]['distance']['value']
                    sub_duration = substeps[j]['duration']['value']
                    try:
                        maneuver = substeps[j]['maneuver']
                    except:
                        maneuver = 'others'
                    temp_sub_df = pd.DataFrame(sub_path,columns=['Lat','Long'])
                    temp_sub_df['Step']=s
                    temp_sub_df['distance'] = 0
                    temp_sub_df['distance'][temp_sub_df.index==0] = sub_distance
                    temp_sub_df['duration'] = 0
                    temp_sub_df['Remark'] = ''  
                    if maneuver == 'ferry':
                        s+=0.5
                        temp_sub_df['Step']=s
                        temp_sub_df['Travel_Mode']='Ferry'
                        temp_sub_df['duration'][temp_sub_df.index==0] = sub_duration
                    else:
                        temp_sub_df['Travel_Mode']='Scooting'
                        temp_sub_df['duration'][temp_sub_df.index==0] = sub_duration/4
                    sub_df = sub_df.append(temp_sub_df,ignore_index=True)
                target[tid]['steps'].append({'Step':1,'df':sub_df})
                temp_df = temp_df.append(sub_df)
                # 2nd step: from target departure bus stop to target arrival bus stop
                r2,p2,d2,t2 = google_map_api_jp([target[tid]['origin_stop']],[target[tid]['destination_stop']],st,['Bus'],my_key)
                match = True
                for k in range(len(r2['legs'][0]['steps'])):
                    if r2['legs'][0]['steps'][k]['travel_mode']!='WALKING':
                        if r2['legs'][0]['steps'][k]['transit_details']['line']['short_name']!=r_name:
                            match = False
                            break
                        else:
                            break
                if not match:
                    continue
                target[tid]['steps'].append({'Step':2,'Lat': [k[0] for k in p2], 'Long':[k[1] for k in p2],'Travel_Mode':'Bus','distance':d2,'duration':t2,'Remark':''})
                bus_df = pd.DataFrame(target[tid]['steps'][1])
                bus_df['distance'][bus_df.index!=0]=0
                bus_df['duration'][bus_df.index!=0]=0
                bus_df['Remark'][bus_df.index==0]=target[tid]['origin_stop_name']
                bus_df['Remark'][bus_df.index==max(bus_df.index)]=target[tid]['destination_stop_name']
                temp_df = temp_df.append(bus_df)
                # 3rd step: from target arrival bus stop to destination, change walking to scooting and travelling time divided by 4
                r3,p3,d3,t3 = google_map_api_jp([target[tid]['destination_stop']],[str(D[0])+','+str(D[1])],st,['Walking'],my_key)
                sub_df = pd.DataFrame(columns=header)
                substeps = r3['legs'][0]['steps']
                s=3
                for j in range(len(substeps)):
                    sub_path = [list(k) for k in polyline.decode(substeps[j]['polyline']['points'])]
                    sub_distance = substeps[j]['distance']['value']
                    sub_duration = substeps[j]['duration']['value']
                    try:
                        maneuver = substeps[j]['maneuver']
                    except:
                        maneuver = 'others'
                    temp_sub_df = pd.DataFrame(sub_path,columns=['Lat','Long'])
                    temp_sub_df['Step']=s
                    temp_sub_df['distance'] = 0
                    temp_sub_df['distance'][temp_sub_df.index==0] = sub_distance
                    temp_sub_df['duration'] = 0
                    temp_sub_df['Remark'] = ''  
                    if maneuver == 'ferry':
                        s+=0.5
                        temp_sub_df['Step']=s
                        temp_sub_df['Travel_Mode']='Ferry'
                        temp_sub_df['duration'][temp_sub_df.index==0] = sub_duration
                    else:
                        temp_sub_df['Travel_Mode']='Scooting'
                        temp_sub_df['duration'][temp_sub_df.index==0] = sub_duration/4
                    sub_df = sub_df.append(temp_sub_df,ignore_index=True)
                target[tid]['steps'].append({'Step':3,'df':sub_df})
                temp_df = temp_df.append(sub_df)
        
                temp_df['Route']=target[tid]['routes']
                temp_df.reset_index(inplace=True,drop=True)
                temp_df.index+=1
                temp_df.reset_index(inplace=True)
                temp_df['Remark'][temp_df.index==0]='Origin: '+s_add
                temp_df['Remark'][temp_df.index==max(temp_df.index)]='Destination: '+e_add             
                df = df.append(temp_df,ignore_index=True)
                route_name.append(r_name)
    df.to_csv(path[0],index=False)
    return 'Done'
            
from tabpy.tabpy_tools.client import Client
client = Client('http://localhost:9004/')
client.deploy('scootxbus', smart_journey, 'suggested routes using e-scooters',override=True)            