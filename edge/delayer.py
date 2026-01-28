import sys
from os import listdir
from os.path import isfile, join
import json
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

def load_file_data(fname, is_master=False):
    un_m = 0.00000898334580010698
    print("This is load file")
    kafka_messages = []
    dict_list = []
    print('Loading data from ', fname)
    with open(fname, 'rb') as file:
        messages = json.load(file)
    for message in messages:
        #print(f"message! = {message}")
        my_dict = json.loads(message)
        for key in my_dict.keys():
            if key in ['timestamp', 'station_lat', 'station_lon', 'conf', 'latitude', 'longitude', 'distance', 'sys_time']:
                my_dict[key] = float(my_dict[key])
            if my_dict[key]=='NaN':
                my_dict[key]='car'
        dlat = np.random.normal(loc=0.,scale=un_m)
        dlon = np.random.normal(loc=0.,scale=un_m)
        my_dict['GT latitude'] = my_dict['latitude']
        my_dict['GT longitude'] = my_dict['longitude']
        if is_master == False:
            my_dict['latitude'] += dlat
            my_dict['longitude'] += dlon
        kafka_messages.append(my_dict)
    df = pd.DataFrame(kafka_messages)
    print(F"Done. Data loaded: {len(df)} entries.")
    return df

def save_data(df,fname):
    records = df.to_dict('records')
    with open(fname, 'w') as f:
        json.dump(records, f, indent=4)
    print(f"Data saved as {fname}")

def delay_all_data(df,delay,offset=0):
    delay /= 1000
    lx = len(df)
    delays = np.random.uniform(low=0,high=delay, size=lx)
    print(f"delay={delay}")
    print(delays)
    df0 = df.copy().reset_index(drop=True)
    df0['sys_time'] = df['sys_time'] + delays
    df_sorted = df0.sort_values(by='sys_time',ascending=True).reset_index(drop=True)
    return df_sorted


def delay_data(df,stations,delays,dstd=0.1):
    
    #df['sys_time'] = pd.to_datetime(df['sys_time'])
    print(df['sys_time'].dtype)
    #df['sys_time'] = df['sys_time'].astype(float)
    #Separating stations in differented dataframes:
    df_dict = dict()
    dfs = []
    all_stations = df['station_id'].unique()
    for station in all_stations:
        df_dict[station] = df[df['station_id'] == station]
    #Delaying selected stations:
    for station, delay in zip(stations, delays):
        #print(f"Station: {station}, Delay: {delay}")
        #print("df before")
        #print(df_dict[station])
        df2 = df_dict[station].copy()
        #lx = len(df2)
        #my_std = dstd*delay
        #delays = np.abs(np.random.normal(delay,my_std,lx))
        #df2.loc[:,'sys_time'] = df2.loc[:,'sys_time'] + delays
        df2.loc[:,'sys_time'] = df2.loc[:,'sys_time'] + delay
        #print(df_dict[station]['sys_time']-df2['sys_time'])
        df_dict[station] = df2
        #print(df_dict)
        print("*************************************")
    #concatenation of dataframes
    for key in df_dict.keys():
        print(key)
        dfs.append(df_dict[key])
    print(f"I have {len(dfs)} stations")
    df_delayed = pd.concat(dfs, ignore_index=True)
    df_sorted = df_delayed.sort_values(by='sys_time',ascending=True)
    #That is all!   
    return df_sorted
    #for station, delay in zip(stations,delays):
    #    #delay = pd.Timedelta(delay)
    #    print(f"stations is {station}")
    #    print(f"delay is {delay}")
    #    print(df.loc[df['station_id'] == station])
    #    mask = df['station_id'] == station
    #    df.loc[mask, 'sys_time'] += delay
    #dfs = []
    #df_delayed = df.sort_values(by='sys_time')
    ##df_delayed['sys_time'] = df['sys_time'].apply(lambda x: x.timestamp())
    #print(df_delayed.loc[df_delayed['station_id'] == station])
    #return df_delayed

def main():
    args = sys.argv
    fname = args[1]
    path = args[2]
    station = args[3]
    delay = float(args[4]) #ms
    dl_str = args[4] #ms
    master = args[5]
    all_files = False
    if  args[6] == 'yes':
        all_files = True
    is_master = False
    if master == 'is-master':
        is_master = True
    #nstats = int(args[2])
    #stations = []
    #delays = []
    #offset = 3
    #for i in range(offset,offset+nstats):
    #    stations.append(int(args[i]))
    #    delays.append(float(args[i+nstats])*1000) #ns
    my_files = []
    my_fnames = []
    if all_files is True:
        my_files = [f for f in listdir(path+'jsons/') if isfile(join(path+'jsons/', f))]
    else:
        my_files = [fname]
    for fname in my_files:
        name, ext = fname.rsplit('.json',1)
        my_fnames.append(name)
    print(f"my files are {my_fnames}")   
    for fname in my_fnames:
        lname = fname + '.json'
        sname = None
        if is_master == False:
            sname = fname +'delay_'+str(int(delay))+'_ms'+'.json'
        else:
            sname = fname +'delay_'+str(int(delay))+'_ms_master'+'.json'
        print("-------------Summary---------------")
        print(f"filename: {fname}")
        print(f"input filename: {lname}")
        print(f"output filename: {sname}")
        print(f"stations: {station}")
        print(f"delays: {delay}")
        df = load_file_data(path+'/jsons/'+lname,is_master=is_master)
        fnamecsv = None
        if is_master == False:
            fnamecsv =  lname.rsplit('.', 1)[0] + '.csv'
        else:
            fnamecsv =  lname.rsplit('.', 1)[0] + '_master.csv'
        df.to_csv(fnamecsv, index=False)
        #if delay > 0:
        df_delayed = delay_all_data(df,delay)
        #df_delayed = delay_data(df,[station],[delay])
        save_data(df_delayed,path+sname)
        snamecsv =  sname.rsplit('.', 1)[0] + '.csv'
        snamecsv = path + 'csvs/' + snamecsv
        df_delayed.to_csv(snamecsv, index=False)
        pd.set_option('display.float_format', '{:.9f}'.format)
        print(df.head(20))
        print(df_delayed.head(20))
        print(f"Data saved on file {snamecsv}")
        print(f"Entries: {len(df_delayed)}")

    print("That is all!") 

if __name__ == '__main__':
    main()