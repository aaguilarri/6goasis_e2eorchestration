import json 
import numpy as np
from numpy.random import uniform
import time
import random
import sys
import os
import re
from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


import pickle
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import threading
from pfilt import Pfilter
from trajectory import Trajectory
import subprocess

kafka_messages = []
discont = 0
colors = [
    'black', 'blue', 'blueviolet', 'brown', 'cadetblue', 'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 
    'crimson', 'cyan', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta',
    'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 
    'darkslategray', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dodgerblue', 
    'firebrick', 'forestgreen', 'fuchsia', 'gold', 'goldenrod', 'gray', 'green', 'greenyellow', 'honeydew', 
    'hotpink', 'indianred', 'indigo', 'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 
    'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightpink', 
    'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 'lightsteelblue', 
    'lime', 'limegreen', 'magenta', 'maroon', 'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 
    'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 
    'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'olive', 'olivedrab', 'orange', 'orangered', 
    'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 'papayawhip', 'peachpuff', 
    'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue', 'saddlebrown', 
    'salmon', 'sandybrown', 'seagreen', 'sienna', 'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 
    'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'yellow', 
    'yellowgreen'
]

def create_consumer(server,port,topic_name):
    print(f"server {server} port {port} topic {topic_name}")
    bootstrap_servers=server+':'+port
    return KafkaConsumer(
        topic_name,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='latest',
        value_deserializer=lambda x: x.decode('utf-8')
    )

def get_the_df(df):
    global kafka_messages
    if len(kafka_messages)<=0:
        return df
    tracklet = kafka_messages.pop(0)
    print(f"tracklet is {tracklet}")
    if df is None:
        df = pd.DataFrame([tracklet])
    else:
        new_row = pd.DataFrame([tracklet])
        df = pd.concat([df,new_row], ignore_index=True)
    return df

def consume_messages(consumer):
    global kafka_messages
    for i, message in enumerate(consumer):
        print("Houston, we got a message!")
        try:
            # Check if message.value is already a string
            if isinstance(message.value, str):
                json_str = message.value
            else:
                json_str = message.value.decode('utf-8')
            
            tracklet = json.loads(json_str)
            for item in tracklet.keys():
                if item in ['station_lat', 'station_lon', 'conf', 'latitude', 'longitude', 'distance']:
                    tracklet[item] = float(tracklet[item])
                elif item in ['timestamp']:
                    tracklet[item] = int((float(tracklet[item]) * 1000) // 1)
                elif item in ['class']:
                    tracklet[item] = int(float(tracklet[item]) // 1)
            print(f"Received tracklet from agent {tracklet['station_id']}")
            if tracklet['station_id'] != '0':
                kafka_messages.append(tracklet)
                print(f" we have {len(kafka_messages)} messages")
        except json.JSONDecodeError:
            print(f"Failed to decode message: {message.value}")
        except KeyError as e:
            print(f"Missing key in tracklet: {e}")
        except ValueError as e:
            print(f"Value conversion error: {e}")
        except AttributeError as e:
            print(f"Unexpected message format: {e}")

#def consume_messages(consumer):
#    global kafka_messages
#    for i, message in enumerate(consumer):
#        print("Houston, we got a message!")
#        tracklet = json.loads(message.value.decode('utf-8'))
#        #tracklet = json.loads(message.payload)
#        for item in tracklet.keys():
#            if item in ['station_lat', 'station_lon', 'conf', 'latitude', 'longitude', 'distance']:
#                tracklet[item] = float(tracklet[item])
#            if item in ['timestamp']:
#                tracklet[item] = int((float(tracklet[item]) *1000) // 1)
#            if item in ['class']:
#                tracklet[item] = int(float(tracklet[item])//1)
#        print(f"Received tracklet from agent {tracklet['station_id']}")
#        if tracklet['station_id'] != '0':
#            kafka_messages.append(tracklet)


def haversine(lat1, lon1, lat2, lon2):
    #print(lat1)
    #print(lon1)
    #print(lat2)
    #print(lon2)
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    radius = 6371000  # approximately 6,371 km
    distance = radius * c
    return distance

def load_file_data(fname):
    print("This is load file")
    global kafka_messages
    global colors
    kafka_messages = []
    dict_list = []
    print('Loading data from ', fname)
    with open(fname, 'r') as f:
        data = json.load(f)
    #cont = 0
    for my_dict in data:
        #print(my_dict)
        if type(my_dict) == str:
            my_dict = json.loads(my_dict)
        for key in my_dict.keys():
            if key in ['timestamp', 'station_lat', 'station_lon', 'conf', 'latitude', 'longitude', 'distance']:
                my_dict[key] = float(my_dict[key])
            if my_dict[key]=='NaN':
                my_dict[key]='car'
        my_dict['color'] = None
        my_dict['PF latitude'] = 0
        my_dict['PF dlat']= 0
        my_dict['PF longitude'] = 0
        my_dict['PF dlon'] = 0
        my_dict['risk'] = 0.
        my_dict['filter'] = 0
        my_dict['last tick'] = 0.
    #with open(fname, 'rb') as file:
    #    messages = json.load(file)
    #for message in messages:
    #    #print(f"message! = {message}")
    #    my_dict = json.loads(message)
    #    for key in my_dict.keys():
    #        if key in ['timestamp', 'station_lat', 'station_lon', 'conf', 'latitude', 'longitude', 'distance']:
    #            my_dict[key] = float(my_dict[key])
    #        if my_dict[key]=='NaN':
    #            my_dict[key]='car'
        
        #if my_dict['station_id'] in ['9999'] or (my_dict['station_id'] in ['1111'] and my_dict['ID'] in ['1111']):
        #    kafka_messages.append(my_dict)
        #    print(".",end="")
    #if my_dict['station_id'] in ['1111', '5555', '9999']:
        kafka_messages.append(my_dict)
        print(".",end="")
        #if cont > 100:
        #    break
        #cont += 1
        #kafka_messages.append(my_dict)
        #print(".",end="")
    print("")
    print('Data loaded from ', fname)
    print(f"Number of entries: {len(kafka_messages)}")
    df = pd.DataFrame(kafka_messages)
    #print(df)
    #exit()

def compute_risk(rx,ry,dt=1):
    #getting positon and sped information from first agent
    lat0 = rx[0]
    lon0 = rx[2]
    dlat0 = rx[1]
    dlon0 = rx[3]
    #print(f"(lat0,lon0) = ({lat0},{lon0}) + ({dlat0}, {dlon0})")
    lat0a = lat0 + dlat0*dt
    lon0a = lon0 + dlon0*dt
    #print(f"(lat0a,lon0a) = ({lat0a},{lon0a})")
    #Speed of vehicle in m/s
    mvv = haversine(lat0a, lon0a, lat0, lon0) / dt 
    #print(f" mvv is {mvv}")
    vvv = np.array([lat0a - lat0, lon0a - lon0])
    nmv = np.linalg.norm(vvv)
    #print(f"vvv is {vvv}")
    #print(f"vvv mag is {nmv}")
    vv = vvv / nmv
    vv = mvv * vv
    #print(f"vv is {vv}")
    #The same for the second one
    lat1 = ry[0]
    lon1 = ry[2]
    dlat1 = ry[1]
    dlon1 = ry[3] 
    #print(f"(lat1,lon1) = ({lat1},{lon1}) + ({dlat1}, {dlon1})")
    lat1a = lat1 + dlat1*dt
    lon1a = lon1 + dlon1*dt
    #print(f"(lat1a,lon1a) = ({lat1a},{lon1a})")
    #Speed of object in m/s
    mvo = haversine(lat1a, lon1a, lat1, lon1) / dt
    #print(f" mvo is {mvo}")
    vvo = np.array([lat1a - lat1, lon1a - lon1])
    nmo = np.linalg.norm(vvo)
    vo = vvo / nmo
    vo = mvo * vo
    #print(f"vo is {vo}")
    #Separation distance between vehicle and object (m/s)
    dsep = haversine(lat0,lon0,lat1,lon1) 
    #unitary direction vector between vehicle and object
    pv = np.array([lat0, lon0])
    po = np.array([lat1, lon1])
    #print(f"pv is {pv}, po is {po}")
    mdsep = np.linalg.norm(po-pv)
    udsep =  (po - pv) / mdsep
    #print(f"vv is size {vv.shape}")
    #print(f"udsep is size {udsep.shape}")
    msv = (vv.T @ udsep).item()
    sv = msv * udsep.T
    mso = (vo.T @ udsep).item()
    so = mso * udsep.T
    srel = sv - so
    msrel = (srel @ srel.T).item()
    signo = np.sign(msv - mso)
    #print(f"srel is {srel}")
    #print(f"msv = {msv}, mso = {mso}")
    risk = msrel * signo
    is_risk = False
    if risk < 0 and np.abs(dsep / risk) < 4:
        is_risk = True
    #print(f"risk is {risk} m/s2")
    return risk, is_risk


def is_hazard(filter_bank, dt=4, eps=0.0000000001):
    #matrix to store risk estimations
    lh = len(filter_bank.keys())
    if lh < 2:
        return 0
    hazard_matrix = np.zeros([lh,lh])
    crows = 0
    hazards = []
    for id in filter_bank.keys():
        pfi = filter_bank[id]
        denshii = pfi.denshi
        xi0 = pfi.estimate_pos(denshii)
        denshii = pfi.predict(denshii,dt)
        xi = pfi.estimate_pos(denshii)
        ccols = 0
        for jd in filter_bank.keys():
            if hazard_matrix[crows,ccols] != 0:
                continue
            pfj = filter_bank[jd]
            denshij = pfj.denshi
            xj0 = pfj.estimate_pos(denshij)
            denshij = pfj.predict(denshij,dt)
            xj = pfj.estimate_pos(denshij)
            risk = 0.
            riskm = 0.
            is_risk = False
            is_riskm = False
            if id != jd:
                risk, is_risk = compute_risk(xi,xj)
                riskm, is_riskm  = compute_risk(xj,xi)
            if is_risk == True:
                hazards.append([[xi0[0].item(),xi0[2].item()], [xj0[0].item(),xj0[2].item()]])
            #print(f"risk = {risk}, riskm = {riskm}")
            hazard_matrix[crows, ccols] = risk
            hazard_matrix[ccols,crows] = riskm
            ccols += 1
        my_risk = 0.
        pos_risk = hazard_matrix[crows,:].min()
        if np.isnan(pos_risk) == False:
            my_risk = pos_risk
        #the_tck['risk'] = hazard_matrix[crows,:].max()
        #tck_bank[id] = [the_tck]
        crows += 1
    return my_risk
    #return hazard_matrix, hazards, tck_bank

    

def edge_loop(filter_bank, tracklets, n=1000, eps=300, discarding=False):
    global colors
    global discont
    print("------------------------------------------------")
    print("This is edge_loop()")
    if len(tracklets) == 0:
        print("No data...")
        return filter_bank, tracklets
    if len(filter_bank) == 0:
        print("Initializing filter bank")
        lat = tracklets.iloc[-1]['latitude']
        lon = tracklets.iloc[-1]['longitude']
        tracklets.loc[tracklets.index[-1], 'PF latitude'] = lat
        tracklets.loc[tracklets.index[-1], 'PF dlat'] = 0
        tracklets.loc[tracklets.index[-1], 'PF longitude'] = lon
        tracklets.loc[tracklets.index[-1], 'PF dlon'] = 0
        tracklets.loc[tracklets.index[-1], 'color'] = random.choice(colors)
        tracklets.loc[tracklets.index[-1], 'risk'] = 0
        color = tracklets.loc[tracklets.index[-1], 'color']
        pf = Pfilter(n, lat, lon, tracklets.iloc[-1]['timestamp'],color) # last tick to keep track of time
        pfid = pf.id
        filter_bank[pfid] = pf
        tracklets.loc[tracklets.index[-1], 'filter'] = pfid
        tracklets.loc[tracklets.index[-1], 'last tick'] = tracklets.loc[tracklets.index[-1], 'timestamp']
        #print(tracklets)
        return filter_bank, tracklets
    my_id = 0
    max_lkl = 0
    t1 = tracklets.loc[tracklets.index[-1], 'timestamp']
    #print(tracklets.tail(3))
    #print(f"t1 = {t1}")
    #checking if tracklet is time valid
    #print(f"discarding is {discarding}")
    t0 = tracklets.iloc[:-1]['timestamp'].max()
    dt = t1 - t0
    #print(f"Maximum last timestamp: {t0}")
    #print(f"difference time is {t1} - {t0} = {dt}")
    if dt <= 0: 
        print("Tracklet no time valid! No filter assigned")
        print(".")
        print(".")
        discont += 1
        lat = tracklets.iloc[-1]['latitude']
        lon = tracklets.iloc[-1]['longitude']
        tracklets.loc[tracklets.index[-1], 'PF latitude'] = lat
        tracklets.loc[tracklets.index[-1], 'PF dlat'] = 0
        tracklets.loc[tracklets.index[-1], 'PF longitude'] = lon
        tracklets.loc[tracklets.index[-1], 'PF dlon'] = 0
        tracklets.loc[tracklets.index[-1], 'color'] = random.choice(colors)
        tracklets.loc[tracklets.index[-1], 'risk'] = 0
        tracklets.loc[tracklets.index[-1], 'filter'] = 0
        tracklets.loc[tracklets.index[-1], 'last tick'] = tracklets.loc[tracklets.index[-1], 'timestamp']
        return filter_bank, tracklets
    # finding best match
    print(f"Looking for a match")
    my_t0 = 0
    for id in filter_bank.keys():
        pf = filter_bank[id]
        t00 = filter_bank[id].last_tck
        [lat0, dlat0, lon0, dlon0, w0] = pf.estimate_pos(pf.denshi)
        lat0 = lat0[0]
        lon0 = lon0[0]
        lat = tracklets.iloc[-1]['latitude']
        lon = tracklets.iloc[-1]['longitude']
        #t1 = tracklets.iloc[-1]['timestamp']
        #if t1-t0 <= 0:
        #    continue
        lkl = pf.likelihood ([lat,lon],[lat0,lon0],t1-t00)
        if lkl > max_lkl:
            my_id = id
            max_lkl = lkl
            my_t0 = t00
        #print(f"class:{tracklets.iloc[-1]['class']}, filter id: {id}, lkl={lkl}, my_id={my_id}, max-lkl={max_lkl}")
    # checkinf whether matching was found
    #print(f"The id selected was {my_id}")
    #print(f"is {max_lkl} < {eps} ?")
    if max_lkl > eps:
        my_pf = filter_bank[my_id]
        print(f"Yes! Matching filter found! ID={my_id}, lkl={max_lkl}")
        lat = tracklets.iloc[-1]['latitude']
        lon = tracklets.iloc[-1]['longitude']
        dt = t1 - filter_bank[my_id].last_tck
        denshi = filter_bank[my_id].filtering([lat,lon],dt)
        tm1 = filter_bank[my_id].last_tck
        filter_bank[my_id].last_tck = t1
        filter_bank[my_id].denshi = denshi
        pos = filter_bank[my_id].estimate_pos(filter_bank[my_id].denshi)
        tracklets.loc[tracklets.index[-1], 'PF latitude'] = pos[0][0]
        tracklets.loc[tracklets.index[-1], 'PF dlat'] = pos[1][0]
        tracklets.loc[tracklets.index[-1], 'PF longitude'] = pos[2][0]
        tracklets.loc[tracklets.index[-1], 'PF dlon'] = pos[3][0]
        tracklets.loc[tracklets.index[-1], 'color'] = filter_bank[my_id].color
        tracklets.loc[tracklets.index[-1], 'filter'] = filter_bank[my_id].id
        tracklets.loc[tracklets.index[-1], 'last tick'] = tm1
    else:
        print("No! No matching filter. Creating new trajectory")
        lat = tracklets.iloc[-1]['latitude']
        lon = tracklets.iloc[-1]['longitude']
        tracklets.loc[tracklets.index[-1], 'PF latitude'] = lat
        tracklets.loc[tracklets.index[-1], 'PF dlat'] = 0
        tracklets.loc[tracklets.index[-1], 'PF longitude'] = lon
        tracklets.loc[tracklets.index[-1], 'PF dlon'] = 0
        tracklets.loc[tracklets.index[-1], 'color'] = random.choice(colors)
        tracklets.loc[tracklets.index[-1], 'risk'] = 0
        color = tracklets.loc[tracklets.index[-1], 'color']
        pf = Pfilter(n, lat, lon, tracklets.iloc[-1]['timestamp'], color) # last tick to keep track of time
        pfid = pf.id
        filter_bank[pfid] = pf
        tracklets.loc[tracklets.index[-1], 'filter'] = pfid
        tracklets.loc[tracklets.index[-1], 'last tick'] = tracklets.iloc[-1]['timestamp']
    print("..................................................")
    #print(tracklets)
    return filter_bank, tracklets


    



def get_classes(i):
    #[0,1,2,3,5,7,9,10,11,12]
    yolo_dict={0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39:'bottle', 40:'wine glass', 41:'cup', 42:'fork', 43:'knife', 44:'spoon', 45:'bowl', 46:'banana', 47:'apple', 48:'sandwich', 49:'orange', 50:'broccoli', 51:'carrot', 52:'hot dog', 53:'pizza', 54:'donut', 55:'cake', 56:'chair', 57:'couch', 58:'potted plant', 59:'bed', 60:'dining table', 61:'toilet', 62:'tv', 63:'laptop', 64:'mouse', 65:'remote', 66:'keyboard', 67:'cell phone', 68:'microwave', 69:'oven', 70:'toaster', 71:'sink', 72:'refrigerator', 73:'book', 74:'clock', 75:'vase', 76:'scissors', 77:'teddy bear', 78:'hair drier', 79:'toothbrush'}
    return yolo_dict[i]

#def show_plot():
#    plt.show()

def main():
    my_columns = ['timestamp','station_id','station_lat','station_lon','raw_lat','raw_lon','latitude','longitude','tracking_ID','class',
    'confidence','dt','speed','haversine','dt1','lat1','dlat1','lon1','dlon1','color']

    t_start = time.time()
    tsave = str(int(time.time()*1**7))
    st_list = []
    t_max = 0.01 #s to stop service
    args = sys.argv
    filter_bank = dict()
    tck_bank = dict()
    df_bank = dict()
    is_kafka = False
    nn = 100
    lkl_edge = 300
    the_df = None
    discarding = False
    is_master = False
    print("Starting summary:")
    print(f"IP:{args[1]}")
    print(f"port:{args[2]}")
    print(f"file name:{args[3]}")
    print(f"kafka?:{args[4]}")
    print(f"Path:{args[5]}")
    print(f"Delay:{args[6]}")
    if len(args) > 7:
        print(f"N Particles: {args[7]}")
        nn = int(args[7])
    if len(args)>8:
        print(f"Lkl Edge: {args[8]}")
        lkl_edge = int(args[8])
    if len(args)>9:
        print(f"Discariding: {args[9]}")
        if args[9] == 'discarding':
            discarding = True
    fpath = args[5]
    if args[4] == "kafka":  #use kafka or load from file
        is_kafka = True
    client = None
    ip = args[1] #"10.1.24.50"  #kafka ip address
    port = args[2] #kafka port
    lfname = args[3] #'oasis9999.json'  #file name
    fname = 'edgepf_'+str(int(time.time()*1000))+'.pkl'
    my_topics = ["tracklets"]
    tx_delay = 0
    if len(args) > 5:
        tx_delay = float(args[6])  #delay to simulate transmission delay
    print(f"kafka is {is_kafka}")
    if is_kafka is True:
        print("Therefore, we start Kafka client!")
        the_df = pd.DataFrame(columns=my_columns)
        consumer = create_consumer(ip,port,my_topics[0])
    else:
        print("Therefore, we load data from file!")
        load_file_data(fpath+lfname)
        if 'master' in lfname:
            is_master = True
        print("Done")
        #exit()
    print("Player 1, welcome to Edge Service")
    plt.style.use('dark_background')
    t0 = time.time()
    ids_list = []
    hazards = []
    global kafka_messages
    df = pd.DataFrame(kafka_messages)
    if is_kafka == False:
        fname =fpath+'raw'+tsave+'.csv'
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        df.to_csv(fname)
        print(f"Raw data saved as {fname}")
    n_filts = 0
    car_filts = 0
    try:
        while True:
            if is_kafka:
                print("trying consume messages...")
                consume_messages(consumer)
            the_df = get_the_df(the_df)
            filter_bank, the_df = edge_loop(filter_bank, the_df, n=nn, eps=lkl_edge, discarding=discarding)
            #print(f"tck bank: {tck_bank}")
            print("Current State:")
            print(the_df.tail(10).to_string(float_format='{:.3f}'.format))
            print(f"Total lines: {len(the_df)}")
            #if len(the_df) > 730:
            #    break     
            print(f"Messages left: {len(kafka_messages)}")
            #If there is filtering we compute risk, othewise we copy the last one
            if int(the_df.loc[the_df.index[-1], 'filter']) > 0:
                risk = is_hazard(filter_bank)
                the_df.loc[the_df.index[-1], 'risk'] = risk
            elif len(the_df) > 1:
                risk = 0. #the_df.loc[the_df.index[-2], 'risk']
                the_df.loc[the_df.index[-1], 'risk'] = risk
            #hazards = hazards + hazards_t
            if len(kafka_messages) > 0:
                t0 = time.time()
            elif time.time() - t0 > t_max:
                print("No more data!")
                car_df = the_df[the_df['class_str']=='car_hero_camera']
                n_filts = len(car_df['filter'].unique())
                break
            #hazard_matrix = is_hazard(filter_bank, tck_bank)
    except KeyboardInterrupt:
            print("Closing loop!")
    finally:
        #subprocess.run(['clear'])
        #fig, fig1 = to_draw(filter_bank, tck_bank, hazards, delay=tx_delay, tsave=tsave)
        #fig.savefig(fpath+'edge_6goasis_tracking_'+tsave+'.png',bbox_inches='tight')
        #print(f"Saved figure: {fpath+'edge_6goasis_tracking_'+tsave+'.png'}")
        #fig1.savefig(fpath+'edge_6goasis_tracking_'+tsave+'_boxplot.png',bbox_inches='tight')
        #print(f"Saved figure: {fpath+'edge_6goasis_tracking_'+tsave+'_boxplot.png'}")
        #print(f" Detected stations: {st_list}")
        #print(f"Hazards: {hazards}")
        lfname_no_ext = os.path.splitext(lfname)[0]
        print(lfname_no_ext)
        fname = None
        if is_master == False:            
            fname =fpath+'edgepf'+tsave+'_'+lfname_no_ext+'_n_parts_'+str(nn)+'_lkl_edge_'+str(lkl_edge)+'.csv'
        else:
            fname =fpath+'edgepf'+tsave+'_'+lfname_no_ext+'_n_parts_'+str(nn)+'_lkl_edge_'+str(lkl_edge)+'_master.csv'
        the_df.to_csv(fname)
        print(f"Data saved as {fname}")
        if is_kafka:
            consumer.close() #here
        print(f"dicarded. {discont}")
        print(f"filter for tracking connected vehicle: {n_filts}")
        match = re.search(r'delay_(\d+)_ms', lfname_no_ext)
        if match:
            delay = str(match.group(1))
        mini_dict = dict()
        mini_dict['timestamp'] = [tsave]
        mini_dict['delay'] = [delay]
        mini_dict['particles'] = [str(nn)]
        mini_dict['likelihood'] = [lkl_edge]
        mini_dict['discarded'] = [discont]
        mini_dict['filters used'] = [n_filts]
        mini_dict['master case'] = 'No'
        if is_master == True:
            mini_dict['master case'] = 'Yes'
        mini_df = pd.DataFrame(mini_dict)
        mini_name = fpath + 'filts_disc.csv'
        try:
            old_mini_df = pd.read_csv(mini_name)
            old_mini_df = pd.concat([old_mini_df,mini_df],ignore_index=True)
            old_mini_df.to_csv(mini_name, index=False)
            print(old_mini_df)
        except Exception as e:
            print(e)
            mini_df.to_csv(mini_name, index=False)
            print(mini_df)
        t_final = time.time()
        print(f"Total time: {t_final - t_start} s")
        print("Sorry! No english! Goodbye...")

if __name__ == '__main__':
    main()
