import re
import sys
import os
from os import listdir
from os.path import isfile, join
import time
import copy
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pymap3d as pm
from pfilt import Pfilter
from pfilt import Kfilter
from pfilt import Ufilter
from counter import count_df, assoc_df



def ttc(dx0, dy0, dx1, dy1, vx0, vy0, vx1, vy1, tmax=30, dt=0.001, min_dist=0.1):
    t = np.arange(0, tmax, dt)
    x0 = dx0 + vx0 * t
    y0 = dy0 + vy0 * t
    x1 = dx1 + vx1 * t
    y1 = dy1 + vy1 * t
    norms = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
    ix = np.argmin(norms)
    #print(f"ix is {ix} at time {t[ix]} width distance {norms[ix]}")
    my_ttc = np.inf
    if norms[ix] < min_dist and t[ix] > 0:
        my_ttc = t[ix]
    return my_ttc
    
def mrel(dx0, dy0, dx1, dy1, vx0, vy0, vx1, vy1, eps=0.001):
    vv = np.array([vx0, vy0]) #vehicle speed vector
    vo = np.array([vx1, vy1]) #object speed vector
    #print(f" vv = {vv}, vo = {vo}")
    dsep = np.sqrt((dx0 - dx1)**2 + (dy0 - dy1)**2) #separation distance between v and o
    #print(f"dsep {dsep}")
    udsep = 0
    udsep =  np.array([dx1 - dx0, dy1 - dy0]) / dsep #uniatry vector of separatio

    if dsep < eps:
        dsep = eps
        udesep = np.array([np.cos(np.pi/4), np.sin(np.pi/4)])
    #print(f"udsep {udsep}")
    pvp0 = np.array([dx1 - dx0, dy1 - dy0]) # vector of distance difference
    #print(f"pvp0 {pvp0}")
    Sv = np.sum(vv * udsep) #projecton of vehicle sepeed on separaton vector 
    S0 = np.sum(vo * udsep) #object vector projecton on separation vector
    #print(f"Sv {Sv}, S0 {S0}")
    Srel = Sv - S0 # difference is speed
    sSrel = np.sign(Srel) #sign of speed from vehicles
    mrel = Srel**2 / dsep # risk metric

    #print(f"risk is {sSrel * mrel} m/s2")
    return sSrel * mrel, dsep

def compute_risk(rx,ry,dt=0, lat00=41.274927054998706, lon00=1.9865927764756361, typef='particles'): 
    #print("+++++++++++++++++++++ risk ++++++++++++++++++++++++++++++++++++++++++++")
    #lat NS
    #lon EW
    #get the positon and speed form agent 0
    lat0 = rx[0]
    lon0 = rx[2]
    dlat0 = rx[1]
    dlon0 = rx[3]
    #get the positon and speed form agent 1
    lat1 = ry[0]
    lon1 = ry[2]
    dlat1 = ry[1]
    dlon1 = ry[3]
    dx0 = 0
    dy0 = 0
    vx0 = 0
    vy0 = 0
    dx1 = 0
    dy1 = 0
    vx1 = 0
    vy1 = 0
    R = 6371000  
    if typef == 'particles':
        dy0 = R * (lat0 - lat00) * (np.pi / 180)
        dx0 = R * (lon0 - lon00) * (np.pi / 180) * np.cos((lat0 + lat00) * (np.pi / 360))
        dy1 = R * (lat1 - lat00) * (np.pi / 180)
        dx1 = R * (lon1 - lon00) * (np.pi / 180) * np.cos((lat1 + lat00) * (np.pi / 360))
        vy0 = R * dlat0 * (np.pi / 180)
        vx0 = R * dlon0 * (np.pi / 180) * np.cos(lat0 * (np.pi / 180)) 
        vy1 = R * dlat1 * (np.pi / 180)
        vx1 = R * dlon1 * (np.pi / 180) * np.cos(lat0 * (np.pi / 180))
    if typef == 'kalman':
        dy0 = lat0
        dx0 = lon0
        vy0 = dlat0
        vx0 = dlon0
        dy1 = lat1
        dx1 = lon1
        vy1 = dlat1
        vx1 = dlon1
    if typef == 'unscented':
        dy0 = R * (lat0 - lat00) * (np.pi / 180)
        dx0 = R * (lon0 - lon00) * (np.pi / 180) * np.cos((lat0 + lat00) * (np.pi / 360))
        dy1 = R * (lat1 - lat00) * (np.pi / 180)
        dx1 = R * (lon1 - lon00) * (np.pi / 180) * np.cos((lat1 + lat00) * (np.pi / 360))
        vy0 = R * dlat0 * (np.pi / 180)
        vx0 = R * dlon0 * (np.pi / 180) * np.cos(lat0 * (np.pi / 180)) 
        vy1 = R * dlat1 * (np.pi / 180)
        vx1 = R * dlon1 * (np.pi / 180) * np.cos(lat0 * (np.pi / 180))
    my_mrel, dsep = mrel(dx0, dy0, dx1, dy1, vx0, vy0, vx1, vy1, eps=0.001)
    my_ttc = ttc(dx0, dy0, dx1, dy1, vx0, vy0, vx1, vy1, min_dist=1)
    #print(f"risk is {mrel} m/s2, ttc {my_ttc}")
    if dsep < 0.01:
        exit()
    #print("++++++++++++++++++++++++++++++++++++++++++++")
    return my_mrel, my_ttc, dsep

def transforma(lats, lons, lat0=41.274927054998706, lon0=1.9865927764756361, h0=0):
    # Define transformer from WGS84 to local meters
    hs = None
    if isinstance(lats, float):
        hs = 0
    else:
        hs = np.zeros(lats.shape[0])
    east, north, up = pm.geodetic2enu(lats, lons, hs, lat0, lon0, h0)
    return [east,north]
    
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    radius = 6371000  # approximately 6,371 km
    distance = radius * c
    return distance

def k_mahalonabis(xm, P, H, z):
    #print("xm")
    #print(xm)
    #print(f"P")
    #print(P)
    #print("H")
    #print(H)
    #print("z")
    #print(z)
    tx = np.array(z)
    tx = tx[:,np.newaxis]
    ym = tx - H @ xm
    S = H @ (P @ H.T)
    iS = scipy.linalg.inv(S)
    lt = iS @ ym
    d2 = ym.T @ lt
    mah = d2.diagonal()
    mah0 = mah[0]
    if mah0 < 0:
        mah0 = 0
    mah0 = np.sqrt(mah0)
    return mah0

def p_mahalonabis(denshi, x, vel=None):
    qs = denshi[:,[0,2]]
    if vel is not None:
        x = [x[0],vel[0],x[1],vel[1]]
        qs = denshi[:,:-1]
    ws = denshi[:,-1] 
    lx = len(x)
    xm = qs.T @ ws
    xm = xm.flatten()
    tx = np.array(x)
    tx = tx[:,np.newaxis]
    xm = xm[:, np.newaxis]
    ws = ws[:, np.newaxis]
    pw = qs*ws
    covs = pw.T @ pw
    icov = scipy.linalg.inv(covs)
    lt = icov @ (tx - xm)
    d2 = (tx - xm).T @ lt
    mah = d2.diagonal()
    mah0 = mah[0]
    if mah0 < 0:
        mah0 = 0
    mah0 = np.sqrt(mah0)
    return mah0

#python edge_experiment.py 
#applies the MOCT traclets fusion, select in the scirpt filter type (kalman/particles), the path in paths, file name in fname (.csv, output form delayer.py)
def main():
    args = sys.argv
    t0 = time.time()
    typef = args[2] #'kalman', 'particles'
    n = 10000 #particles
    top_maha = 0
    lbl = None
    #define mahalonabis threshold
    if typef == 'particles':
        lbl ='PF'
        top_maha = scipy.stats.chi2.ppf((1-0.1), df=2)
    if typef == 'kalman':
        lbl = 'KF'
        top_maha = scipy.stats.chi2.ppf((1-0.1), df=2)
    if typef == 'unscented':
        lbl = 'UF'
        top_maha = scipy.stats.chi2.ppf((1-0.1), df=2)
    #defining file saving path
    pathos = args[1]
    my_files = [f for f in listdir(pathos+'csvs/') if isfile(join(pathos+'csvs/', f))]
    #This for can process several files at the same run
    for fname in my_files:
        #load data
        print(fname)
        match = re.search(r'^(.*?)_ts_(\d+)delay_(\d+)_ms\.csv$', fname)
        delay = match.group(3)
        df = pd.read_csv(pathos+'csvs/'+fname)
        print(df.head)
        print(f"columns are {df.columns.values}")
        #transforming coordiantes in the case of using kalman
        [xs,ys] = transforma(df['GT latitude'], df['GT longitude'])
        [zxs,zys] = transforma(df['latitude'], df['longitude'])
        #adding rectangular coordinates data to the df
        xydf = pd.DataFrame(np.column_stack([xs, ys, zxs, zys]),columns=['GT x', 'GT y', 'zx', 'zy'])
        df = pd.concat([df, xydf], axis=1)
        #will have 2 dfs, one for data, and one for results, they will ber processed by rows and concatenated at the end of the loop
        cols=['PF ID', 'PF type','PF color', 'PF Latitude', 'PF Longitude', 'Mahalonabis','Risk', 'ttc', 'dsep', 'NFs', 'difH', 'dt']
        cols0= df.columns.values #['timestamp', 'station_id','station_lat', 'station_lon', 'ID', 'latitude','longitude', 'distance', 'class_str', 'sys_time', '']
        #Initialize data to avoid type errors
        df_pf = pd.DataFrame({cols[0]:pd.Series(dtype="str"),
                            cols[1]:pd.Series(dtype="str"),
                            cols[2]:pd.Series(dtype="object"),
                            cols[3]:pd.Series(dtype="float"),
                            cols[4]:pd.Series(dtype="float"),
                            cols[5]:pd.Series(dtype="float"),
                            cols[6]:pd.Series(dtype="float"),
                            cols[7]:pd.Series(dtype="int"),
                            cols[8]:pd.Series(dtype="int"),
                            cols[9]:pd.Series(dtype="float"),
                            cols[10]:pd.Series(dtype="float"),
                            cols[11]:pd.Series(dtype="float")})
        gdf = pd.DataFrame({cols0[0]:pd.Series(dtype="int"),
                            cols0[1]:pd.Series(dtype="str"),
                            cols0[2]:pd.Series(dtype="float"),
                            cols0[3]:pd.Series(dtype="float"),
                            cols0[4]:pd.Series(dtype="str"),
                            cols0[5]:pd.Series(dtype="int"),
                            cols0[6]:pd.Series(dtype="float"),
                            cols0[7]:pd.Series(dtype="float"),
                            cols0[8]:pd.Series(dtype="float"),
                            cols0[9]:pd.Series(dtype="float"),
                            cols0[10]:pd.Series(dtype="str"),
                            cols0[11]:pd.Series(dtype="float"),
                            cols0[12]:pd.Series(dtype="float"),
                            cols0[13]:pd.Series(dtype="float"),
                            cols0[14]:pd.Series(dtype="float"),
                            cols0[15]:pd.Series(dtype="float")})
        #pfs, saves the filters, lpfs: # of filters, tmax: time to delete idle filters
        pfs = []
        lpfs = 0
        tmax = 10
        timos = []
        #Start the run, iterating each row in data df
        for ix, row in df.iterrows():
            t0 = time.time()
            #print(gdf.head)
            print(f"row GT is {row['GT latitude']}, {row['GT longitude']}")
            lat = row['latitude']
            lon = row['longitude']
            #transform data for kalman
            if typef == 'particles':
                pass
            if typef == 'kalman':
                [lat, lon] = transforma(lat, lon)
            if typef == 'unscented':
                pass

            #this is used to determne dt, normally is from timestamp, but sys_time (time when data is received) could be used as well.
            #collect data form the row to initialize filters
            ts = float(row['sys_time'])
            #print(f"timestamp {int(row['timestamp'])/1**7}, ts {ts}")
            tipos = row['class_str']
            #Case 1. we have no filters, need to make some
            if len(pfs) <= 0:
                color = np.random.rand(3,)
                pf0 = None
                #print("is n?")
                #print(f"n is {n}, lat is {lat}, lon is {lon}, ts {ts}, color {color} tipos {tipos} typef is {typef}" )
                if typef == 'particles':
                    #print("is p!")
                    pf0 = Pfilter(n, lat, lon, ts ,color, typo=tipos)
                if typef == 'kalman':
                    #print("is k!")
                    pf0 = Kfilter([lat, lon], ts, typo=tipos)
                if typef == 'unscented':
                    #print("is u!")
                    pf0 = Ufilter([lat, lon], ts, typo=tipos)
                #print(f"pf0 is {pf0} typef is {typef}")
                pfs.append(copy.deepcopy(pf0))
                lpfs += 1
                #print([lat, lon])
                #print("starting case")
                difH = None
                if typef == 'particles':
                    difH = haversine(lat, lon, row['GT latitude'], row['GT longitude'])
                if typef == 'kalman':
                    difH = np.sqrt((lat-row['GT x'])**2 + (lon-row['GT y'])**2)
                if typef == 'unscented':
                    difH = haversine(lat, lon, row['GT latitude'], row['GT longitude'])
                    #difH = np.sqrt((lat-row['GT x'])**2 + (lon-row['GT y'])**2)
                #making the result df row, and concatenated to the result df
                pf_row = pd.DataFrame([{'PF ID': pf0.id, 'PF type':pf0.type, 'PF color': pf0.color, 'PF Latitude':lat, 'PF Longitude':lon, 'Mahalonabis':0, 'Risk':0, 'ttc':np.inf, 'dsep':np.inf, 'NFs':lpfs, 'difH':difH, 'dt':0}])
                df_pf = pd.concat([df_pf, pf_row], ignore_index=True)
                gdf = pd.concat([gdf, row.to_frame().T], ignore_index=True)
                #print(gdf.head)
                #print("--------------------------------")
                t1 = time.time()
                timos.append(t1 - t0)
                continue        # if there are filters, check the measuemnrer fits
            lkls = []
            mahas = []
            #discarnding late tracklets
            min_last_tck = np.inf
            max_last_tck = -np.inf
            for pf in pfs:
                #print(f"min({min_last_tck},{pf.last_tck} ) = {np.min([min_last_tck, pf.last_tck])}")
                min_last_tck = np.min([min_last_tck, pf.last_tck])
                max_last_tck = np.max([max_last_tck, pf.last_tck])
                #print(f"{ts} < {min_last_tck}?")
            if ts < min_last_tck:
                print("Late tracklet! Discarded...")
                pf_row = pd.DataFrame([{'PF ID': 'None', 'PF type':'None', 'PF color': [0,0,0], 'PF Latitude':lat, 'PF Longitude':lon, 'Mahalonabis':0, 'Risk':0, 'ttc':np.inf, 'NFs':len(pfs), 'difH':0, 'dt':0}])
                print(f"row is {pf_row}")
                #df_pf = pd.concat([df_pf, pf_row], ignore_index=True)
                #gdf = pd.concat([gdf_row], ignore_index=True)
                t1 = time.time()
                timos.append(t1 - t0)
                continue    
            for pf in pfs:
                dpf = copy.deepcopy(pf) 
                dt = (ts*1000 - dpf.last_tck*1000)/1000
                if dt <= 0:
                    print(f"Not valid ts for pf {dt}")
                    continue
                pos0 = dpf.estimate_pos(dpf.denshi)
                pos0 = pos0.flatten()
                lat0 = pos0[0]
                lon0 = pos0[2]
                vlat = (lat - lat0) / dt
                vlon = (lon - lon0) / dt
                lkl = None
                if typef == 'particles': 
                    lkl = dpf.likelihood([lat,lon],None,dt)
                if typef == 'kalman':
                    dpf.predict(dt)
                    lkl = dpf.likelihood([lat,lon],None,dt)
                if typef == 'unscented':
                    lkl = dpf.likelihood([lat,lon],None,dt)
                lkls.append(lkl)
                #print(f"z is {[lat, lon]}")
                pos = dpf.estimate_pos(dpf.denshi).flatten()
                #print(f"estimation is {[float(pos[0]), float(pos[2])]}")
                denshi = copy.copy(dpf.denshi)
                # apply mahalonabis to rule out a new filter must be created
                maha = None
                if typef == 'particles':
                    maha = p_mahalonabis(denshi, [lat, lon], vel=None)
                if typef == 'kalman':
                    #print(f"processing kalman filter {dpf.id}")
                    #H = dpf.getH()
                    maha = k_mahalonabis(dpf.x, dpf.P, dpf.H, [lat, lon])
                    #print(f"k maha is {maha}")
                if typef == 'unscented':
                    H = dpf.getH()
                    maha = k_mahalonabis(dpf.x, dpf.P, H, [lat, lon])
                    #print(f"u maha is {maha}")
                mahas.append(maha)
            if len(lkls) <= 0:
                print("no candidate case")
                df
                difH = None
                if typef == 'particles': #or typef == 'unscented':
                    difH = haversine(lat, lon, row['GT latitude'], row['GT longitude'])
                if typef == 'kalman':
                    difH = np.sqrt((lat-row['GT x'])**2 + (lon-row['GT y'])**2)
                if typef == 'unscented':
                    difH = haversine(lat, lon, row['GT latitude'], row['GT longitude'])
                    #difH = np.sqrt((lat-row['GT x'])**2 + (lon-row['GT y'])**2)
                pf_row = pd.DataFrame([{'PF ID': 'None', 'PF type':'None', 'PF color': [0,0,0], 'PF Latitude':lat, 'PF Longitude':lon, 'Mahalonabis':0, 'Risk':0, 'ttc':np.inf, 'dsep':np.inf, 'NFs':len(pfs), 'difH':difH, 'dt':dt}])
                print(f"row is {pf_row}")
                #df_pf = pd.concat([df_pf, pf_row], ignore_index=True)
                print('+++')
                continue
            ix = np.argmin(mahas)
            dtt = ts - pfs[ix].last_tck
            flag = dtt > 0 and dtt < 3
            if mahas[ix] < top_maha and flag: #lkls[ix] > 300: #mahas[ix] < np.inf:
                my_pf = pfs.pop(ix)
                dt = (ts*1000 - my_pf.last_tck*1000)/1000
                if typef == 'particles':
                    p_den = my_pf.predict(my_pf.denshi, dt)
                    #print(f"after predict den is  {p_den.shape}")
                    u_den, is_ok = my_pf.update(p_den, [lat, lon], 0)
                    #print(f"after update den is  {u_den.shape}")
                    #print(f"we have {p_den.shape[0]} parts / {my_pf.n}")
                    w_eff = 1/np.sum(u_den[:,-1]**2)
                    if w_eff < n/2:
                        #den = copy.copy(my_pf.denshi)
                        u_den = my_pf.resample(u_den, [lat, lon])
                    #print(f"after sample den is  {u_den.shape}")
                    my_pf.denshi = copy.copy(u_den)
                    my_pf.last_tck = ts
                    pos = my_pf.estimate_pos(u_den)
                    pos = pos.flatten()
                    difH = haversine(pos[0], pos[2], row['GT latitude'], row['GT longitude'])
                    pf_row = pd.DataFrame([{'PF ID': my_pf.id, 'PF type':my_pf.type, 'PF color': my_pf.color, 'PF Latitude':pos[0], 'PF Longitude':pos[2], 'Mahalonabis':mahas[ix], 'Risk':0, 'ttc':np.inf, 'dsep':np.inf,'NFs':lpfs, 'difH':difH, 'dt':dt}])
                    print(f"row is {pf_row}")
                    df_pf = pd.concat([df_pf, pf_row], ignore_index=True)
                    gdf = pd.concat([gdf, row.to_frame().T], ignore_index=True)
                    pfs.append(copy.deepcopy(my_pf))
                if typef == 'kalman':
                    my_pf.predict(dt)
                    #print(f"filter {my_pf.id} precitected dt {dt}, x is {my_pf.x}")
                    my_pf.update([lat,lon])
                    #print(f"filter {my_pf.id} updated, x is {my_pf.x}")
                    my_pf.last_tck = ts
                    pos = my_pf.estimate_pos(0)
                    difH = np.sqrt((lat-row['GT x'])**2 + (lon-row['GT y'])**2)
                    pf_row = pd.DataFrame([{'PF ID': my_pf.id, 'PF type':my_pf.type, 'PF color': my_pf.color, 'PF Latitude':pos[0], 'PF Longitude':pos[2], 'Mahalonabis':mahas[ix], 'Risk':0, 'ttc':np.inf, 'dsep':np.inf, 'NFs':lpfs, 'difH':difH, 'dt':dt}])
                    print(f"row is {pf_row}")
                    df_pf = pd.concat([df_pf, pf_row], ignore_index=True)
                    gdf = pd.concat([gdf, row.to_frame().T], ignore_index=True)
                    pfs.append(copy.deepcopy(my_pf))
                if typef == 'unscented':
                    my_pf.predict(dt)
                    #print(f"filter {my_pf.id} precitected dt {dt}, x is {my_pf.x}")
                    my_pf.update([lat,lon])
                    #print(f"filter {my_pf.id} updated, x is {my_pf.x}")
                    my_pf.last_tck = ts
                    pos = my_pf.estimate_pos(0)
                    #difH = np.sqrt((lat-row['GT x'])**2 + (lon-row['GT y'])**2)
                    difH = haversine(pos[0], pos[2], row['GT latitude'], row['GT longitude'])
                    pf_row = pd.DataFrame([{'PF ID': my_pf.id, 'PF type':my_pf.type, 'PF color': my_pf.color, 'PF Latitude':pos[0], 'PF Longitude':pos[2], 'Mahalonabis':mahas[ix], 'Risk':0, 'ttc':np.inf, 'dsep':np.inf, 'NFs':lpfs, 'difH':difH, 'dt':dt}])
                    print(f"row is {pf_row}")
                    df_pf = pd.concat([df_pf, pf_row], ignore_index=True)
                    gdf = pd.concat([gdf, row.to_frame().T], ignore_index=True)
                    pfs.append(copy.deepcopy(my_pf))
                print('+++')
            else:
                color = np.random.rand(3,)
                tipos = row['class_str']
                pf0 = None
                if typef == 'particles':
                    pf0 = Pfilter(n, lat, lon, ts ,color, typo=tipos)
                if typef == 'kalman':
                    pf0 = Kfilter([lat, lon], ts, typo=tipos)
                if typef == 'unscented':
                    pf0 = Ufilter([lat, lon], ts, typo=tipos)
                pfs.append(copy.deepcopy(pf0))
                lpfs += 1
                print([lat, lon])
                difH = None
                if typef == 'particles':# or typef == 'unscented':
                    difH = haversine(lat, lon, row['GT latitude'], row['GT longitude'])
                if typef == 'kalman':
                    difH = np.sqrt((lat-row['GT x'])**2 + (lon-row['GT y'])**2)
                if typef == 'unscented':
                    difH = haversine(lat, lon, row['GT latitude'], row['GT longitude'])
                pf_row = pd.DataFrame([{'PF ID': pf0.id, 'PF type':pf0.type, 'PF color': pf0.color, 'PF Latitude':lat, 'PF Longitude':lon, 'Mahalonabis':mahas[ix], 'Risk':0, 'ttc':np.inf, 'dsep':np.inf, 'NFs':lpfs, 'difH':difH, 'dt':0}])
                df_pf = pd.concat([df_pf, pf_row], ignore_index=True)
                gdf = pd.concat([gdf, row.to_frame().T], ignore_index=True)
                print('+++')
            lpfs = len(pfs)
            #deleting too old filters
            good_pfs = []
            for pf in pfs:
                if np.abs(ts - pf.last_tck) < tmax:
                    good_pfs.append(copy.deepcopy(pf))
            pfs = []
            for pf in good_pfs:
                pfs.append(copy.deepcopy(pf))
            #compute risk:
            if row['class_str'] != 'car_hero_camera': #compute only for car hero
                continue
            my_pf_id = df_pf.iloc[-1]['PF ID'] #get the filter to evaluate
            df_risk = pd.concat([gdf, df_pf], axis=1) # concatante the dfs to have all data
            df_risk = df_risk[df_risk['class_str'] != 'car_hero_camera'] # discar all car hero camera entrys to avoid fals risk
            filts_ids = set(df_risk['PF ID'].tolist()) # get the filters that should be compared
            my_x = None
            not_my_x = []
            # traverse the current filters, find the state of my filter and the states of other fitlers to be evaluated
            for pf in pfs:
                denshi = None
                if typef == 'particles':
                    denshi = np.copy(pf.denshi)
                if pf.id == my_pf_id:
                    my_x = pf.estimate_pos(denshi)
                else:
                    if pf.id in filts_ids:
                        not_my_x.append(pf.estimate_pos(denshi))
            control = 0
            risk0 = np.inf
            ttc0 = np.inf
            dsep0 = np.inf
            for x in not_my_x:
                risko, ttci, dsep = compute_risk(my_x, x, typef=typef)
                if ttci < ttc0:
                    control = np.abs(risko)
                    risk0 = risko
                    ttc0 = ttci
                    dsep0 = dsep
            #if np.isinf(my_risk):
            #    my_risk = 0
            df_pf.at[df_pf.index[-1], 'Risk'] = risk0
            df_pf.at[df_pf.index[-1], 'ttc'] = ttc0
            df_pf.at[df_pf.index[-1], 'dsep'] = dsep0
            t1 = time.time()
            timos.append(t1 - t0)

                
        print(df_pf)
        percs = np.percentile(timos, [0, 25, 50, 75, 90, 99, 100])
        print(f"boxplot data: {percs} std {np.std(percs)}")
        the_df = pd.concat([gdf, df_pf], axis=1)
        #gdf = pd.concat([gdf, row], ignore_index=True)
        the_df = the_df[the_df['PF ID'] != 'None']
        print("------------------------- HOTA REPORT -------------------------------")
        print(f" {the_df.shape[0]} tracklets processeed")
        det_accuracy, det_dicto = count_df(the_df)
        print(f"accuracy {det_accuracy}")
        for key in det_dicto.keys():
            print(f"{key} {det_dicto[key]}")
        assoc_acc, assoc_dicto = assoc_df(the_df)
        
        print(f"accuracy {assoc_acc}")
        for key in assoc_dicto.keys():
            print(f"{key} {assoc_dicto[key]}")
        print(f"HOTA {np.sqrt(det_accuracy * assoc_acc)}")
        hota = np.sqrt(det_accuracy * assoc_acc)
        # Data to add
        new_row = {'label': lbl+'_'+str(delay)+'ms', 'det_acc':det_accuracy, 'assoc_acc':assoc_acc, 'HOTA':hota}
        csv_file = 'hota_report.csv'

        # Check if file exists
        if os.path.exists(pathos+'results/'+csv_file):
            df = pd.read_csv(pathos+'results/'+csv_file)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        # Save back to CSV
        df.to_csv(pathos+'results/'+csv_file, index=False)

        the_df.to_csv(pathos+'results/'+'output_'+typef+'_'+fname)
        
        fig, ax = plt.subplots(1,3, figsize=(12,6))
        flag = True
        for ix, row in the_df.iterrows():
            try:
                if typef == 'particles': 
                    ax[0].scatter(row['PF Longitude'], row['PF Latitude'], color=row['PF color'], marker='o', label='estimations')
                    ax[0].scatter(row['GT longitude'], row['GT latitude'], color=row['PF color'], marker='x', label='ground truth')
                    ax[0].scatter(row['longitude'], row['latitude'], color=row['PF color'], marker='<', label='measurements')
                    #if flag:
                    #    plt.xlabel('Longitude')
                    #    plt.ylabel('Latitude')
                    #    plt.legend()
                    #    flag=False
                if typef == 'kalman':
                    #print(f"colro is {row['PF color']}")
                    ax[0].scatter(row['PF Latitude'], row['PF Longitude'], color=row['PF color'], marker='o', label='estimations')
                    ax[0].scatter(row['GT x'], row['GT y'], color=row['PF color'], marker='x', label='ground truth')
                    ax[0].scatter(row['zx'], row['zy'], color=row['PF color'], marker='<', label='measurements')
                    #if flag:
                    #    ax[0].legend()
                    #    flag=False
                if typef == 'unscented': 
                    ax[0].scatter(row['PF Longitude'], row['PF Latitude'], color=row['PF color'], marker='o', label='estimations')
                    ax[0].scatter(row['GT longitude'], row['GT latitude'], color=row['PF color'], marker='x', label='ground truth')
                    ax[0].scatter(row['longitude'], row['latitude'], color=row['PF color'], marker='<', label='measurements')
                    #if flag:
                    #    plt.xlabel('Longitude')
                    #    plt.ylabel('Latitude')
                    #    plt.legend()
            except Exception as e:
                print(e)
        plot_df = the_df[(the_df['dsep'] != np.inf) & (the_df['class_str']=='car_hero_camera')]         
        ax[1].plot(plot_df['sys_time'], 0.1 * plot_df['Risk'], color='blue', label='mrel')
        ax[2].plot(plot_df['sys_time'], plot_df['ttc'], color='red', label='ttc')
        ax[0].set_title('a) Trajectories')
        ax[0].set_xlabel('Longitude')
        ax[0].set_ylabel('Latitude')
        ax[1].set_title('b) m_rel risk metric')
        ax[1].set_ylabel('m/s²')
        ax[1].set_xlabel('timestamp')
        ax[2].set_title('c) ttc risk metric')
        ax[2].set_ylabel('seconds')
        ax[2].set_xlabel('timestamp')
        print(f"top maha {top_maha}")
        t1 = time.time()
        print(f"Total run time {t1 - t0}")
        #ax[1].legend()
        #plt.title(f"Trajectories on {lbl+'_'+str(delay)+'ms'}")
        plt.savefig(f"{pathos}fig_{typef}_{fname[:-4]}.jpeg")
        print(f"Output done: {pathos+'output_'+typef+'_'+fname}")
        print(f" timos is mean {np.mean(timos)} std {np.std(timos)}")
        print("------------------------- HOTA REPORT -------------------------------")
        plt.show()
    
            
if __name__=='__main__':
    main()