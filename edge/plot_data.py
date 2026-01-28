import pickle
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
from pfilt import Pfilter
from trajectory import Trajectory
import time
import numpy as np
import random

def plot_trajectories(trajectories_dict):
    fig, ax = plt.subplots()
    n = 1000
    lat = 41.274927054998706
    lon = 1.9865927764756361
    filt = Pfilter(n,lat,lon)
    ddg = filt.mdeg
    delta = 1000
    latN = lat + delta * ddg
    latS = lat - delta * ddg
    lonE = lon + delta * ddg
    lonW = lon - delta * ddg
    plt.xlim(latS, latN)
    plt.ylim(lonW, lonE)
    to_plot = []
    #ax.set_facecolor('black')
    trajo =[]
    for traj in trajectories_dict.values():
        trajito = []
        for tj in traj.trajectory:
            trajito.append(tj[1])
        trajo.append(trajito)
        trajo = np.array(trajo)
        traza = np.vstack(trajo)
        colors = ['red', 'green', 'blue']
        cont = 0
        ax.scatter(traza[:,0], traza[:,1], marker='*', color=colors[cont % 3], linestyle='--')
        cont += 1
    plt.pause(0.2)
    plt.close()
    #plt.show(block=False)

def manage_trajectories(trajectories_dict, tracklet, n=1000):
    #Extract tracklet info
    tsp = tracklet['timestamp']
    st_id = tracklet['station_id']
    st_lat = tracklet['station_lat']
    st_lon = tracklet['station_lon']
    id = tracklet['ID']
    clss = tracklet['class']
    conf = tracklet['conf']
    lat = tracklet['latitude']
    lon = tracklet['longitude']
    dst = tracklet['distance']
    if len(trajectories_dict) <= 0: #No trajectories
        print("Creating a new trajectory")
        traj = Trajectory(n, lat, lon, clss, conf, tsp)
        id = str(time.time())
        trajectories_dict[id] = traj
        print(f"Dict updated {trajectories_dict}")
        return trajectories_dict
    the_id = 0
    lkl0 = np.Inf
    crit0 = False
    for id, traj in trajectories_dict.items():
        print("Trajectory evaluation")
        crit,lkl = traj.eval_tracklet(lat,lon,clss,conf,tsp)
        if lkl < lkl0: #other critera possible as well
            lkl0 = lkl
            crit0 = crit
            the_id = id
    if the_id == 0: #data did not fit any current trajectory
        print("A new trajectory will be created")
        traj = Trajectory(n, lat, lon, clss, conf, tsp)
        id = str(time.time())
        trajectories_dict[id] = traj
        return trajectories_dict
    else: #tracklet fits any trajectory
        print("A trajectory will be updated")
        traj = trajectories_dict[the_id]
        traj.add_tracklet(lat, lon, tsp, clss, conf)
        trajectories_dict[the_id] = traj
    return trajectories_dict

def plot_data(df):
    
    # Plot the GeoDataFrame
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    ax = world.plot(figsize=(10, 6))
    gdf.plot(ax=ax, color='red', marker='o', label='Locations')
    gdf2.plot(ax=ax, color='blue', marker='*', label='Locations')

    # Set labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Scatter Plot of Latitude/Longitude Coordinates')

    # Show the plot
    plt.show()
    return

# Load the data back
#with open('oasis.pkl', 'rb') as file:
#    loaded_data = pickle.load(file)
#print(loaded_data)
# Load all data back
loaded_data = []
with open('edgepf.pkl', 'rb') as file:
    try:
        while True:
            loaded_data.append(pickle.load(file))
    except EOFError:
        pass
df =pd.DataFrame(loaded_data)
#print(df.to_string())
#df = dfx #dfx[dfx['class_str']=='traffic light']
print(df.to_string())
#df = pd.DataFrame(loaded_data)
geometry = [Point(xy) for xy in zip(df['station_lon'], df['station_lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
geometry2 = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf2 = gpd.GeoDataFrame(df, geometry=geometry2, crs="EPSG:4326")
fig, ax = plt.subplots()
ax.scatter(df['station_lon'], df['station_lat'], marker='+', color='blue', label='Camera')
ax.scatter(df['longitude'], df['latitude'], marker='o', facecolors='none', edgecolors='red', label='Detection')
#ax.scatter(gdf['station_lat'], gdf['station_lon'], marker='+', color='blue', label='Camera')
#ax.scatter(gdf2['latitude'], gdf2['longitude'], marker='o', facecolors='none', edgecolors='red', label='Detection')
plt.legend()
plt.title('Edge Detections')
plt.show()
#print(gdf[['station_lon', 'station_lat']][0:10])
df['station_lon'] = df['station_lon'].astype(float)
df['station_lat'] = df['station_lat'].astype(float)
df['timestamp'] = df['timestamp'].astype(float)
print('*')
#print(gdf[300:400])
lls = df[['timestamp', 'station_lat', 'station_lon']]
lls.loc[1:, 'timestamp'] = lls['timestamp'][1:].values - lls['timestamp'][:-1].values
lls['timestamp'][0]=1
lls = lls[lls['timestamp']>0]
lls['timestamp'][0]=0
filt = Pfilter(1000, lls['station_lat'][0], lls['station_lon'][0])
past = []
#print(lls.to_string())
plt.scatter(lls['station_lon'],lls['station_lat'])
plt.show()
lls.reset_index(inplace=True)
for ix, row in lls.iterrows():
    #print(f"row is {row}")
    sample = [row['station_lat'], row['station_lon']]
    dt = 10 * random.random()
    df = pd.DataFrame(filt.denshi)
    print(df.to_string())
    denshi = filt.predict(filt.denshi, dt)
    denshi = filt.update(denshi, sample)
    [lat, lon] = filt.estimate_pos(denshi)
    denshi = filt.resample(denshi, [lat, lon])
    filt.estimate = [lat, lon]
    filt.denshi = denshi
    past.append([sample])
    filt.plot_state(denshi, sample, filt.estimate, dt, past)
plt.show()
   


