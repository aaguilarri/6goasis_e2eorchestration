import numpy as np
import time
import random
import sys
import re
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

def get_dtw(r0, r1):
    print(r0.head())
    print(r1.head())
    r0 = r0[(r0['timestamp'] >= r1['timestamp'].min()) & (r0['timestamp'] <= r1['timestamp'].max())]
    r0.reset_index(drop=True)
    r1.reset_index(drop=True)
    m = len(r0)
    n = len(r1)
    R = np.zeros([m,n])
    for i in range(m):
        for j in range(n):
            la0 = r0.loc[i,'GT latitude']
            lo0 = r0.loc[i,'GT longitude']
            la1 = r1.loc[j,'PF latitude']
            lo1 = r1.loc[j,'PF longitude']
            R[i,j] = haversine(la0,lo0,la1,lo1)
    for i in range(m):
        for j in range(n):
            if i > 0 or j > 0:
                Ra = np.Inf
                Rb = np.Inf
                Rc = np.Inf
                if i > 0: 
                    Ra = R[i-1,j]
                if j > 0:
                    Rb = R[i,j-1]
                if i>0 and j >0:
                    Rc = R[i-1, j-1]
                R[i,j] += np.min([Ra,Rb,Rc])
    print(f"DTW = {R[-1,-1]}")
    return R[-1,-1]
    
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

def mse(lat1, lon1, lat2, lon2):    
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlats = np.array(lat1 - lat2)
    dlons = np.array(lon1 - lon2)
    difs = np.hstack([dlats[:,np.newaxis], dlons[:,np.newaxis]])
    errs = np.linalg.norm(difs,axis=1)
    return errs

def main():
    args = sys.argv
    folder_path = args[1]
    save_path = args[2]
    legend = args[3]
    dfs = [] #DataFrames
    dls = [] #delays from filename
    nss = [] #N particles from filename
    lkls = [] #Lkl edge from filename
    id = None
    my_dict = {'delays':'ms','particles':'n = ','likelihood':'L = ','risk':' m/s'}
    print("------Summary------")
    print(f"Loading from {folder_path}")
    print(f"Saving to {save_path}")
    print(f"Legend: {legend}")
    print("-------------------")
    # Iterate through files in the folder
    id = str(int(time.time()))
    df_dict = dict()
    the_labels = []
    the_route = None
    for filename in os.listdir(folder_path):
        print(f"Processing file: {filename}")
        print(f"Path is {folder_path}")
        if 'edgepf' not in filename:
            print("Not a data file")
            continue
        if filename.endswith('.csv'):
            # Extract delay value using regex
            match = re.search(r'delay_(\d+)_ms', filename)
            if match:
                delay = str(match.group(1))
                dls.append(delay)
            match = re.search(r'n_parts_(\d+)_', filename)
            if match:
                n = str(match.group(1))
                nss.append(n)
            match = re.search(r'lkl_edge_(\d+).', filename)
            if match:
                lkl = str(match.group(1))
                lkls.append(lkl)
            a_label = delay+'/'+n+'/'+lkl
            if 'master' in filename:
                a_label = a_label + '*'
            # Read the CSV file into a dataframe
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            print(f"Opening file: {filename}")
            print(df[['filter', 'GT latitude', 'GT longitude', 'latitude', 'longitude', 'PF latitude', 'PF longitude']])
            hvs = haversine(df['GT latitude'],df['GT longitude'],df['PF latitude'],df['PF longitude'])
            df['haversine'] = hvs
            errs = mse(df['GT latitude'],df['GT longitude'],df['PF latitude'],df['PF longitude'])
            df['MSE'] = errs
            #df.to_csv(folder_path+'updated_'+filename)
            #df[['PF latitude', 'PF longitude']] = df[['PF latitude', 'PF longitude']].fillna(0)
            df_clean = df.dropna(subset=['PF latitude','PF longitude'])
            df = df_clean.copy()
            df = df.reset_index(drop=True)
            print(f"saving data from file {filename} to label {a_label}")
            df_dict[a_label] = df
            the_labels.append(a_label)
            if a_label == '0/500/300':
                the_route = df[['timestamp','GT latitude', 'GT longitude']]
    #################################### GT routes #################################################
    print("Creating routes plot GTs delay 0 ms vs delay 1000 ms")
    my_labels = ['0/500/300','1000/500/300']
    df1 = df_dict[my_labels[0]]
    df2 = df_dict[my_labels[1]]
    fig, (ax0, ax1) = plt.subplots(1,2,figsize=(15,10))
    for ix, row in df1.iterrows():
        lat = row['GT latitude']
        lon = row['GT longitude']
        colo = row['color']
        ax0.scatter(lon, lat, color=colo, marker='*')
        ax0.set_xlim(1.98650,1.98750)
        ax0.set_ylim(41.272,41.2735)
        ax0.set_xlabel('Longitude')
        ax0.set_ylabel('Latitude')
        ax0.set_title(my_labels[0])
    for ix, row in df2.iterrows():
        if row['filter'] == 0:
            continue
        lat = row['GT latitude']
        lon = row['GT longitude']
        colo = row['color']
        ax1.scatter(lon, lat, color=colo, marker='*')
        ax1.set_xlim(1.98650,1.98750)
        ax1.set_ylim(41.272,41.2735)
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(my_labels[1])
    fig.suptitle(f"Effect of Delays on Edge Service (Ground Truth)")
    sstr = save_path+'edge_gtroute'+'.png'
    fig.savefig(sstr)
    print("Figure saved as ", sstr)
    ################################### Routes ##################################################
    print("Making figures for routes")
    print("Delays routes")
    print(f"labels {the_labels}")
    label_pairs = [['0/500/300','1000/500/300']]
    title_labels = ['Delays', 'Number of Particles', 'Threshold']
    for my_labels, tlab in zip(label_pairs,title_labels):
        df1 = df_dict[my_labels[0]]
        df2 = df_dict[my_labels[1]]
        fig, (ax0, ax1) = plt.subplots(1,2,figsize=(15,10))
        for ix, row in df1.iterrows():
            if row['filter'] == 0:
                continue
            lat = row['PF latitude']
            lon = row['PF longitude']
            colo = row['color']
            ax0.scatter(lon, lat, color=colo, marker='*')
            ax0.set_xlim(1.98650,1.98750)
            ax0.set_ylim(41.272,41.2735)
            ax0.set_xlabel('Longitude')
            ax0.set_ylabel('Latitude')
            ax0.set_title(my_labels[0])
        for ix, row in df2.iterrows():
            if row['filter'] == 0:
                continue
            lat = row['PF latitude']
            lon = row['PF longitude']
            colo = row['color']
            ax1.scatter(lon, lat, color=colo, marker='*')
            ax1.set_xlim(1.98650,1.98750)
            ax1.set_ylim(41.272,41.2735)
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_title(my_labels[1])
        fig.suptitle(f"Effect of {tlab} on Edge Service (Estimation)")
        sstr = save_path+'edge_'+tlab+'_route'+'.png'
        fig.savefig(sstr)
        print("Figure saved as ", sstr)
    ################################## Risk Plots ###################################################
    print("Making risk scatter plots")
    #label_pairs = [['0/500/300','10/500/300','100/500/300','1000/500/300']]
    #all_labels = ['0/500/300','10/500/300','100/500/300','1000/500/300']
    label_pairs = [['0/500/300','10/500/300','100/500/300','1000/500/300']]
    all_labels = ['0/500/300','10/500/300','100/500/300','1000/500/300']    
    title_labels = ['Delays', 'Number of Particles', 'Threshold']
    for my_labels, tlab in zip(label_pairs,title_labels):
        fig, ax = plt.subplots(1,1,figsize=(12,12))
        risks = []
        for label in my_labels:
            a_df = df_dict[label]
            a_df_clean = a_df.dropna(subset=['risk'])
            a_df_clean.reset_index(drop=True)
            risk = a_df_clean['risk']
            risks.append(risk)
        ax.boxplot(risks, labels=my_labels)
        ax.set_title(f"Risk Metric for {tlab} Experiment")
        ax.set_ylabel('m/s')
        sstr = save_path+'edge_'+tlab+'_boxplot'+'.png'
        fig.savefig(sstr)
        print("Figure saved as ", sstr)
    print("No bar plot will be generated. That is all!")
    exit()
    ################################## Bar Plots ###################################################
    print("Making bar plots")
    which_one = 'DataFrame'
    if which_one == 'DataFrame':
        fnames = ['filts_disc.csv'] #,'filts_disc_parts.csv','filts_disc_threshold.csv']
        data_dict = dict()
        #Getting data in a dictionary per label
        for fname in fnames:
            a_df = pd.read_csv(save_path+fname)
            dtws = []
            for ix, row in a_df.iterrows():
                label = str(row['delay'])+'/'+str(row['particles'])+'/'+str(row['likelihood'])
                print(f"Processing experiment {label}")
                b_df = df_dict[label]
                mini_df = b_df[['timestamp', 'PF latitude', 'PF longitude']]
                dtw = get_dtw(the_route, mini_df)
                data_dict[label] = [row['discarded'], row['filters used'], dtw]
        q_df = pd.DataFrame(data_dict)
        q_df.to_csv(save_path+'qdf.csv')
    else:
        q_df = pd.read_csv(save_path+'qdf.csv')
        a_dict = q_df.to_dict('dict')
        data_dict = dict()
        for key in all_labels:
            b_dict = a_dict[key]
            data_dict[key]=[int(b_dict[0]), int(b_dict[1]), float(b_dict[2])]
    #grouping data per experiment
    for my_labels, tlabs in zip(label_pairs, title_labels):
        discarded = []
        used = []
        dtws = []
        for label in my_labels:
            print(f"data dict is {data_dict[label]}")
            rowa = data_dict[label]
            discarded.append(rowa[0])
            used.append(rowa[1])
            dtws.append(rowa[2])
        figa, (axa, axb) = plt.subplots(1,2,figsize=(12, 6))
        width = 0.35
        x = range(len(discarded))
        axa.bar([i - width/2 for i in x], discarded, width, color='blue', label='Discarded')
        axa.bar([i + width/2 for i in x], used, width, color='orange', label='Filters Used')
        axb.bar([i - width/2 for i in x], dtws, width, color='orange', label='DTW(Haversine)')
        labels = []
        axa.legend()
        x = np.arange(len(my_labels))  # Create numeric x-coordinates
        axa.set_xticks(x)  # Set the tick locations
        axa.set_xticklabels(my_labels)
        x_routes = np.arange(len(my_labels))  # Create numeric x-coordinates
        axb.set_xticks(x_routes)  # Set the tick locations
        axb.set_xticklabels(my_labels)
        axb.set_ylabel('Meters')
        axb.legend()
        figa.suptitle(f"Effect of {tlabs} on Edge Service Performance")
        plt.tight_layout()
        strss= save_path+'edge_'+tlabs+'_barplots'+id+'.png'
        figa.savefig(strss)
        print("Figure seved as ", strss)
        print("That is all!")

        
if __name__=='__main__':
    main()