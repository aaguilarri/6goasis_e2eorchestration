import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pymap3d as pm
import scikit_posthocs as sp

classes = ['bus','car','car_hero_camera','car_agent_camera','fire hydrant','parking meter','person','stop sign','traffic light','truck'] 
pathos = '/home/aa/gitlab/6goasis/edge_data/results/'

def compute_risk(rx,ry,dt=0, lat00=41.274927054998706, lon00=1.9865927764756361, typef='particles'): 

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
    if typef == 'particles':
        east, north, up = pm.geodetic2enu(lat0, lon0, 0, lat00, lon00, 0)
        east1, north1, dup = pm.geodetic2enu(lat0 + dlat0, lon0 + dlon0, 0, lat00, lon00, 0)
        lat0 = east
        lon0 =  north
        dlat0 = east1 - east
        dlon0 = north1 - north
        east, north, up = pm.geodetic2enu(lat1, lon1, 0, lat00, lon00, 0)
        east1, north1, dup = pm.geodetic2enu(lat1 + dlat1, lon1 + dlon1, 0, lat00, lon00, 0)
        lat0 = east
        lon0 =  north
        dlat0 = east1 - east
        dlon0 = north1 - north

    #dt is used to make a predcition so dt=0 is current state.
    lat0a = lat0 + dlat0*dt
    lon0a = lon0 + dlon0*dt
    lat1a = lat1 + dlat1*dt
    lon1a = lon1 + dlon1*dt

    
    vv = [dlat0, dlon0] #vehicle speed vector
    vo = [dlat1, dlon1] #object speed vector
    dsep = np.sqrt((lat0 - lat1)**2 + (lon0 - lon1)**2) #separation distance between v and o
    udsep =  [lat1 - lat0, lon1 - lon0] / dsep #uniatry vector of separation
    pvp0 = [lat1 - lat0, lon1 - lon0] # vector of distance difference
    Sv = np.sum(vv * pvp0) #projecton of vehicle sepeed on separaton vector 
    S0 = np.sum(vo * pvp0) #object vector projecton on separation vector
    Srel = Sv - S0 # difference is speed
    sSrel = np.sign(Srel) #sign of speed from vehicles
    mrel = Srel**2 / dsep # risk metric
    print(f"risk is {sSrel * mrel} m/s2")
    return sSrel * mrel
    #Speed of vehicle in m/s


    mvv = np.sqrt((lat0a - lat0)**2 + (lon0a - lon0)**2) / dt
    #print(f" mvv is {mvv}")
    vvv = np.array([lat0a - lat0, lon0a - lon0])
    nmv = np.linalg.norm(vvv)
    #print(f"vvv is {vvv}")
    #print(f"vvv mag is {nmv}")
    vv = vvv / nmv
    vv = mvv * vv
    #print(f"vv is {vv}")
    #The same for the second one
     
    #print(f"(lat1,lon1) = ({lat1},{lon1}) + ({dlat1}, {dlon1})")
    
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

#Instructions:
#Exp 1:
#for experiment 1 (dsistances between GT and filter) set the data path in pathos, and the list of filenames that containe the data, they will be plotted in this order
# filenames must have the format 'output_XXXX_town02_XX_ms.csv'
# set a suitable title for the plot on title_str
#the experiment will output a boxplot comparing the distance differences between the data
#kurskal wallis test is pefromed for the data, set tests with the tuples of indices of the groups to be compared
#current setup assumes data with the same delay are just next each other (0ms with 0ms and so on...)
#the idea is only the pair with the same delay can be compared, because only these are chronollogically aligned
#the toher test is between grups of same algorithm but different delays, this si to investigate the effect of delay.
#the current result is even there is a dfference, statistically is not significa for delays, but it does for algorihtms
#Exp 2
#only data form carmera_car_hero are considered (more control oer it)
#we compute confuison matrix, errors come from the column labeled 'PF type' every non car label is an error

def get_dtw(r0, r1, typef='particles'):
    print(r0.head())
    print(r1.head())
    r0.reset_index(drop=True)
    r1.reset_index(drop=True)
    r0 = r0[(r0['timestamp'] >= r1['timestamp'].min()) & (r0['timestamp'] <= r1['timestamp'].max())]
    
    m = len(r0)
    n = len(r1)
    R = np.zeros([m,n])
    #print(r0.head)
    #print(r1.head)
    #print(f"columns are {r0.columns.values}")
    #print(f"columns are {r1.columns.values}")
    for i in range(m):
        for j in range(n):
            if typef == 'particles':
                la0 = r0.iloc[i]['GT latitude']
                lo0 = r0.iloc[i]['GT longitude']
                la1 = r1.iloc[j]['PF Latitude']
                lo1 = r1.iloc[j]['PF Longitude']
                R[i,j] = haversine(la0,lo0,la1,lo1)
            if typef == 'kalman':
                la0 = r0.iloc[i]['GT x']
                lo0 = r0.iloc[i]['GT y']
                la1 = r1.iloc[j]['PF Latitude']
                lo1 = r1.iloc[j]['PF Longitude']
                R[i,j] = np.sqrt((la0 - la1)**2 + (lo0 - lo1)**2)
            if typef == 'unscented':
                la0 = r0.iloc[i]['GT latitude']
                lo0 = r0.iloc[i]['GT longitude']
                la1 = r1.iloc[j]['PF Latitude']
                lo1 = r1.iloc[j]['PF Longitude']
                R[i,j] = haversine(la0,lo0,la1,lo1)
            #print(f"data is {la0},{lo0},{la1},{lo1}")
            
            
    for i in range(m):
        for j in range(n):
            if i > 0 or j > 0:
                Ra = np.inf
                Rb = np.inf
                Rc = np.inf
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
    lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    radius = 6371000  # approximately 6,371 km
    distance = radius * c
    return distance

def fill_blanks(): #to fix the issue of some blank data in the df
    pathos = './' #'./data/'
    fnames = ['output_kalman_town02_10_ms.csv'] #['output_kalman_town02_0_ms.csv', 'output_kalman_town02_10_ms.csv', 'output_kalman_town02_20_ms.csv', 'output_kalman_town02_50_ms.csv','output_particles_town02_0_ms.csv', 'output_particles_town02_10_ms.csv', 'output_particles_town02_20_ms.csv', 'output_particles_town02_50_ms.csv']
    for fname in fnames:
        df = pd.read_csv(pathos+fname) #read the file to dataframe
        for ix, row in df.iterrows():
            print(f"{row['PF ID']}: {row['PF type']} {type(row['PF type'])} ")
            if row['PF type'] == '':
                row['PF type'] = row['class_str']
        df.to_csv(pathos+fname)

def confusion_to_matrix(confdict):
    global classes
    rows = []
    for cli in classes:
        row = []
        for clj in classes:
            #print(f"{cli}, {clj}, {confdict[cli+'__'+clj]}")
            row.append(confdict[cli+'__'+clj])
        rows.append(row)
    confmat = np.vstack(rows)
    return confmat

def conf_exp(the_class='car_hero_camera'):
    global pathos
    global classes
    max_maha = stats.chi2.ppf((1-0.1), df=2)
    pathos = './data/'
    ###NOTA: place file names KFs first and PFs second, in order of delay
    fnames = ['output_kalman_town02_0_ms.csv', 'output_kalman_town02_10_ms.csv', 'output_kalman_town02_20_ms.csv','output_particles_town02_0_ms.csv', 'output_particles_town02_10_ms.csv', 'output_particles_town02_20_ms.csv', 'output_unscented_town02_0_ms.csv', 'output_unscented_town02_10_ms.csv', 'output_unscented_town02_20_ms.csv']
    lbls = []
    #make labels
    for fname in fnames:
        match = re.search(r'^output_(.*?)_.*?_(\d+)_ms\.csv$', fname)
        if match:
            my_str = None
            algo = match.group(1)      # e.g., 'kalman'
            if algo == 'kalman':
                my_str = 'KF'
            if algo == 'unscented':
                my_str = 'UF'
            if algo == 'particles':
                my_str = 'PF'
            number = int(match.group(2))  # e.g., 0
            lbls.append(my_str+'_'+str(number)+'ms')
    #Process data
    kf_trues = []
    kf_preds = []
    uf_trues = []
    uf_preds = []
    pf_trues = []
    pf_preds = []
    kf_dicts = []
    pf_dicts = []
    uf_dicts = []
    all_kf_trues = []
    all_kf_preds = []
    all_pf_trues = []
    all_pf_preds = []
    all_uf_trues = []
    all_uf_preds = []
    #for each file, we get the metrics generate by sklearn classification_report()
    #we split them in KFs and PFs to allow comparison
    print("Printing report per file")
    for fname, lbl in zip(fnames, lbls):
        print(f"lbl is {lbl}")
        df = pd.read_csv(pathos+fname) 
        trues = df['class_str'].tolist()
        preds = df['PF type'].tolist()
        dicto = classification_report(trues, preds, output_dict=True)
        dicto_str = classification_report(trues, preds)
        if 'KF' in lbl:
            kf_trues.append(trues)
            kf_preds.append(preds)
            kf_dicts.append(dicto)
            all_kf_trues = all_kf_trues + trues
            all_kf_preds = all_kf_preds + preds
        if 'UF' in lbl:
            uf_trues.append(trues)
            uf_preds.append(preds)
            uf_dicts.append(dicto)
            all_uf_trues = all_uf_trues + trues
            all_uf_preds = all_uf_preds + preds
        if 'PF' in lbl:
            pf_trues.append(trues)
            pf_preds.append(preds)
            pf_dicts.append(dicto)
            all_pf_trues = all_pf_trues + trues
            all_pf_preds = all_pf_preds + preds
        print(dicto_str)
    #we add classification_report() for aggreagted KFs, aggregated PFs and total aggregation
    print("Printing report for KFs")
    kfs_dicto = classification_report(all_kf_trues, all_kf_preds, output_dict=True)
    kfs_dicto_str = classification_report(all_kf_trues, all_kf_preds)
    print(kfs_dicto_str)
    print("Printing report for UFs")
    ufs_dicto = classification_report(all_uf_trues, all_uf_preds, output_dict=True)
    ufs_dicto_str = classification_report(all_uf_trues, all_uf_preds)
    print(kfs_dicto_str)
    print("Printing report for PFs")
    pfs_dicto = classification_report(all_pf_trues, all_pf_preds, output_dict=True)
    pfs_dicto_str = classification_report(all_pf_trues, all_pf_preds)
    print(pfs_dicto_str)
    print("Printing total report")
    all_dicto = classification_report(all_kf_trues + all_pf_trues + all_uf_trues , all_kf_preds + all_pf_preds + all_uf_preds, output_dict=True)
    all_dicto_str = classification_report(all_kf_trues + all_pf_trues + all_uf_trues , all_kf_preds + all_pf_preds + all_uf_preds)
    print(all_dicto_str)
    print(all_dicto.keys())
    #Plot generation, we append all dictionaries to have data ready, and add more lables for the aggregated cases
    accuracies = []
    macro_averages = []
    w_averages =[]
    kf_pf_dicts = kf_dicts + pf_dicts + [kfs_dicto] + [pfs_dicto] + [all_dicto]
    strs = lbls + ['KFs', 'PFs', 'UFs' 'All']
    for dicto in kf_pf_dicts:
        accuracies.append(dicto['accuracy'])
        macro_averages.append(dicto['macro avg'])
        w_averages.append(dicto['weighted avg'])
    #Start plotting
    print("Accuracy plot")
    fig, ax = plt.subplots(figsize=(10,8))
    ax.bar(strs, accuracies)
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Values")
    plt.title("Classificaton Accuracy per Experiment")
    plt.savefig(pathos+'accuracy.jpeg')
    plt.show()
    print("Macro average plot")
    x = np.arange(len(strs))  # positions for classes
    width = 0.25  # width of each bar
    precision = []
    recall = []
    f1 = []
    #In the case of m average and w average, we have a further dicitonary that contains precision, recall, f1, so we get the value and plot in grouped bar plot
    for dc in macro_averages:
        precision.append(dc['precision'])
        recall.append(dc['recall'])
        f1.append(dc['f1-score'])
    fig, ax = plt.subplots(figsize=(10,10))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x,         recall,    width, label="Recall")
    ax.bar(x + width, f1,        width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(strs, rotation=45)
    ax.set_ylabel("Score")
    ax.set_title("Macro Averaged Classification Metrics per Experiment")
    ax.legend()
    plt.tight_layout()
    plt.savefig(pathos+'m_avgs.jpeg')
    plt.show()
    #Same code as above, not the ideal, but it works...
    print("Weighted average plot")
    x = np.arange(len(strs))  # positions for classes
    width = 0.25  # width of each bar
    precision = []
    recall = []
    f1 = []
    for dc in w_averages:
        precision.append(dc['precision'])
        recall.append(dc['recall'])
        f1.append(dc['f1-score'])
    fig, ax = plt.subplots(figsize=(10,10))
    ax.bar(x - width, precision, width, label="Precision")
    ax.bar(x,         recall,    width, label="Recall")
    ax.bar(x + width, f1,        width, label="F1-score")
    ax.set_xticks(x)
    ax.set_xticklabels(strs, rotation=45)
    ax.set_ylabel("Score")
    ax.set_title("Weighted Averaged Classification Metrics per Experiment")
    ax.legend()
    plt.tight_layout()
    plt.savefig(pathos+'w_avgs.jpeg')
    plt.show()

    
    
def risk_exp():
    global pathos
    top_maha = scipy.stats.chi2.ppf((1-0.5), df=2)
    print(f"top maha is {top_maha}")
    fnames = ['output_kalman_town02_0_ms.csv', 'output_kalman_town02_10_ms.csv', 'output_kalman_town02_20_ms.csv',
            'output_particles_town02_0_ms.csv', 'output_particles_town02_10_ms.csv', 'output_particles_town02_20_ms.csv',
            'output_unscented_town02_0_ms.csv', 'output_unscented_town02_10_ms.csv', 'output_unscented_town02_20_ms.csv']
    lbls = []
    for fname in fnames:
        match = re.search(r'^output_(.*?)_.*?_(\d+)_ms\.csv$', fname)
        if match:
            my_str = None
            algo = match.group(1)      # e.g., 'kalman'
            if algo == 'kalman':
                my_str = 'KF'
            if algo == 'particles':
                my_str = 'PF'
            if algo == 'unscented':
                my_str = 'UF'
            number = int(match.group(2))  # e.g., 0
            lbls.append(my_str+'_'+str(number)+'ms')
    title_str = 'DTW Between Estimations and GT (All): Town02 Experiment'
    fig, ax = plt.subplots(2,1,figsize=(10,8))
    marks = ['o', '+', '','o', '+', '']
    for fname, lbl, mark in zip(fnames, lbls, marks):
        df = pd.read_csv(pathos+fname)
        df_ch = df[df['class_str'] == 'car_hero_camera']
        df_ch = df[df['Mahalonabis'] < top_maha]
        ixo = 0
        if 'PF' in lbl: ixo = 1
        #if 'PF' in lbl: # in ['KF_0ms', 'PF_0ms']:
        #ax.stem(df["timestamp"], df["Risk"], basefmt=" ", label=lbl)
        ax[ixo].plot(df_ch['timestamp'], df_ch['Risk'], label=lbl)
    ax[0].set_xlabel('timestamp')
    ax[0].set_ylabel('m_rel / g')
    ax[1].set_xlabel('timestamp')
    ax[1].set_ylabel('m_rel / g')
    ax[0].legend()
    ax[1].legend()
    plt.suptitle("Risk metric results (Town02)")
    plt.savefig(pathos+'risk_results.jpeg')
    plt.show()
            

def track_exp():    
    global pathos
    #fnames = ['output_kalman_town02delay_0_ms.csv', 'output_particles_town02delay_0_ms.csv', 'output_kalman_town02delay_10_ms.csv', 'output_particles_town02delay_10_ms.csv', 'output_kalman_town02delay_50_ms.csv', 'output_particles_town02delay_50_ms.csv']
    #tests = [(0,1), (2,3), (4,5), (0,2,4), (1,3,5)]
    #fnames = ['output_kalman_town02_0_ms.csv','output_kalman_town02_10_ms.csv','output_kalman_town02_50_ms.csv']
    pgtfname = 'output_particles_town02_0_ms.csv'
    kgtfname = 'output_kalman_town02_0_ms.csv'
    ugtfname = 'output_unscented_town02_0_ms.csv'
    #fnames = ['output_kalman_town02_0_ms.csv','output_kalman_town02_10_ms.csv','output_kalman_town02_20_ms.csv']
    #fnames = ['output_particles_town02_0_ms.csv','output_particles_town02_10_ms.csv','output_particles_town02_20_ms.csv']
    #fnames = ['output_kalman_town02_0_ms.csv','output_kalman_town02_10_ms.csv','output_kalman_town02_15_ms.csv','output_kalman_town02_20_ms.csv','output_kalman_town02_50_ms.csv',
    #            'output_particles_town02_0_ms.csv','output_particles_town02_10_ms.csv','output_particles_town02_15_ms.csv','output_particles_town02_20_ms.csv','output_particles_town02_50_ms.csv']
    fnames = ['output_kalman_town02_0_ms.csv', 'output_kalman_town02_10_ms.csv', 'output_kalman_town02_20_ms.csv',
                'output_particles_town02_0_ms.csv', 'output_particles_town02_10_ms.csv', 'output_particles_town02_20_ms.csv',
                'output_unscented_town02_0_ms.csv', 'output_unscented_town02_10_ms.csv', 'output_unscented_town02_20_ms.csv']
    #fnames = ['output_unscented_town02_0_ms.csv', 'output_unscented_town02_10_ms.csv', 'output_unscented_town02_20_ms.csv']
    tests = [(0,1,2,3)]
    lbls = []
    for fname in fnames:
        match = re.search(r'^output_(.*?)_.*?_(\d+)_ms\.csv$', fname)
        if match:
            my_str = None
            algo = match.group(1)      # e.g., 'kalman'
            if algo == 'kalman':
                my_str = 'KF'
            if algo == 'particles':
                my_str = 'PF'
            if algo == 'unscented':
                my_str = 'UF'
            number = int(match.group(2))  # e.g., 0
            lbls.append(my_str+'_'+str(number)+'ms')
    dists =[]
    title_str = 'DTW Between Estimations and GT (All): Town02 Experiment'
    print("Experiment 1, Quality of trajectory")
    the_dtws = []
    for fname, lbl in zip(fnames, lbls):
        typef = None
        dfgt = None
        if 'kalman' in fname:
            typef = 'kalman'
            dfgt = pd.read_csv(pathos+kgtfname)
        if 'unscented' in fname:
            typef = 'unscented'
            dfgt = pd.read_csv(pathos+ugtfname)
        if 'particles' in fname:
            typef = 'particles'
            dfgt = pd.read_csv(pathos+pgtfname)
        df = pd.read_csv(pathos+fname)
        #df = df[df['class_str']=='car_hero_camera']
        disto = None
        chunk_sz = 30
        dtws = []
        #dists = []
        #if typef == 'kalman':
        #    #disto = np.sqrt((dfgt['GT x'] - df['PF Latitude'])**2 + (dfgt['GT y'] - df['PF Longitude'])**2)
        #    disto = haversine(dfgt['GT x'], dfgt['GT y'], df['PF Latitude'], df['PF Longitude'], typef=typef)
        #if typef == 'particles':
        #    disto = haversine(dfgt['GT latitude'], dfgt['GT longitude'], df['PF Latitude'], df['PF Longitude'], typef=typef)
        #dist = pd.DataFrame(disto, columns=[lbl])
        #dists.append(disto)
        #mn = np.mean(dist)
        #sd = np.std(disto)
        #print(f"{lbl}-> mean: {mn}, std {sd}")
        for i in range(0, len(df), chunk_sz):
            df_chunk = df.iloc[i:i + chunk_sz].reset_index(drop=True)
            #dfgt_chunk = dfgt.iloc[i:i + chunk_sz].reset_index(drop=True)
            a_dtw = get_dtw(dfgt, df_chunk, typef=typef)
            dtws.append(a_dtw)
        df_dtws = pd.DataFrame(dtws, columns=[lbl])
        dtws = np.vstack(dtws)
        print(f"dtww has length {dtws.shape}")
        the_dtws.append(df_dtws)
        #the_dtws.append(dists)
    df_dtw = pd.concat(the_dtws, axis=1)
    df_dtw = df_dtw.dropna()
    df_dtw.to_csv(pathos+'dtws.csv')
    print(df_dtw.head)
    print(df_dtw.median(axis=0))
    print(df_dtw.std(axis=0))
    df_dtw.boxplot()
    plt.xticks(ticks=range(1, len(lbls)+1), labels=lbls)
    plt.title(title_str)
    plt.savefig(pathos+'Fig '+title_str+'.jpeg')          
    result_0ms = stats.kruskal(df_dtw['KF_0ms'], df_dtw['PF_0ms'], df_dtw['UF_0ms'])
    result_10ms = stats.kruskal(df_dtw['KF_10ms'], df_dtw['PF_10ms'], df_dtw['UF_10ms'])
    result_20ms = stats.kruskal(df_dtw['KF_20ms'], df_dtw['PF_20ms'], df_dtw['UF_20ms'])
    result_kf = stats.kruskal(df_dtw['KF_0ms'], df_dtw['KF_10ms'], df_dtw['KF_20ms'])
    result_pf = stats.kruskal(df_dtw['PF_0ms'], df_dtw['PF_10ms'], df_dtw['PF_20ms'])
    result_uf = stats.kruskal(df_dtw['UF_0ms'], df_dtw['UF_10ms'], df_dtw['UF_20ms'])
    result = stats.kruskal(*df_dtw)
    print(df_dtw['KF_0ms'])
    stat_0ms, p_val_0ms, med_0ms, tbl_0ms = scipy.stats.median_test(df_dtw['KF_0ms'], df_dtw['PF_0ms'], df_dtw['UF_0ms'])
    stat_10ms, p_val_10ms, med_10ms, tbl_10ms = scipy.stats.median_test(df_dtw['KF_10ms'], df_dtw['PF_10ms'], df_dtw['UF_10ms'])
    stat_20ms, p_val_20ms, med_20ms, tbl_20ms = scipy.stats.median_test(df_dtw['KF_20ms'], df_dtw['PF_20ms'], df_dtw['UF_20ms'])
    print("Kruskal 0ms")
    print(result_0ms)
    print("Mood’s median test statistic:", stat_0ms)
    print("p-value:", p_val_0ms)
    print("Common median estimated:", med_0ms)
    print(tbl_0ms)
    print("Kruskal 10ms")
    print(result_10ms)
    print("Mood’s median test statistic:", stat_10ms)
    print("p-value:", p_val_10ms)
    print("Common median estimated:", med_10ms)
    print("Kruskal 20ms")
    print(result_20ms)
    print("Mood’s median test statistic:", stat_20ms)
    print("p-value:", p_val_20ms)
    print("Common median estimated:", med_20ms)
    # Convert wide -> long
    df_long = df_dtw.melt(var_name='condition', value_name='value')
    # Extract filter name and delay (optional)
    df_long['filter'] = df_long['condition'].str.extract(r'(^[A-Z]+)')   # KF, PF, UF
    df_long['delay'] = df_long['condition'].str.extract(r'(\d+)').astype(int)
    for d in [0, 10, 20]:
        df_delay = df_long[df_long['delay'] == d]
        print(f"\nDelay {d} ms")
        print(sp.posthoc_dunn(df_delay, val_col='value', group_col='filter', p_adjust='bonferroni'))
    print("total")
    print(result)
    print("result kf")
    print(result_kf)
    print("result pf")
    print(result_pf)
    print("result uf")
    print(result_uf)
    print("...")
    k = df_dtw.shape[1]
    n = k * df_dtw.shape[0]
    print(f"H {result[0]}, n {n}, k {k}")
    epsilon_sq = (result[0] - k + 1) / (n - k)
    print("Epsilon-squared:", epsilon_sq)
    f_stat, p_val = stats.f_oneway(*the_dtws)
    print("F-statistic:", f_stat)
    print("p-value:", p_val)
    #cols = the_dtws.columns
    stat, p_val, med, tbl = scipy.stats.median_test(*(col[lbl] for col, lbl in zip(the_dtws, lbls)))
    print("Mood’s median test statistic:", stat_0ms)
    print("p-value:", p_val_0ms)
    print("Common median estimated:", med_0ms)
    plt.show()
    exit()
    #    #dfh = df[df['class_str']=='car_hero_camera']
    #    #print(df.head)
    #    #print(f"columns are {df.columns.values}")
    #    dist = None
    #    if typef == 'kalman':
    #        disto = np.sqrt((dfgt['GT x'] - df['PF Latitude'])**2 + (dfgt['GT y'] - df['PF Longitude'])**2)
    #    if typef == 'particles':
    #        disto = haversine(dfgt['GT latitude'], dfgt['GT longitude'], df['PF Latitude'], df['PF Longitude'])
    #    #dist = pd.DataFrame(disto, columns=[fname])
    #    dists.append(disto)
    #    mn = np.mean(disto)
    #    sd = np.std(disto)
    #    print(f"{lbl}-> mean: {mn}, std {sd}")
    df_dist =  pd.concat(dists, axis=1)
    df_dist = df_dist.dropna()
    print(df_dist.head)
    df_dist.to_csv(pathos+'_dists.csv')
    #print(df_dist.head)
    #print(f"columns are {df_dist.columns.values}")
    #print(df_dist)
    #for dist in dists:
    #    print(dist.shape)
    df_dist.boxplot()
    plt.xticks(ticks=range(1, len(lbls)+1), labels=lbls)
    plt.title(title_str)
    plt.savefig(pathos+'Fig '+title_str+'.jpeg')          
    result = stats.kruskal(*df_dist)
    print(result)
    plt.show()
    exit()
    print("Experiment 2. Effect on trajectory tracking")
    title_str2 = 'Mahalonabis Distance During Tracking: Town02 Experiment'
    mahas = []
    for fname, lbl in zip(fnames, lbls):
        df = pd.read_csv(pathos+fname)
        dfh = df[df['class_str']=='car_hero_camera']
        maha = dfh['Mahalonabis']
        mahas.append(maha)
        mn = np.mean(maha)
        sd = np.std(maha)
        print(f"{lbl}-> mean: {mn}, std {sd}")
    plt.boxplot(mahas, tick_labels=lbls)
    plt.title(title_str2)
    plt.savefig(pathos+'Fig '+title_str2+'.jpeg')
    
    for test in tests:
        print("Files:")
        tst_dat = []
        for ix in test:
            print(lbls[ix])
            tst_dat.append(mahas[ix])                
        result = stats.kruskal(*tst_dat)
        print(result)
    plt.show()
    print("Experiment 3. Number of Filters Evolution")
    nps = []
    for fname, lbl in zip(fnames, lbls):
        df = pd.read_csv(pathos+fname)
        #dfh = df[df['class_str']=='car_hero_camera']
        npi = df['NFs']
        nps.append(npi)
        mn = np.mean(npi)
        sd = np.std(npi)
        print(f"{lbl}-> mean: {mn}, std {sd}")
        plt.plot(df['timestamp'], npi, label=lbl)
    #plt.(mahas, tick_labels=lbls)
    plt.title(title_str2)
    plt.legend()
    plt.savefig(pathos+'Fig '+'nfilts'+'.jpeg')
    plt.show()

def main():
    pathos=''
    track_exp()
    #conf_exp()
    #fill_blanks()
    risk_exp()

if __name__ == '__main__':
    main()
