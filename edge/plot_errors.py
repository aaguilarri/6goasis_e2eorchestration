from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

pathos = '/home/sakin/Code/gitlab/6goasis/edge_data/results/'
print(pathos)
my_files = [f for f in listdir(pathos) if isfile(join(pathos, f))]
dict_dat = dict()
for file in my_files:
    print(file)
    match = re.search(r'^output_(.*?)_(.*?)_ts_(\d+)delay_(\d+)_ms\.csv$', file)
    if match is None:
        continue
    tipo = match.group(1)
    delay = match.group(4)
    print(f"{tipo}, {delay}")
    df = pd.read_csv(pathos+file)
    print(df.head)
    #output_kalman_OppositeVehicleRunningRedLight4_ts_17622649735447630delay_10_ms
    dicto = {'kalman':'KF_', 'particles':'PF_', 'unscented':'UF_'}
    lbl = dicto[tipo]+delay+'ms'
    err = df['difH'].to_list()
    dict_dat[lbl] = err

lbls = ['KF_0ms', 'KF_10ms', 'KF_20ms','KF_50ms', 'KF_100ms', 'PF_0ms', 'PF_10ms', 'PF_20ms','PF_50ms', 'PF_100ms', 'UF_0ms', 'UF_10ms', 'UF_20ms','UF_50ms', 'UF_100ms']
fig, ax = plt.subplots(2,1, figsize=(10,5))
ax[0].boxplot([dict_dat[i] for i in lbls], labels=lbls, showfliers=False)
ax[0].set_ylabel('meters')
ax[0].set_title('a) Tracking Error for Filter/Delay')
ax[1].boxplot([dict_dat[i] for i in lbls], labels=lbls, showfliers=True)
ax[1].set_ylabel('meters')
ax[1].set_title('b) Tracking Error for Filter/Delay (with outliers)')
#plt.boxplot(errors)
plt.show()