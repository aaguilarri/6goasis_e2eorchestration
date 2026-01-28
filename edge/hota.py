import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

path = '/home/sakin/Code/gitlab/6goasis/edge_data/results/' # '/home/aa/gitlab/6goasis/edge_data/results/'
fname = 'hota_report.csv'
df = pd.read_csv(path+fname)
print(df)
labels = df['label'].unique().tolist()
hotas = []
dh = dict()
for lbl in labels:
    df_lbl = df[df['label']==lbl]
    hota = df_lbl['HOTA'].to_list()
    hotas.append(hota)
    dh[lbl] = hota
print(f"{len(hotas)} {labels}")
fig, ax = plt.subplots(1,3)
kf_lbl = ['KF_0ms', 'KF_10ms', 'KF_20ms','KF_50ms', 'KF_100ms']
pf_lbl = ['PF_0ms', 'PF_10ms', 'PF_20ms','PF_50ms', 'PF_100ms']
uf_lbl = ['UF_0ms', 'UF_10ms', 'UF_20ms','UF_50ms', 'UF_100ms']
ax[0].boxplot([dh[i] for i in kf_lbl], labels=kf_lbl)
ax[1].boxplot([dh[i] for i in pf_lbl], labels=pf_lbl)
ax[2].boxplot([dh[i] for i in uf_lbl], labels=uf_lbl)
ax[0].set_ylabel('HOTA')
ax[1].set_ylabel('HOTA')
ax[2].set_ylabel('HOTA')
ax[0].set_title('a) HOTA for KFs')
ax[1].set_title('b) HOTA for PFs')
ax[2].set_title('c) HOTA for UFs')
result = stats.kruskal(*hotas)
print(result)
plt.show()
fig, ax = plt.subplots(1,1)
all_lbl = ['KF_0ms', 'KF_10ms', 'KF_20ms','KF_50ms', 'KF_100ms', 'PF_0ms', 'PF_10ms', 'PF_20ms','PF_50ms', 'PF_100ms', 'UF_0ms', 'UF_10ms', 'UF_20ms','UF_50ms', 'UF_100ms']
ax.boxplot([dh[i] for i in all_lbl], labels=all_lbl)
ax.set_ylabel('HOTA')
ax.set_title('d) HOTA for All Filters')
plt.show()


#KruskalResult(statistic=np.float64(70.88141491286), pvalue=np.float64(1.3358334881214114e-09))
