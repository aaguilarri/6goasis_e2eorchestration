import numpy as np
import pandas as pd

count_dict = {'clutter':['clutter'], 'traffic.unknown':['car', 'truck'], 'vehicle.lincoln.mkz_2017':['car_hero_camera'],
              'traffic.speed_limit.90':['None', 'car'], 'traffic.traffic_light':['traffic light'], 'vehicle.dodge.charger_police':['car'],
              'vehicle.lincoln.mkz_2017':['car_hero_camera']}

def count_df(df):
    tot_rows = df.shape[0]
    tot_cols = df.shape[1]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for ix, row in df.iterrows():
        row_key = row['carla_class']
        row_val = row['class_str']
        true_vals = count_dict[row_key]
        if row_key == 'clutter':
            FP += 1
            continue
        if row_val in true_vals:
            TP += 1
            continue
        if row_val not in true_vals:
            FN += 1
            continue
    TN = tot_rows - TP - FP - FN
    #print(f"TP {TP}, TN {TN}, FP {FP}, FN {FN} TOT {tot_rows}")
    accuracy = TP / (TP + FN + FP)
    #print(f"accuracy {accuracy}")
    cf_dict = {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
    return accuracy, cf_dict

def assoc_df(df):
    tot_rows = df.shape[0]
    tot_cols = df.shape[1]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    filter_names = df['PF ID'].unique().tolist()
    #print(filter_names)
    for fni in filter_names:
        fni_df = df[df['PF ID'] == fni]
        class_names = fni_df['carla_class'].unique().tolist()
        class_counts = fni_df['carla_class'].value_counts()
        maxo = np.max(class_counts.values)
        otros = np.sum(class_counts.values) - maxo
        #print(class_names)
        #print(class_counts)
        #print(maxo)
        #print(otros)
        TP += maxo
        FN += otros
    #print(f"TP {TP}, FN {FN}, tot {tot_rows}")
    accuracy = TP / (TP  + FP + FN )
    cf_dict = {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}
    return accuracy , cf_dict


