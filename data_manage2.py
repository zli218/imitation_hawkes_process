import pandas as pd
import numpy as np
import os
import data_manage
import torch

def read_csv(file_path,b_len):
    data = pd.read_csv(file_path)
    data = data.sort_values(by=['time'], ascending=True)

    e = data['typeIdx'].tolist()
    t = data['time'].tolist()

    final_data=[]

    while len(e)>=b_len:
        d={}
        d['nodes']=e[0:b_len]
        e=e[b_len:]
        d['times']=t[0:b_len]
        t=t[b_len:]
        final_data.append(d)
    if len(e)!=0:
        d={}
        d['nodes']=e
        d['times']=t
        final_data.append(d)

    return final_data

def read_csv_2(file_path,b_len):
    #read from orignal data
    data = pd.read_csv(file_path)
    data = data.sample(frac=1)

    e = data[['typeIdx','time']]


    final_data=[]

    while len(e)>=b_len:
        d={}

        batch_data=e[:b_len].sort_values(by=['time'], ascending=True)

        d['nodes']=batch_data['typeIdx'].tolist()
        d['times']=batch_data['time'].tolist()
        e=e[b_len:]
        final_data.append(d)
    if len(e)!=0:
        d={}
        e=e.sort_values(by=['time'], ascending=True)
        d['nodes']=e['typeIdx'].tolist()
        d['times']=e['time'].tolist()
        final_data.append(d)

    return final_data

def read_csv_3(file_path,b_len,mc):
    #imcomplete by mc
    data = pd.read_csv(file_path)
    data = data.sample(frac=1)

    e = data[['typeIdx','time']]

    #e = e.drop(e[e['time']==0].index)

    e= incomp_train_dataset(e,mc)


    final_data=[]

    while len(e)>=b_len:
        d={}

        batch_data=e[:b_len].sort_values(by=['time'], ascending=True)

        d['nodes']=batch_data['typeIdx'].tolist()
        d['times']=batch_data['time'].tolist()
        e=e[b_len:]
        final_data.append(d)
    if len(e)!=0:
        d={}
        e=e.sort_values(by=['time'], ascending=True)
        d['nodes']=e['typeIdx'].tolist()
        d['times']=e['time'].tolist()
        final_data.append(d)

    return final_data

def count_mc(file_path):
    data = pd.read_csv(file_path)
    group1=data.groupby('typeIdx')
    k=group1.count()
    count=np.array(k['id'].tolist())
    c=count/count.max()
    mc = (1/(1+np.exp(-c-0.5)))/2
    mc=mc.tolist()


    return mc

def verify(data,type_number):
    d=[]
    for i in data:
        d=d+i['nodes']

    type=np.unique(d)
    if np.array(type).all()==np.array(list(range(0,type_number))).all():
        print("Right")
        return
    print("Wrong")

    return


def incomp_train_dataset(seqs,mc):


    m_c=[1-x for x in mc]

    for i in range(len(seqs)):
        u = np.random.uniform()
        if u < m_c[seqs['typeIdx'][i]] or seqs['time'][i]==0.0:
            seqs=seqs.drop([i])

    return seqs


def seq_data(b):
    data1=read_csv('data/subsample_IPTV_train.csv',b_len=b)
    data1=data_manage.since_last_time(data1)
    data2=read_csv('data/IPTV_valid_short.csv',b_len=b)
    data2=data_manage.since_last_time(data2)
    data3=read_csv('data/IPTV_test_short.csv',b_len=b)
    data3=data_manage.since_last_time(data3)
    np.save('data/IPTV_train.npy',data1)
    np.save('data/IPTV_dev.npy', data2)
    np.save('data/IPTV_test.npy',data3)

def inseq_data(b):
    # data1 = read_csv_3('data/IPTV_train.csv', b_len=b, mc=[0.9, 0.7, 0.2, 0.2, 0.7, 0.2, 1.0])
    # data1 = read_csv_3('data/IPTV_train.csv', b_len=b, mc=[0.3, 0.4, 0.5, 0.3, 0.8, 0.7, 0.5])
    # data1 = read_csv_3('data/IPTV_train.csv', b_len=b, mc=[0.2, 0.3, 0.4, 0.2, 0.5, 0.4, 0.3])
    data1 = read_csv_3('data/IPTV_train.csv', b_len=b, mc=[0.9, 0.7, 0.2, 0.7, 0.3, 0.1, 1.0])
    data1 = data_manage.since_last_time(data1)
    verify(data1, 7)

    # data2=read_csv_2('data/IPTV_valid_short.csv',b_len=b)
    # data2=data_manage.since_last_time(data2)

    data2 = read_csv_3('data/IPTV_valid_short.csv', b_len=b, mc=[0.9, 0.7, 0.2, 0.7, 0.3, 0.1, 1.0])
    # data2 = read_csv_2('data/IPTV_valid_short.csv', b_len=b)
    data2=data_manage.since_last_time(data2)
    verify(data2, 7)
    data3=read_csv_2('data/IPTV_test_short.csv',b_len=b)
    data3=data_manage.since_last_time(data3)
    np.save('data/IPTV_new_train3.npy',data1)
    np.save('data/IPTV_dev3.npy', data2)
    np.save('data/IPTV_test3.npy',data3)

def inseq_tegressor_data(b):
    mc=count_mc('data/IPTV_train.csv')

    data1=read_csv_3('data/IPTV_train.csv',b_len=b,mc=mc)
    data1=data_manage.since_last_time(data1)
    verify(data1,7)
    np.save('data/IPTV_new_train2.npy',data1)

    data2 = read_csv_3('data/IPTV_valid_short.csv', b_len=b,mc=mc)
    data2 = data_manage.since_last_time(data2)
    verify(data2, 7)
    np.save('data/IPTV_dev2.npy', data2)

    data3 = read_csv_2('data/IPTV_test_short.csv', b_len=b)
    data3 = data_manage.since_last_time(data3)
    verify(data3, 7)
    np.save('data/IPTV_test2.npy', data3)


# inseq_tegressor_data(128)
inseq_data(128)