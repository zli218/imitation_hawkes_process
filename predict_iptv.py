import time
import datetime
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
import dataloader_new
from collections import Counter
import CTLSTM
import utils
import generator
import numpy as np
import random

def predict(path):
    hidden_size=32
    type_size=7
    model_path = path
    model_gen = CTLSTM.CTLSTM(hidden_size, type_size)
    model_gen.load_state_dict(torch.load(model_path))

    settings_gen_seqs = {
        'num_seqs': 200,
        'min_len': 40,
        'max_len': 40,
        # 'max_time': 200  # max time
    }
    wa = model_gen.wa.weight.detach().numpy().T
    emb = model_gen.emb.weight.detach().numpy()
    wr = model_gen.rec.weight.detach().numpy().T
    br = model_gen.rec.bias.detach().numpy()

    settings_gen = {
        'dim_process': type_size,
        'dim_LSTM': hidden_size,
        # 'dim_states': args.DimStates,
        'seed_random': random.randint(10000, 20000),
        'path_pre_train': None,
        'sum_for_time': 0,  # Do we use total intensity for time sampling? 0 -- False; 1 -- True
        'args': None
    }

    gen_model = generator.NeuralHawkesCTLSTM(settings_gen, wa=wa, emb=emb, wr=wr, br=br)
    s = gen_model.gen_seqs(settings_gen_seqs)
    sample = utils.data_restructure(s)
    result1 = list()
    result2 = list()
    result3 = list()

    result=list()
    for seq in sample:
        for i in range(1,len(seq['times'])):
            seq['times'][i]=seq['times'][i]+seq['times'][i-1]
        seq['times'] = [x*(60/seq['times'][-1]) for x in seq['times']]
        seq['times']=np.floor(seq['times'])
        for i in range(1,len(seq['times'])):
            if seq['times'][i]==57 :
                result1.append(seq['nodes'][i])
            elif seq['times'][i]==58:
                result2.append(seq['nodes'][i])
            elif seq['times'][i]==59:
                result3.append(seq['nodes'][i])


    # print(sample)

    cnt1=Counter(result1)
    result.append(cnt1)
    cnt2 = Counter(result2)
    result.append(cnt2)
    cnt3 = Counter(result3)
    result.append(cnt3)

    return  result


def groundtruth():
    data = np.loadtxt("data/iptv/iptv.txt")
    cnt=list()
    data1=data[2462:2494]
    cnt1 = Counter(data1[:, 0].flatten())
    cnt.append(cnt1)
    data2 = data[2494:2544]
    cnt2 = Counter(data2[:, 0].flatten())
    cnt.append(cnt2)
    data3 = data[2544:2595]
    cnt3 = Counter(data3[:, 0].flatten())
    cnt.append(cnt3)

    return cnt

def rank(r,cnt,top):
    p=list()
    t=list()
    a=list()
    for i in range(0,len(r)):
        prediction=r[i].most_common(top)
        all=r[i].most_common()
        truth=cnt[i].most_common()
        prediction=np.array(prediction)
        # prediction=prediction[:,0]
        prediction = prediction[:, 0]
        truth=np.array(truth)
        truth=truth[:,0]
        p.append(prediction)
        t.append(truth)
        a.append(all)
    np.savetxt("predict/iptv_pre_all_{}.txt".format(top), a, fmt='%s')
    np.savetxt("predict/iptv_pre_{}.txt".format(top), p,fmt='%s')
    np.save("predict/iptv_truth.npy",t)
def MAP_cal(para):
    d1 = np.loadtxt("predict/iptv_pre_{}.txt".format(para))
    d2=np.load("predict/iptv_truth.npy",allow_pickle=True)
    MAP = 0
    for i in range(0,3):
        map_sub = 0
        c = 0
        for j in range(0,len(d1[i])):
            for k in range(0,len(d2[i])):
                if d1[i][j]==d2[i][k]:
                    c=c+1
                    map_sub=map_sub+c/(j+1)
                    break
        MAP=MAP+map_sub/len(d1[i])
    MAP=MAP/3
    return MAP

def MAR_cal(para):
    d1 = np.loadtxt("predict/iptv_pre_{}.txt".format(para))
    d2 = np.load("predict/iptv_truth.npy", allow_pickle=True)
    MAR = 0
    # size=[87,105,49]
    for i in range(0, 3):
        mar_sub = 0
        c = 0
        for j in range(0, len(d1[i])):
            for k in range(0, len(d2[i])):
                if d1[i][j] == d2[i][k]:
                    c=c+1
                    mar_sub = mar_sub + c/len(d2[i])
                    break
        MAR = MAR + mar_sub / len(d1[i])
    MAR = MAR / 3
    return MAR

def MMR_cal(para):
    d1 = np.loadtxt("predict/iptv_pre_{}.txt".format(para))
    d2 = np.load("predict/iptv_truth.npy", allow_pickle=True)
    MMR=0
    for i in range(0,3):
        mmr_sub=0
        for j in range(0, len(d1[i])):
            for k in range(0, min(len(d1[i]),len(d2[i]))):
                if d1[i][j] == d2[i][k]:
                    mmr_sub=mmr_sub+1/(j+1)
        MMR=MMR+mmr_sub/len(d1[i])
    MMR=MMR/3
    return MMR


r=predict(path='model_mle_iptv_top1.pkl')
cnt=groundtruth()
para_list=[1,2,3,5]
for para in para_list:
    rank(r,cnt,para)
    x=MAP_cal(para)
    print("map:")
    print(x)
    y=MAR_cal(para)
    print("mar:")
    print(y)
    z=MMR_cal(para)
    print("mmr:")
    print(z)
    with open("predict/prediction_iptv_rl.txt", 'a') as l:
        l.write("prediction MAP top{} :{}\n".format(para,x))
        l.write("prediction MAR top{} :{}\n".format(para,y))
        l.write("prediction MRR top{} :{}\n".format(para, z))
