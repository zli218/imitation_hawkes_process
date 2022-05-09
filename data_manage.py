import scipy
from scipy import io
import mat4py
import numpy as np

# some_type=[1,6,7,9,10,11,19,63,74,247]
some_type=[1,6,7,10,19,63,247]
data=mat4py.loadmat('data/IPTV.mat')
q_matrix=data['q']
allevents=data['allevents']
q=q_matrix[-35:]
new_q=[]
for i in q:
    k=[]
    for j in range(0,len(i)):
        if j in some_type:
            k.append(i[j])
    new_q.append(k)

np.save('data/feature.npy',new_q)




# print(allevents)

def tran(allevents,some_type):
    data=list()
    for i in allevents:
        i=i[0]
        if isinstance(i['nodes'],int):
            if i['nodes'] in some_type:
                i['nodes']=[i['nodes']]
                i['times']=[i['times']]
                data.append(i)
            #data=np.vstack((data,[i['nodes'],i['times']]))
        else:
            for k in range(0,len(i['nodes'])):
                if i['nodes'][k] not in some_type:
                    i['nodes'][k]=0
                    i['times'][k]=0
            while 0 in i['nodes']:
                i['nodes'].remove(0)
                i['times'].remove(0)
            if i['nodes']!=[]:
                    data.append(i)
                #data=np.vstack((data,[i['nodes'][k],i['times'][k]]))
    return data
def since_last_time(data):
    for i in data:
        if isinstance(i['nodes'],int):
            i['times']=[0]
        else:
            a=i['times']
            b=i['times'][:-1]
            i['times'][0]=0
            for k in range(1,len(a)):
                i['times'][k]=a[k]-b[k-1]
    return data
def since_last_time2(data):
    for i in data:
        first=i['times'][0]
        if isinstance(i['nodes'],int):
            i['times']=[0]
        else:
            a=i['times']
            b=i['times'][:-1]
            i['times'][0]=first
            for k in range(1,len(a)):
                i['times'][k]=a[k]-b[k-1]
    return data
def time_window(data,window):
    new_data=[]
    for i in data:
        for k in range(0,len(i['times'])):
            if i['times'][k]-i['times'][0]>window:
                i['nodes'][k]=10000
                i['times'][k]=10000
            else:
                i['times'][k]=i['times'][k]-i['times'][0]
        while 10000 in i['nodes']:
            i['nodes'].remove(10000)
            i['times'].remove(10000)
        if len(i['nodes'])>1:
            new_data.append(i)
    return new_data
def map_type(some_type,data):

    for i in data:
        for k in range(0,len(i['nodes'])):
            i['nodes'][k]=some_type.index(i['nodes'][k])
    return data




# data1=tran(allevents,some_type)
# data=time_window(data1,200)
# # data3=since_last_time(data2)
# # data=map_type(some_type,data2)
#
# np.save('data/selected_iptv_train.npy',data[0:350])
# np.save('data/selected_iptv_dev.npy', data[350:500])
# np.save('data/selected_iptv_test.npy',data[500:])
# np.save('data/selected_iptv_small_full.npy',data)







