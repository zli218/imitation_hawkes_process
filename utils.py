"""Utility functions for CTLSTM model."""

import torch
import numpy as np
import matplotlib.pyplot as plt

#todo: simple method to 0-1
def mapping_to(mc):

    sigmc = torch.nn.Sigmoid()
    m_c = sigmc(mc)


    #todo simple but more efficient map function

    # mc=mc.detach().numpy()
    # m_c=(np.exp(mc)-1)/(np.exp(1)-1)

    return m_c

def new_incomp(seqs_,mc):
    seqs=seqs_.copy()

    incomp=[]
    idx=[]
    # print(mc)



    sigmc = torch.nn.Sigmoid()
    m_c = 1 - sigmc(mc)
    # m_c=1-(2 / (1+torch.exp(-mc.mul(torch.tensor(2.0))))-1)
    # m_c=(np.exp(mc)-1)/(np.exp(1)-1)



    # print(m_c)
    for seq in seqs:
        sub_idx=[]
        for i in range(len(seq['nodes'])):
            u=np.random.uniform()
            if u<m_c[seq['nodes'][i]]:
                seq['nodes'][i]=1000
                seq['times'][i]=1000
                sub_idx.append(i)
        while 1000 in seq['nodes']:
            seq['nodes'].remove(1000)
            seq['times'].remove(1000)
        incomp.append(seq)
        idx.append(sub_idx)
    return seqs,idx,mc



###############################################################
def data_restructure(seqs):

    data = list()
    for s in seqs:
        k = {}
        k['nodes'] = []
        k['times'] = []
        for d in s:
            k['nodes'].append(d['type_event'])
            k['times'].append(d['time_since_last_event'])
        k['times'][0] = 0.0
        #########################################
        # a=numpy.array(k['times'])
        # a=a*200/sum(a)
        # #print(sum(a))
        # k['times']=a.tolist()
        data.append(k)
    return data
def reward_emb(emb1,emb2,gamma):
    e1=emb1.detach().numpy().copy()#fake
    e2=emb2.detach().numpy().copy()#real
    reward = []

    for i in e1:
        # reward=reward+(i-e2).dot((i-e2).T)
        #reward.append(np.sum(np.abs((i-e2))))
        reward_row=[]
        for j in e2:
            reward_row.append(float((i - j).dot((i - j).T)))
        # reward.append(float(np.sum(np.abs((i-e2)))))
        reward.append(reward_row)
    reward=np.mat(reward)
    reward=np.sum(reward,axis=1)
    X_norm = np.array(reward) ** 2
    K = np.exp(-gamma * X_norm)



    return torch.tensor(K)


def new_reward(realevent,realtime,fake_event,fake_time,T,gamma):
    r1=reward_by_type(fake_event,fake_time,fake_event,fake_time,T,gamma)
    r2=reward_by_type(realevent,realtime,fake_event,fake_time,T,gamma)
    reward = r2 - r1
    X_norm = np.array(reward) ** 2
    K = np.exp(-100* X_norm)
    return torch.tensor(K)

def reward_by_type(realevent,realtime,fake_event,fake_time,T,gamma):
    reward=reward_by_type_core(realevent,realtime,fake_event,fake_time,T)
    X_norm = np.array(reward) ** 2
    K = np.exp(-gamma * X_norm)
    return torch.tensor(K)

def reward_by_type_core(realevent,realtime,fake_event,fake_time,T):
    time = realtime.detach().numpy().copy()
    event = realevent.detach().numpy().copy()
    fake_time = fake_time.detach().numpy().copy()
    fake_event = fake_event.detach().numpy().copy()

    for i in time:
        for j in range(1, len(i)):
            i[j] = i[j] + i[j - 1]
    for i in fake_time:
        for j in range(1, len(i)):
            i[j] = i[j] + i[j - 1]
    # todo rescale
    # for i in range(0,len(fake_time)):
    #     fake_time[i]=fake_time[i]/fake_time[i][-1]*200
    # with open("loss_reward1.txt", 'a') as l:
    #
    #     l.write("real_event {}:\n".format( event))
    #     l.write("real_time {}:\n".format(time))
    #     l.write("fake_event {}:\n".format( fake_event))
    #     l.write("fake_time {}:\n".format( fake_time))
    reward = []
    for e, t in zip(event, time):
        sub_reward = []
        for k, q in zip(fake_event, fake_time):
            sub_reward_type = []
            for type in set(k):
                sub_fake_event = []
                sub_fake_time = []
                sub_real_event = []
                sub_real_time = []
                for i in range(0, len(k)):
                    if k[i] == type:
                        sub_fake_event.append(k[i])
                        sub_fake_time.append(q[i])
                for i in range(0, len(t)):
                    if e[i] == type:
                        sub_real_event.append(e[i])
                        sub_real_time.append(t[i])
                min_len = min(len(sub_real_time), len(sub_fake_time))
                # sub_diff=[abs(sub_real_time-sub_fake_time) for i in
                #     range(min_len)]
                sub_diff = []
                for i in range(min_len):
                    sub_diff.append(abs(sub_real_time[i] - sub_fake_time[i]))

                sub_extra = sub_fake_time[min_len:] if len(sub_fake_time) > min_len else sub_real_time[min_len:]
                # print(sub_extra)
                sub_reward_fake = (sum(sub_diff) + abs(len(sub_real_time) - len(sub_fake_time)) * T
                                   - sum(sub_extra[i] for i in range(len(sub_extra))))
                sub_reward_type.append(sub_reward_fake)
            sub_reward.append(float(sum(sub_reward_type)))
        reward.append(sub_reward)
    reward = np.array(reward)
    reward = np.sum(reward, axis=0)
    return  reward
# def reward_by_type_bad(realevent,realtime,fake_event,fake_time,T,gamma):
#     reward=reward_by_type_core_bad(realevent,realtime,fake_event,fake_time,T)
#     X_norm = np.array(reward) ** 2
#     K = np.exp(-gamma * X_norm)
#     return torch.tensor(K)
# def reward_by_type_core_bad(realevent,realtime,fake_event,fake_time,T):
#     time = [[0.0,1.0,2.0,3.0]]
#     event = [[0,3,4,5]]
#     fake_time = fake_time.detach().numpy().copy()
#     fake_event = fake_event.detach().numpy().copy()
#
#     for i in time:
#         for j in range(1, len(i)):
#             i[j] = i[j] + i[j - 1]
#     for i in fake_time:
#         for j in range(1, len(i)):
#             i[j] = i[j] + i[j - 1]
#     # todo rescale
#     # for i in range(0,len(fake_time)):
#     #     fake_time[i]=fake_time[i]/fake_time[i][-1]*200
#     # with open("loss_reward1.txt", 'a') as l:
#     #
#     #     l.write("real_event {}:\n".format( event))
#     #     l.write("real_time {}:\n".format(time))
#     #     l.write("fake_event {}:\n".format( fake_event))
#     #     l.write("fake_time {}:\n".format( fake_time))
#     reward = []
#     for e, t in zip(event, time):
#         sub_reward = []
#         for k, q in zip(fake_event, fake_time):
#             sub_reward_type = []
#             for type in set(k):
#                 sub_fake_event = []
#                 sub_fake_time = []
#                 sub_real_event = []
#                 sub_real_time = []
#                 for i in range(0, len(k)):
#                     if k[i] == type:
#                         sub_fake_event.append(k[i])
#                         sub_fake_time.append(q[i])
#                 for i in range(0, len(t)):
#                     if e[i] == type:
#                         sub_real_event.append(e[i])
#                         sub_real_time.append(t[i])
#                 min_len = min(len(sub_real_time), len(sub_fake_time))
#                 # sub_diff=[abs(sub_real_time-sub_fake_time) for i in
#                 #     range(min_len)]
#                 sub_diff = []
#                 for i in range(min_len):
#                     sub_diff.append(abs(sub_real_time[i] - sub_fake_time[i]))
#
#                 sub_extra = sub_fake_time[min_len:] if len(sub_fake_time) > min_len else sub_real_time[min_len:]
#                 # print(sub_extra)
#                 sub_reward_fake = (sum(sub_diff) + abs(len(sub_real_time) - len(sub_fake_time)) * T
#                                    - sum(sub_extra[i] for i in range(len(sub_extra))))
#                 sub_reward_type.append(sub_reward_fake)
#             sub_reward.append(float(sum(sub_reward_type)))
#         reward.append(sub_reward)
#     reward = np.array(reward)
#     reward = np.sum(reward, axis=0)
#     return  reward

def generate_sim_time_seqs(time_seqs, seqs_length):
    """
    Generate a simulated time interval sequences from original time interval sequences based on uniform distribution
    
    Args:
        time_seqs: list of torch float tensors
    Results:
        sim_time_seqs: list of torch float tensors
        sim_index_seqs: list of torch long tensors
    """
    sim_time_seqs = torch.zeros((time_seqs.size()[0], time_seqs.size()[1]-1)).float()
    sim_index_seqs = torch.zeros((time_seqs.size()[0], time_seqs.size()[1]-1)).long()
    restore_time_seqs, restore_sim_time_seqs = [], []
    for idx, time_seq in enumerate(time_seqs):
        restore_time_seq = torch.stack([torch.sum(time_seq[0:i]) for i in range(1,seqs_length[idx]+1)])

        restore_sim_time_seq, _ = torch.sort(torch.empty(seqs_length[idx]-1).uniform_(0, restore_time_seq[-1]))
        
        sim_time_seq = torch.zeros(seqs_length[idx]-1)
        sim_index_seq = torch.zeros(seqs_length[idx]-1).long()

        for idx_t, t in enumerate(restore_time_seq):
            indices_to_update = restore_sim_time_seq > t

            sim_time_seq[indices_to_update] = restore_sim_time_seq[indices_to_update] - t
            sim_index_seq[indices_to_update] = idx_t

        restore_time_seqs.append(restore_time_seq)
        restore_sim_time_seqs.append(restore_sim_time_seq)
        sim_time_seqs[idx, :seqs_length[idx]-1] = sim_time_seq
        sim_index_seqs[idx, :seqs_length[idx]-1] = sim_index_seq

    return sim_time_seqs, sim_index_seqs


def pad_bos(batch_data, type_size):
    event_seqs, time_seqs, total_time_seqs, seqs_length = batch_data
    pad_event_seqs = torch.zeros((event_seqs.size()[0], event_seqs.size()[1]+1)).long() * type_size
    pad_time_seqs = torch.zeros((time_seqs.size()[0], event_seqs.size()[1]+1)).float()

    pad_event_seqs[:, 1:] = event_seqs.clone()
    pad_event_seqs[:, 0] = type_size
    pad_time_seqs[:, 1:] = time_seqs.clone()

    return pad_event_seqs, pad_time_seqs, total_time_seqs, seqs_length




if __name__ == '__main__':
    a = torch.tensor([0., 1., 2., 3., 4., 5.])
    b = torch.tensor([0., 2., 4., 6., 0., 0.])

    sim_time_seqs, sim_index_seqs, restore_time_seqs, restore_sim_time_seqs =\
        generate_sim_time_seqs(torch.stack([a,b]), torch.LongTensor([6,4]))

def loss_plot(train,test,val,epoch,name1,name2):
    x=list(range(0,epoch))

    plt.plot(x,val)
    plt.plot(x,test)
    plt.legend()
    plt.savefig(name1)
    plt.show()

    plt.plot(x,train)
    plt.savefig(name2)
    plt.show()



