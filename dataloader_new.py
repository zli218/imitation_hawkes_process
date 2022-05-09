# -*- coding: utf-8 -*-
"""Dataloader for neural hawkes process.

Args:
    path: file path for the dataset
    batch_size: size of one batch

Results:
    A Dataset class
"""

import pickle
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

class IPTVDataset(Dataset):
    ''' Dataset class for neural hawkes data
    '''
    def __init__(self, file_path):
        self.file_path = file_path
        self.event_seqs = []
        self.time_seqs = []

        seqs=np.load(file_path,allow_pickle=True)


        for seq in seqs:
            # if idx == 1:
            #     print(seq[0].keys())
            self.event_seqs.append(torch.LongTensor(seq['nodes']))
            self.time_seqs.append(torch.FloatTensor(seq['times']))

    def __len__(self):
        return len(self.event_seqs)
    
    def __getitem__(self, index):
        sample = {
            'event_seq': self.event_seqs[index],
            'time_seq': self.time_seqs[index]
        }

        return sample


class FakeDataset(Dataset):
    ''' Dataset class for neural hawkes data
    '''

    def __init__(self, fake_data):
        self.fake=fake_data
        self.event_seqs = []
        self.time_seqs = []

        seqs = fake_data

        for seq in seqs:
            # if idx == 1:
            #     print(seq[0].keys())
            self.event_seqs.append(torch.LongTensor(seq['nodes']))
            self.time_seqs.append(torch.FloatTensor(seq['times']))

    def __len__(self):
        return len(self.event_seqs)

    def __getitem__(self, index):
        sample = {
            'event_seq': self.event_seqs[index],
            'time_seq': self.time_seqs[index]
        }

        return sample

class IPTV_NEW_Dataset(Dataset):
    '''
    Dataset class for merge data
    '''

    #todo df data dataloder

    def __init__(self, file_path):
        self.file_path = file_path
        self.event_seqs = []
        self.time_seqs = []

        data=pd.read_csv(file_path)
        data=data.sort_values(by=['time'],ascending=True)
        e=data['typeIdx'].tolist()
        t=data['time'].tolist()
        for i in range(len(e)):
            self.event_seqs.append(e[i])
            self.time_seqs.append(t[i])


        #todo


        # seqs = np.load(file_path, allow_pickle=True)
        #
        # for seq in seqs:
        #     # if idx == 1:
        #     #     print(seq[0].keys())
        #     self.event_seqs.append(torch.LongTensor(seq['nodes']))
        #     self.time_seqs.append(torch.FloatTensor(seq['times']))

    def __len__(self):
        return len(self.event_seqs)

    def __getitem__(self, index):
        sample = {
            'event': self.event_seqs[index],
            'time': self.time_seqs[index]
        }

        return sample

def pad_batch_fn(batch_data):
    sorted_batch = sorted(batch_data, key=lambda x: x['event_seq'].size(), reverse=True)
    event_seqs = [seq['event_seq'].long() for seq in sorted_batch]
    time_seqs = [seq['time_seq'].float() for seq in sorted_batch]
    seqs_length = torch.LongTensor(list(map(len, event_seqs)))
    last_time_seqs = torch.stack([torch.sum(time_seq) for time_seq in time_seqs])

    event_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).long()
    time_seqs_tensor = torch.zeros(len(sorted_batch), seqs_length.max()).float()

    for idx, (event_seq, time_seq, seqlen) in enumerate(zip(event_seqs, time_seqs, seqs_length)):
        event_seqs_tensor[idx, :seqlen] = torch.LongTensor(event_seq)
        time_seqs_tensor[idx, :seqlen] = torch.FloatTensor(time_seq)
    
    return event_seqs_tensor, time_seqs_tensor, last_time_seqs, seqs_length

# def restore_batch(sample_batched, type_size):
#     event_seqs, time_seqs, seqs_length = sample_batched

#     event_seqs_list, time_seqs_list = [], []
#     total_time_list = []

#     for idx, (event_seq, time_seq, seq_length) in enumerate(zip(event_seqs, time_seqs, seqs_length)):
#         tmp_event_seq = torch.ones(seq_length + 1, dtype=torch.int32) * type_size
#         tmp_event_seq[1:] = event_seq[:seq_length]
#         event_seqs_list.append(tmp_event_seq)

#         tmp_time_seq = torch.zeros(seq_length + 1, dtype=torch.float)
#         tmp_time_seq[1:] = time_seq[:seq_length]
#         time_seqs_list.append(tmp_time_seq)

#         total_time_list.append(torch.sum(tmp_time_seq))
    
#     return event_seqs_list, time_seqs_list, total_time_list
