# -*- coding: utf-8 -*-
"""A continuous time LSTM network."""
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from torch.utils.data import DataLoader
import numpy as np

class CTLSTM(nn.Module):
    """Continuous time LSTM network with decay function."""
    def __init__(self, hidden_size, type_size, batch_first=True):
        super(CTLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.type_size = type_size
        self.batch_first = batch_first
        self.num_layers = 1

        # Parameters
        # recurrent cells
        self.rec = nn.Linear(2*self.hidden_size, 7*self.hidden_size)#input 32*64 output 32*224
        # output mapping from hidden vectors to unnormalized intensity
        self.wa = nn.Linear(self.hidden_size, self.type_size,bias=False)
        # embedding layer for valid events, including BOS
        self.emb = nn.Embedding(self.type_size+1, self.hidden_size)

    def init_states(self, batch_size):
        self.h_d = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        self.c_d = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        self.c_bar = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)
        self.c = torch.zeros(batch_size, self.hidden_size, dtype=torch.float)

    def recurrence(self, emb_event_t, h_d_tm1, c_tm1, c_bar_tm1):
        feed = torch.cat((emb_event_t, h_d_tm1), dim=1)
        # B * 2H
        (gate_i,
        gate_f,
        gate_z,
        gate_o,
        gate_i_bar,
        gate_f_bar,
        gate_delta) = torch.chunk(self.rec(feed), 7, -1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_z = torch.tanh(gate_z)
        gate_o = torch.sigmoid(gate_o)
        gate_i_bar = torch.sigmoid(gate_i_bar)
        gate_f_bar = torch.sigmoid(gate_f_bar)
        gate_delta = F.softplus(gate_delta)

        c_t = gate_f * c_tm1 + gate_i * gate_z
        c_bar_t = gate_f_bar * c_bar_tm1 + gate_i_bar * gate_z

        return c_t, c_bar_t, gate_o, gate_delta

    def decay(self, c_t, c_bar_t, o_t, delta_t, duration_t):
        c_d_t = c_bar_t + (c_t - c_bar_t) * \
            torch.exp(-delta_t * duration_t.view(-1,1))

        h_d_t = o_t * torch.tanh(c_d_t)

        return c_d_t, h_d_t

    # todo:reward by the emmbeding layer output: get_emb

    def forward(self, event_seqs, duration_seqs, batch_first = True):
        if batch_first:
            event_seqs = event_seqs.transpose(0,1)
            duration_seqs = duration_seqs.transpose(0,1)
        
        batch_size = event_seqs.size()[1]
        batch_length = event_seqs.size()[0]

        h_list, c_list, c_bar_list, o_list, delta_list = [], [], [], [], []

        for t in range(batch_length):
            self.init_states(batch_size)
            c, self.c_bar, o_t, delta_t = self.recurrence(self.emb(event_seqs[t]), self.h_d, self.c_d, self.c_bar)
            self.c_d, self.h_d = self.decay(c, self.c_bar, o_t, delta_t, duration_seqs[t])
            h_list.append(self.h_d)
            c_list.append(c)
            c_bar_list.append(self.c_bar)
            o_list.append(o_t)
            delta_list.append(delta_t)
        # print(self.h_d)
        h_seq = torch.stack(h_list)
        c_seq = torch.stack(c_list)
        c_bar_seq = torch.stack(c_bar_list)
        o_seq = torch.stack(o_list)
        delta_seq = torch.stack(delta_list)
        
        self.output = torch.stack((h_seq, c_seq, c_bar_seq, o_seq, delta_seq))
        return self.output
    # def log_likelihood_mc(self, event_seqs,sim_time_seqs,idx,):
    # def log_likelihood_mc(self):
    #     for idx, (event_seq, seq_len) in enumerate(zip(event_seqs_fake_short, seqs_length)):
    #         original_loglikelihood[idx] = torch.sum(torch.log(
    #             lambda_k[idx, torch.arange(seq_len).long(), event_seq[1:seq_len+1]]))
    #     simulated_like=simulate_like*mc
    #     loglikelihood = torch.sum(original_loglikelihood - simulated_likelihood)
    def reward_log_likelihood(self, reward,mc,sub_idx,fake_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, fake_seqs_length, batch_first=True):
        """Calculate log likelihood per sequence."""
        #reward=torch.from_numpy(reward)
        batch_size, batch_length = fake_seqs.shape

        h, c, c_bar, o, delta = torch.chunk(self.output, 5, 0)
        # L * B * H
        h = torch.squeeze(h, 0)#len*32*32
        c = torch.squeeze(c, 0)
        c_bar = torch.squeeze(c_bar, 0)
        o = torch.squeeze(o, 0)
        delta = torch.squeeze(delta, 0)

        # Calculate the sum of log intensities of each event in the sequence
        original_loglikelihood = torch.zeros(batch_size)
        original_lambda_k = F.softplus(self.wa(h)).transpose(0, 1)

        sigmc = torch.nn.Sigmoid()
        mc = sigmc(mc)
        # mc=(torch.exp(mc)-1)/(torch.exp(torch.tensor(1.0))-1)
        # mc = 2 / (1+torch.exp(-mc.mul(torch.tensor(2.0))))-1
        # lambda_k=original_lambda_k.mul(mc)
        lambda_k=original_lambda_k


        lambda_k_copy=lambda_k.clone()
        #TODO:finish history
        for idx, (event_seq,seq_len) in enumerate(zip(fake_seqs,fake_seqs_length)):

            # for i in range(seq_len):
            #     if i not in sub_idx[idx]:
            #         lambda_k_copy[idx, torch.arange(seq_len).long(), event_seq[i + 1]] = torch.log(lambda_k_copy[idx, torch.arange(seq_len).long(), event_seq[i + 1]])
            #     else :
            #         lambda_k_copy[idx,torch.arange(seq_len).long(),event_seq[i+1]]=0
            # x=torch.sum(
            #     lambda_k_copy[idx, torch.arange(seq_len).long(), event_seq[1:seq_len+1]])
            # original_loglikelihood[idx]=x
            sum=0
            for i  in range(seq_len):
                if i not in sub_idx[idx]:
                    k=torch.log( lambda_k_copy[idx, torch.arange(seq_len).long(), event_seq[1:seq_len+1]][i])
                    sum=sum+k
            original_loglikelihood[idx]=sum
        # for idx, (event_seq, seq_len) in enumerate(zip(fake_seqs, fake_seqs_length)):
        #     original_loglikelihood[idx] = torch.sum(torch.log(
        #         lambda_k[idx, torch.arange(seq_len).long(), event_seq[1:seq_len+1]]))

        # Calculate simulated loss from MCMC method
        h_d_list = []
        if batch_first:
            sim_time_seqs = sim_time_seqs.transpose(0,1)
        for idx, sim_duration in enumerate(sim_time_seqs):
            _, h_d_idx = self.decay(c[idx], c_bar[idx], o[idx], delta[idx], sim_duration)
            h_d_list.append(h_d_idx)
        h_d = torch.stack(h_d_list)

        sim_lambda_k = F.softplus(self.wa(h_d)).transpose(0,1)
        sim_lambda_k=sim_lambda_k.mul(mc)
        simulated_likelihood = torch.zeros(batch_size)
        for idx, (total_time, seq_len) in enumerate(zip(total_time_seqs, fake_seqs_length)):
            mc_coefficient = total_time / (seq_len)
            simulated_likelihood[idx] = mc_coefficient * torch.sum(torch.sum(sim_lambda_k[idx, torch.arange(seq_len).long(), :]))

        #########################################################################
        loglikelihood = torch.sum(reward.mul(original_loglikelihood - simulated_likelihood))
        #######################################################################################
        return loglikelihood
    def log_likelihood(self, event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length, batch_first=True):
        """Calculate log likelihood per sequence."""
        batch_size, batch_length = event_seqs.shape
        h, c, c_bar, o, delta = torch.chunk(self.output, 5, 0)
        # L * B * H
        h = torch.squeeze(h, 0)#len*32*32
        c = torch.squeeze(c, 0)
        c_bar = torch.squeeze(c_bar, 0)
        o = torch.squeeze(o, 0)
        delta = torch.squeeze(delta, 0)

        # Calculate the sum of log intensities of each event in the sequence
        original_loglikelihood = torch.zeros(batch_size)
        lambda_k = F.softplus(self.wa(h)).transpose(0, 1)

        for idx, (event_seq, seq_len) in enumerate(zip(event_seqs, seqs_length)):
            original_loglikelihood[idx] = torch.sum(torch.log(
                lambda_k[idx, torch.arange(seq_len).long(), event_seq[1:seq_len+1]]))

        # Calculate simulated loss from MCMC method
        h_d_list = []
        if batch_first:
            sim_time_seqs = sim_time_seqs.transpose(0,1)
        for idx, sim_duration in enumerate(sim_time_seqs):
            _, h_d_idx = self.decay(c[idx], c_bar[idx], o[idx], delta[idx], sim_duration)
            h_d_list.append(h_d_idx)
        h_d = torch.stack(h_d_list)

        sim_lambda_k = F.softplus(self.wa(h_d)).transpose(0,1)
        simulated_likelihood = torch.zeros(batch_size)
        for idx, (total_time, seq_len) in enumerate(zip(total_time_seqs, seqs_length)):
            mc_coefficient = total_time / (seq_len)
            simulated_likelihood[idx] = mc_coefficient * torch.sum(torch.sum(sim_lambda_k[idx, torch.arange(seq_len).long(), :]))

        loglikelihood = torch.sum(original_loglikelihood - simulated_likelihood)
        return loglikelihood
    # def log_likelihood(self, event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length, batch_first=True):
    #     """Calculate log likelihood per sequence."""
    #     batch_size, batch_length = event_seqs.shape
    #     h, c, c_bar, o, delta = torch.chunk(self.output, 5, 0)
    #     # L * B * H
    #     h = torch.squeeze(h, 0)
    #     c = torch.squeeze(c, 0)
    #     c_bar = torch.squeeze(c_bar, 0)
    #     o = torch.squeeze(o, 0)
    #     delta = torch.squeeze(delta, 0)
    #
    #     # Calculate the sum of log intensities of each event in the sequence
    #     original_loglikelihood = torch.zeros(batch_size)
    #     lambda_k = F.softplus(self.wa(h)).transpose(0, 1)
    #
    #     for idx, (event_seq, seq_len) in enumerate(zip(event_seqs, seqs_length)):
    #         original_loglikelihood[idx] = torch.mul(lambda_k[idx, torch.arange(seq_len).long(), event_seq[1:seq_len+1]])/
    #
    #     # Calculate simulated loss from MCMC method
    #     h_d_list = []
    #     if batch_first:
    #         sim_time_seqs = sim_time_seqs.transpose(0,1)
    #     for idx, sim_duration in enumerate(sim_time_seqs):
    #         _, h_d_idx = self.decay(c[idx], c_bar[idx], o[idx], delta[idx], sim_duration)
    #         h_d_list.append(h_d_idx)
    #     h_d = torch.stack(h_d_list)
    #
    #     sim_lambda_k = F.softplus(self.wa(h_d)).transpose(0,1)
    #     simulated_likelihood = torch.zeros(batch_size)
    #     for idx, (total_time, seq_len) in enumerate(zip(total_time_seqs, seqs_length)):
    #         mc_coefficient = total_time / (seq_len)
    #         simulated_likelihood[idx] = mc_coefficient * torch.sum(torch.sum(sim_lambda_k[idx, torch.arange(seq_len).long(), :]))
    #
    #     loglikelihood = torch.sum(original_loglikelihood - simulated_likelihood)
    #     return loglikelihood

class Discriminator(nn.Module):
    def __init__(self,size):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, event_seqs):
        size=event_seqs.size()[1]
        validity = self.model(event_seqs,size)

        return validity