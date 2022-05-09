# -*- coding: utf-8 -*-

"""Training code for neural hawkes model."""
import time
import datetime
import torch
import torch.optim as opt
from torch.utils.data import DataLoader

import dataloader
import dataloader_new
import CTLSTM
import utils


def train(settings):
    """Training process."""
    hidden_size = settings['hidden_size']#
    type_size = settings['type_size']#
    train_path = settings['train_path']
    dev_path = settings['dev_path']
    batch_size = settings['batch_size']#
    epoch_num = settings['epoch_num']#
    current_date = settings['current_date']
    test_path=settings['test_path']

    model = CTLSTM.CTLSTM(hidden_size, type_size)
    optim = opt.Adam(model.parameters())

    # M_c=[0.5,0.5,0.5,0.5,0.5]
    # print('initial lambda and M_c,',M_c)
    train_loss_history = []
    val_loss_history = []
    test_loss_history = []

    """
    modified
    """
    # train_dataset = dataloader.CTLSTMDataset(train_path)
    # dev_dataset = dataloader.CTLSTMDataset(dev_path)
    #
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=dataloader.pad_batch_fn, shuffle=True)
    # dev_dataloader = DataLoader(dev_dataset, collate_fn=dataloader.pad_batch_fn, shuffle=True)

    train_dataset=dataloader_new.IPTVDataset(train_path)
    dev_dataset=dataloader_new.IPTVDataset(dev_path)
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size, collate_fn=dataloader_new.pad_batch_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, collate_fn=dataloader_new.pad_batch_fn, shuffle=True)
    test_dataset = dataloader_new.IPTVDataset(test_path)
    test_dataloader = DataLoader(test_dataset, collate_fn=dataloader_new.pad_batch_fn, shuffle=True)

    last_dev_loss = 0.0

    for epoch in range(epoch_num):
        tic_epoch = time.time()
        epoch_train_loss = 0.0
        epoch_dev_loss = 0.0
        epoch_test_loss = 0.0
        train_event_num = 0
        dev_event_num = 0
        test_event_num=0
        print('Epoch.{} starts.'.format(epoch))
        tic_train = time.time()
        for i_batch, sample_batched in enumerate(train_dataloader):
            tic_batch = time.time()
            
            optim.zero_grad()
            
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length)
            batch_event_num = torch.sum(seqs_length)
            batch_loss = -likelihood

            batch_loss.backward()
            optim.step()
            
            toc_batch = time.time()
            if i_batch % 100 == 0:
                print('Epoch.{} Batch.{}:\nBatch Likelihood per event: {:5f} nats\nTrain Time: {:2f} s'.format(epoch, i_batch, likelihood/batch_event_num, toc_batch-tic_batch))
            epoch_train_loss += batch_loss
            train_event_num += batch_event_num

        train_loss_history.append(float(-epoch_train_loss / train_event_num))
        toc_train = time.time()
        print('---\nEpoch.{} Training set\nTrain Likelihood per event: {:5f} nats\nTrainig Time:{:2f} s'.format(epoch, -epoch_train_loss/train_event_num, toc_train-tic_train))

        tic_eval = time.time()
        for i_batch, sample_batched in enumerate(dev_dataloader):
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs,seqs_length)
            
            dev_event_num += torch.sum(seqs_length)
            epoch_dev_loss -= likelihood

        val_loss_history.append(float(-epoch_dev_loss / dev_event_num))
        toc_eval = time.time()

        print('Epoch.{} Devlopment set\nDev Likelihood per event: {:5f} nats\nEval Time:{:2f}s.'.format(epoch, -epoch_dev_loss/dev_event_num, toc_eval-tic_eval))

        #todo test
        tic_test = time.time()
        for i_batch, sample_batched in enumerate(test_dataloader):
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length)

            test_event_num += torch.sum(seqs_length)
            epoch_test_loss -= likelihood

        test_loss_history.append(float(-epoch_test_loss / test_event_num))
        toc_test = time.time()

        print('Epoch.{} Test set\nTest Likelihood per event: {:5f} nats\nTest Time:{:2f}s.\n'.format(epoch,
                                                                                                          -epoch_test_loss / test_event_num,
                                                                                                          toc_test - tic_test))
        toc_epoch = time.time()


        with open(settings['result_saving'].format(current_date), 'a') as l:
            l.write("Epoch {}:\n".format(epoch))
            l.write("Train Event Number:\t\t{}\n".format(train_event_num))
            l.write("Train Likelihood per event:\t{:.5f}\n".format(-epoch_train_loss/train_event_num))
            l.write("Training time:\t\t\t{:.2f} s\n".format(toc_train-tic_train))
            l.write("Dev Event Number:\t\t{}\n".format(dev_event_num))
            l.write("Dev Likelihood per event:\t{:.5f}\n".format(-epoch_dev_loss/dev_event_num))
            l.write("Dev evaluating time:\t\t{:.2f} s\n".format(toc_eval-tic_eval))
            l.write("Test Event Number:\t\t{}\n".format(test_event_num))
            l.write("Test Likelihood per event:\t{:.5f}\n".format(-epoch_test_loss / test_event_num))
            l.write("Test evaluating time:\t\t{:.2f} s\n".format(toc_test - tic_test))
            l.write("Epoch time:\t\t\t{:.2f} s\n".format(toc_epoch-tic_epoch))
            l.write("\n")
        
        # gap = epoch_dev_loss/dev_event_num - last_dev_loss
        # if abs(gap) < 1e-1:
        #     print('Final log likelihood: {} nats'.format(-epoch_dev_loss/dev_event_num))
        #     break
        #
        # last_dev_loss = epoch_dev_loss/dev_event_num
    name1 = 'plot_result/mle1_{}.jpg'.format(current_date)
    name2 = 'plot_result/mle2_{}.jpg'.format(current_date)
    utils.loss_plot(train_loss_history, test_loss_history, val_loss_history, epoch_num, name1, name2)

    torch.save(model.state_dict(),settings['model_saving'])
    
    return


if __name__ == "__main__":
    settings = {
        'hidden_size': 32,
        'type_size': 7,
        # 'train_path': 'data/selected_iptv_train_incomplete.npy',
        # 'dev_path': 'data/selected_iptv_dev.npy',
        # 'batch_size': 32,
        # 'train_path': 'data/data_retweet/incomp_train.npy',
        # 'dev_path': 'data/data_retweet/incomp_dev.npy',
        # 'test_path': 'data/data_retweet/incomp_test.npy',
        # 'train_path': 'data/houston/train_top1.npy',
        # 'dev_path': 'data/houston/dev_top1.npy',
        # 'test_path': 'data/houston/test_top1.npy',
        # 'train_path': 'data/iptv/train_top1.npy',
        # 'dev_path': 'data/iptv/dev_top1.npy',
        # 'test_path': 'data/iptv/test_top1.npy',
        # 'train_path': 'data/houston/train.npy',
        # 'dev_path': 'data/houston/dev.npy',
        # 'test_path': 'data/houston/test.npy',
        # 'train_path': 'data/iptv/train.npy',
        # 'dev_path': 'data/iptv/dev.npy',
        # 'test_path': 'data/iptv/test.npy',
        # 'train_path': 'data/chicago/train.npy',
        # 'dev_path': 'data/chicago/dev.npy',
        # 'test_path': 'data/chicago/test.npy',
        # 'train_path': 'data/dallas/train.npy',
        # 'dev_path': 'data/dallas/dev.npy',
        # 'test_path': 'data/dallas/test.npy',
        # 'train_path': 'data/dallas/train_top1.npy',
        # 'dev_path': 'data/dallas/dev_top1.npy',
        # 'test_path': 'data/dallas/test_top1.npy',
        'train_path': 'data/IPTV_new_train3.npy',
        'dev_path': 'data/IPTV_dev3.npy',
        'test_path': 'data/IPTV_test3.npy',
        'batch_size':32,
        'epoch_num': 1000,
        'current_date': datetime.date.today(),
        # 'test_path':'data/incomp_iptv_test.npy',
    }
    # settings = {
    #     'hidden_size': 32,
    #     'type_size': 3,
    #     'train_path': 'data/data_retweet/train.pkl',
    #     'dev_path': 'data/data_retweet/dev.pkl',
    #     'batch_size': 32,
    #     'epoch_num': 100,
    #     'current_date': datetime.date.today()
    # }
    settings2 = {
        'hidden_size': 32,
        'type_size': 5,
        'train_path': 'data/simulation/random025_train.npy',
        'dev_path': 'data/simulation/random025_dev.npy',
        'test_path': 'data/simulation/random025_test.npy',
        'model_saving': 'model/simualtion_raondom_mle.pkl',
        'result_saving': "simulation_result/loss_{}_mle_random025.txt",
        'batch_size': 32,
        'epoch_num': 200,
        'current_date': datetime.date.today(),

    }
    settings3 = {
        'hidden_size': 32,
        'type_size': 5,
        'train_path': 'data/simulation/type_train.npy',
        'dev_path': 'data/simulation/type_dev.npy',
        'test_path': 'data/simulation/type_test.npy',
        'model_saving': 'model/simualtion_type_mle.pkl',
        'result_saving': "simulation_result/loss_{}_mle_type.txt",
        'batch_size': 32,
        'epoch_num': 200,
        'current_date': datetime.date.today(),

    }

    train(settings2)
    train(settings3)
