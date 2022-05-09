import time
import datetime
import torch
import torch.optim as opt
from torch.utils.data import DataLoader

import dataloader
import dataloader_new
import CTLSTM
import utils
import generator
import numpy as np
import random




# def new_incomp(seqs,mc):
#     incomp=[]
#     idx=[]
#     # print(mc)
#
#     sigmc = torch.nn.Sigmoid()
#     m_c = 1 - sigmc(mc)
#     # print(m_c)
#     for seq in seqs:
#         sub_idx=[]
#         for i in range(len(seq['nodes'])):
#             u=np.random.uniform()
#             if u<m_c[seq['nodes'][i]]:
#                 seq['nodes'][i]=100
#                 seq['times'][i]=100
#                 sub_idx.append(i)
#         while 100 in seq['nodes']:
#             seq['nodes'].remove(100)
#             seq['times'].remove(100)
#         incomp.append(seq)
#         idx.append(sub_idx)
#     return seqs,idx,mc



def train(settings):
    """Training process."""
    hidden_size = settings['hidden_size']#
    type_size = settings['type_size']#
    train_path = settings['train_path']
    dev_path = settings['dev_path']
    batch_size = settings['batch_size']#
    batch_size_real=settings['batch_size_real']
    epoch_num = settings['epoch_num']#
    current_date = settings['current_date']
    test_path = settings['test_path']
    gamma=settings['gamma']
###################################################
    mc = settings['mc']
    mc = np.array(mc)
    mc = torch.from_numpy(mc)

    train_loss_history=[]
    val_loss_history=[]
    test_loss_history=[]



#################################################


    model = CTLSTM.CTLSTM(hidden_size, type_size)
    model_gen = CTLSTM.CTLSTM(hidden_size, type_size)
    model2 = CTLSTM.CTLSTM(hidden_size, type_size)
    model_gen.load_state_dict(torch.load('model1.pkl'))
    lr=1e-3
    betas = [0.9, 0.999]
    optim = opt.Adam(model.parameters(),lr=lr)
    M_c = torch.autograd.Variable(mc, requires_grad=True)






    train_dataset=dataloader_new.IPTVDataset(train_path)
    dev_dataset=dataloader_new.IPTVDataset(dev_path)
    #######################################################################
    train_dataloader=DataLoader(train_dataset,batch_size=batch_size_real, collate_fn=dataloader_new.pad_batch_fn, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, collate_fn=dataloader_new.pad_batch_fn, shuffle=True)

    test_dataset = dataloader_new.IPTVDataset(test_path)
    test_dataloader = DataLoader(test_dataset, collate_fn=dataloader_new.pad_batch_fn, shuffle=True)

    last_dev_loss = 0.0

    settings_gen_seqs = {
        'num_seqs': batch_size,
        'min_len': 128,
        'max_len': 128,
        'max_time': 200  # max time
    }
    for epoch in range(epoch_num):
        tic_epoch = time.time()
        epoch_train_loss = 0.0
        epoch_dev_loss = 0.0
        epoch_test_loss = 0.0
        train_event_num = 0
        dev_event_num = 0
        test_event_num = 0
        print('Epoch.{} starts.'.format(epoch))
        tic_train = time.time()
        for i_batch, sample_batched in enumerate(train_dataloader):

            # if i_batch%2==1:
                tic_batch = time.time()
                optim.zero_grad()
                #if i_batch == 0:
                if True:

                    wa = model_gen.wa.weight.detach().numpy().T
                    emb = model_gen.emb.weight.detach().numpy()
                    wr = model_gen.rec.weight.detach().numpy().T
                    br = model_gen.rec.bias.detach().numpy()

                # else:
                #     wa = model.wa.weight.detach().numpy().T
                #     emb = model.emb.weight.detach().numpy()
                #     wr = model.rec.weight.detach().numpy().T
                #     br = model.rec.bias.detach().numpy()

            ###########################
                settings_gen = {
                    'dim_process': type_size,
                    'dim_LSTM': hidden_size,
                    # 'dim_states': args.DimStates,
                    'seed_random': random.randint(10000, 20000),
                    'path_pre_train': None,
                    'sum_for_time': 0,  # Do we use total intensity for time sampling? 0 -- False; 1 -- True
                    'args': None

                }

                gen_model = generator.NeuralHawkesCTLSTM(settings_gen,wa=wa,emb=emb,wr=wr,br=br)
                s = gen_model.gen_seqs(settings_gen_seqs)
                sample = utils.data_restructure(s)
                #TODO:sk
                ######################################################################

                event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)

                sk=sample
                sk_dataset = dataloader_new.FakeDataset(sk)
                sk_dataloader = DataLoader(sk_dataset, batch_size=batch_size, collate_fn=dataloader_new.pad_batch_fn,
                                           shuffle=True)
                for batch, sample_batched in enumerate(sk_dataloader):
                    fake_event_seqs, fake_time_seqs, fake_total_time_seqs, fake_seqs_length = utils.pad_bos(
                        sample_batched, model.type_size)
                    fake_sim_time_seqs, fake_sim_index_seqs = utils.generate_sim_time_seqs(fake_time_seqs,
                                                                                           fake_seqs_length)
                    torch.save(model.state_dict(),"model_save.pkl")
                    model.forward(fake_event_seqs, fake_time_seqs)
                    h1=model.h_d
                    model2.load_state_dict(torch.load('model_save.pkl'))
                    model2.forward(event_seqs, time_seqs)
                    h2=model2.h_d


                sk_sub, sk_sub_idx,M_c = utils.new_incomp(sample,M_c)
                sk_sub_dataset = dataloader_new.FakeDataset(sk_sub)
                sk_sub_dataloader = DataLoader(sk_sub_dataset, batch_size=batch_size, collate_fn=dataloader_new.pad_batch_fn,
                                           shuffle=True)
                for batch, sample_batched in enumerate(sk_sub_dataloader):
                    fake_sub_event_seqs, fake_sub_time_seqs, _, _ = utils.pad_bos(
                        sample_batched, model.type_size)
                    # fake_sim_time_seqs, fake_sim_index_seqs = utils.generate_sim_time_seqs(fake_time_seqs,
                    #                                                                        fake_seqs_length)


                #TODO:sk_sub_event


                # real_event=event_seqs
                # real_time=time_seqs
                #reward1=utils.reward_by_type(real_event,real_time,fake_sub_event_seqs,fake_sub_time_seqs,settings_gen_seqs['max_time'],gamma)

                reward1=utils.reward_emb(h1,h2,gamma)
                if i_batch==0:
                    with open("loss_reward1.txt", 'a') as l:
                        l.write("Epoch.{}Batch.{}\nreward {}:\n".format(epoch, i_batch, reward1))
                likelihood = model.reward_log_likelihood(reward1,M_c,sub_idx=sk_sub_idx,fake_seqs=fake_event_seqs,
                                                         sim_time_seqs=fake_sim_time_seqs, sim_index_seqs=fake_sim_index_seqs,
                                                         total_time_seqs=fake_total_time_seqs, fake_seqs_length=fake_seqs_length)

                ##########################################################
                batch_event_num = torch.sum(seqs_length)
                batch_loss = -likelihood

                batch_loss.backward(retain_graph=True)
                optim.step()



                toc_batch = time.time()
                if i_batch % 100 == 1:
                    print('Epoch.{} Batch.{}:\nBatch Likelihood per event: {:5f} nats\nTrain Time: {:2f} s'
                          .format(epoch, i_batch, likelihood / batch_event_num, toc_batch - tic_batch))
                # print('Epoch.{} Batch.{}:\nBatch Likelihood per event: {:5f} nats\nTrain Time: {:2f} s'
                #            .format(epoch, i_batch, likelihood / batch_event_num, toc_batch - tic_batch))
                epoch_train_loss += batch_loss
                train_event_num += batch_event_num


            #TODO modify the output

        train_loss_history.append(float(-epoch_train_loss/train_event_num))
        toc_train = time.time()
        print('---\nEpoch.{} Training set\nTrain Likelihood per event: {:5f} nats\nTrainig Time:{:2f} s'.format(epoch, -epoch_train_loss/train_event_num, toc_train-tic_train))

        #TODO validition
        ################################################################################################################
        tic_eval = time.time()

        for i_batch, sample_batched in enumerate(dev_dataloader):
            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length)
            dev_event_num += torch.sum(seqs_length)
            epoch_dev_loss -= likelihood
            # gap = epoch_dev_loss/dev_event_num - last_dev_loss
            # if abs(gap) < 1e-4:
            #     print('Final log likelihood: {} nats'.format(-epoch_dev_loss/dev_event_num))
            #     break
            #
            # last_dev_loss = epoch_dev_loss/dev_event_num
        val_loss_history.append(float(-epoch_dev_loss/dev_event_num))
        toc_eval = time.time()
        # print(M_c)
        print('Epoch.{} Devlopment set\nDev Likelihood per event: {:5f} nats\nEval Time:{:2f}s.\n'.format(epoch, -epoch_dev_loss/dev_event_num, toc_eval-tic_eval))
        ################################################################################################################
        if True:#epoch==epoch_num-1:
            # # todo test
            tic_test = time.time()
            for i_batch, sample_batched in enumerate(test_dataloader):
                event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
                sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
                model.forward(event_seqs, time_seqs)
                likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs, seqs_length)

                test_event_num += torch.sum(seqs_length)
                epoch_test_loss -= likelihood

            test_loss_history.append(float(-epoch_test_loss/test_event_num))
            toc_test = time.time()

            print('Epoch.{} Test set\nTest Likelihood per event: {:5f} nats\nTest Time:{:2f}s.\n'.format(epoch,
                                                                                                         -epoch_test_loss / test_event_num,
                                                                                                         toc_test - tic_test))

        toc_epoch = time.time()
        #TODO output
        with open("loss_{}_rl_fix_mc.txt".format(current_date), 'a') as l:
            if epoch==0:
                l.write("Batch_size_real : {}\nBatch_size_fake :{}\nmc :{}\nlr :{}\nbetas :{}\n".format(batch_size_real,
                                                                                            batch_size,mc,lr,betas))
            l.write("Epoch {}:\n".format(epoch))
            #l.write("MC {}:\n".format(output_mc))
            l.write("Train Event Number:\t\t{}\n".format(train_event_num))
            l.write("Train Likelihood per event:\t{:.5f}\n".format(-epoch_train_loss/train_event_num))
            l.write("Training time:\t\t\t{:.2f} s\n".format(toc_train-tic_train))
            l.write("Dev Event Number:\t\t{}\n".format(dev_event_num))
            l.write("Dev Likelihood per event:\t{:.5f}\n".format(-epoch_dev_loss/dev_event_num))
            l.write("Dev evaluating time:\t\t{:.2f} s\n".format(toc_eval-tic_eval))
            if True: #epoch == epoch_num - 1:
                l.write("Test Event Number:\t\t{}\n".format(test_event_num))
                l.write("Test Likelihood per event:\t{:.5f}\n".format(-epoch_test_loss / test_event_num))
                l.write("Test evaluating time:\t\t{:.2f} s\n".format(toc_test - tic_test))
            l.write("Epoch time:\t\t\t{:.2f} s\n".format(toc_epoch-tic_epoch))
            l.write("\n")



        # gap = epoch_dev_loss/dev_event_num - last_dev_loss
        # if abs(gap) < 1e-4:
        #     print('Final log likelihood: {} nats'.format(-epoch_dev_loss/dev_event_num))
        #     break

        # last_dev_loss = epoch_dev_loss/dev_event_num
    # np.save('plot_result/test.npy',test_loss_history)
    # np.save('plot_result/train.npy', train_loss_history)
    # np.save('plot_result/val.npy', val_loss_history)
    name1='plot_result/test1.jpg'
    name2='plot_result/test2.jpg'
    utils.loss_plot(train_loss_history, test_loss_history, val_loss_history, epoch_num, name1,name2)
    torch.save(model.state_dict(),'model2.pkl')
    
    return


if __name__ == "__main__":


    settings = {
        'hidden_size': 32,
        'type_size': 7,
        'train_path': 'data/IPTV_new_train.npy',
        'dev_path': 'data/IPTV_dev.npy',
        'test_path': 'data/IPTV_test.npy',
        # 'train_path': 'data/selected_iptv_train_incomplete.npy',
        # 'dev_path': 'data/selected_iptv_dev.npy',
        'batch_size': 5,
        'batch_size_real':1,
        'epoch_num': 500,
        'current_date': datetime.date.today(),
        'mc':[0.9,0.7,0.2,0.2,0.7,0.2,1.0],
        #  'mc':[0.5,0.5,0.5,0.5,0.5,0.5,0.5],
        # 'mc': [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0],
        # 'mc': [2.4, 0.8, -1.3, -1.3, 0.8, -1.3, 7.0],
        'gamma': 0.1
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

    train(settings)