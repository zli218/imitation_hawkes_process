import time
import datetime
import torch
import torch.optim as opt
from torch.utils.data import DataLoader
import dataloader_new
import CTLSTM
import utils
import generator
import numpy as np
import random



#############################################################################
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
    model_path = 'model_mle_iptv_top1.pkl'
##############################################)#####
    mc = settings['mc']
    mc = np.array(mc)
    mc = torch.from_numpy(mc)

    train_loss_history = []
    val_loss_history = []
    test_loss_history = []


#################################################


    model = CTLSTM.CTLSTM(hidden_size, type_size)
    model_gen=CTLSTM.CTLSTM(hidden_size, type_size)
    model2 = CTLSTM.CTLSTM(hidden_size, type_size)
    model_gen.load_state_dict(torch.load(model_path))
    lr=5e-4
    lr2=1e-4
    betas=[0.9,0.999]
    optim = opt.Adam(model.parameters(),lr=lr,betas=betas)
    M_c = torch.autograd.Variable(mc, requires_grad=True)
    optim2 = opt.SGD([M_c], lr=lr2)

    # M_c=[0.5,0.5,0.5,0.5,0.5]
    # print('initial lambda and M_c,',M_c)



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
        'min_len': 20,
        'max_len': 40,
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
                # if i_batch==0:
                if True:
                    wa = model_gen.wa.weight.detach().numpy().T
                    emb = model_gen.emb.weight.detach().numpy()
                    wr = model_gen.rec.weight.detach().numpy().T
                    br = model_gen.rec.bias.detach().numpy()
                #     event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
                #
                #     model.forward(event_seqs, time_seqs)
                else:
                    wa = model.wa.weight.detach().numpy().T
                    emb = model.emb.weight.detach().numpy()
                    wr = model.rec.weight.detach().numpy().T
                    br = model.rec.bias.detach().numpy()

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
                    # model.forward(fake_event_seqs, fake_time_seqs)

                    torch.save(model.state_dict(), "model_save2.pkl")
                    model.forward(fake_event_seqs, fake_time_seqs)
                    h1 = model.h_d
                    model2.load_state_dict(torch.load('model_save2.pkl'))
                    model2.forward(event_seqs, time_seqs)
                    h2 = model2.h_d


                sk_sub, sk_sub_idx,M_c = utils.new_incomp(sample,M_c)
                # print(M_c)
                sk_sub_dataset = dataloader_new.FakeDataset(sk_sub)
                sk_sub_dataloader = DataLoader(sk_sub_dataset, batch_size=batch_size, collate_fn=dataloader_new.pad_batch_fn,
                                           shuffle=True)

                for batch, sample_batched in enumerate(sk_sub_dataloader):
                    fake_sub_event_seqs, fake_sub_time_seqs, _, _ = utils.pad_bos(
                        sample_batched, model.type_size)
                    # fake_sim_time_seqs, fake_sim_index_seqs = utils.generate_sim_time_seqs(fake_time_seqs,
                    #                                                                        fake_seqs_length)


                #TODO:sk_sub_event
                sk_sub_event=0

                # real_event=event_seqs
                # real_time=time_seqs
                reward1 = utils.reward_emb(h1, h2, gamma)
                # with open("loss_reward1.txt", 'a') as l:
                #     l.write("Epoch.{}Batch.{}\nreward {}:\n".format(epoch, i_batch, reward1))

                # reward1 = utils.new_reward(real_event, real_time, fake_sub_event_seqs, fake_sub_time_seqs,
                #                        settings_gen_seqs['max_time'], gamma)
                # reward1 = utils.reward_by_type(real_event, real_time, fake_sub_event_seqs, fake_sub_time_seqs,
                #                    settings_gen_seqs['max_time'], gamma)
                # print(reward1)
                # reward=utils.new_reward(real_event,real_time,fake_sub_event_seqs,fake_sub_time_seqs,settings_gen_seqs['max_time'],gamma)
                # with open("loss_{}_reward1.txt".format(datetime.date.today()), 'a') as l:
                #     l.write("Epoch.{}Batch.{}\nreward {}:\n".format(epoch,i_batch,reward1))
                likelihood = model.reward_log_likelihood(reward1,M_c,sub_idx=sk_sub_idx,fake_seqs=fake_event_seqs,
                                                         sim_time_seqs=fake_sim_time_seqs, sim_index_seqs=fake_sim_index_seqs,
                                                         total_time_seqs=fake_total_time_seqs, fake_seqs_length=fake_seqs_length)

                ##########################################################
                batch_event_num = torch.sum(seqs_length)
                batch_loss = -likelihood

                batch_loss.backward(retain_graph=True)
                optim.step()
                toc_batch = time.time()
                # if i_batch % 100 == 1:
                #     print('Epoch.{} Batch.{}:\nBatch Likelihood per event: {:5f} nats\nTrain Time: {:2f} s'
                #           .format(epoch, i_batch, likelihood / batch_event_num, toc_batch - tic_batch))
                # print('Epoch.{} Batch.{}:\nBatch Likelihood per event: {:5f} nats\nTrain Time: {:2f} s'
                #            .format(epoch, i_batch, likelihood / batch_event_num, toc_batch - tic_batch))
                epoch_train_loss += batch_loss
                train_event_num += batch_event_num
            ####################################################################################
            # if i_batch%2==0:
                tic_batch = time.time()
                optim2.zero_grad()

                wa = model.wa.weight.detach().numpy().T
                emb = model.emb.weight.detach().numpy()
                wr = model.rec.weight.detach().numpy().T
                br = model.rec.bias.detach().numpy()
                gen_model = generator.NeuralHawkesCTLSTM(settings_gen, wa=wa, emb=emb, wr=wr, br=br)
                s2 = gen_model.gen_seqs(settings_gen_seqs)
                sample2 = utils.data_restructure(s2)

                #event_seqs2, time_seqs2, total_time_seqs2, seqs_length2 = utils.pad_bos(sample_batched, model.type_size)

                sk2 = sample2
                sk_dataset2 = dataloader_new.FakeDataset(sk2)
                sk_dataloader2 = DataLoader(sk_dataset2, batch_size=batch_size, collate_fn=dataloader_new.pad_batch_fn,
                                           shuffle=True)
                for batch, sample_batched in enumerate(sk_dataloader2):
                    fake_event_seqs2, fake_time_seqs2, fake_total_time_seqs2, fake_seqs_length2 = utils.pad_bos(
                        sample_batched, model.type_size)
                    fake_sim_time_seqs2, fake_sim_index_seqs2 = utils.generate_sim_time_seqs(fake_time_seqs2,
                                                                                       fake_seqs_length2)

                    torch.save(model.state_dict(), "model_save2.pkl")
                    model.forward(fake_event_seqs2, fake_time_seqs2)
                    h1 = model.h_d
                    model2.load_state_dict(torch.load('model_save2.pkl'))
                    model2.forward(event_seqs, time_seqs)
                    h2 = model2.h_d


                sk_sub2, sk_sub_idx2,M_c = utils.new_incomp(sample,M_c)


                sk_sub_dataset2 = dataloader_new.FakeDataset(sk_sub2)
                sk_sub_dataloader2 = DataLoader(sk_sub_dataset2, batch_size=batch_size,
                                               collate_fn=dataloader_new.pad_batch_fn,
                                               shuffle=True)
                for batch, sample_batched in enumerate(sk_sub_dataloader2):
                    fake_sub_event_seqs2, fake_sub_time_seqs2, _, _ = utils.pad_bos(
                        sample_batched, model.type_size)
                    # fake_sim_time_seqs, fake_sim_index_seqs = utils.generate_sim_time_seqs(fake_time_seqs,
                    #                                                                        fake_seqs_length)

                # real_event2 = event_seqs
                # real_time2 = time_seqs
                # reward2=utils.reward_by_type(real_event2, real_time2, fake_sub_event_seqs2, fake_sub_time_seqs2,settings_gen_seqs['max_time'],gamma)
                reward2=utils.reward_emb(h1,h2,gamma)
                # with open("loss_{}_reward2.txt".format(datetime.date.today()), 'a') as l:
                #     l.write("Epoch.{}Batch.{}\nreward {}:\n".format(epoch,i_batch,reward2))
                likelihood2= model.reward_log_likelihood(reward2, M_c, sk_sub_idx2,fake_event_seqs2, fake_sim_time_seqs2,
                                                     fake_sim_index_seqs2, fake_total_time_seqs2, fake_seqs_length2)

                batch_event_num2 = torch.sum(seqs_length)
                batch_loss2=-likelihood2
                batch_loss2.backward(retain_graph=True)
                optim2.step()

                # output_mc = (torch.exp(mc) - 1) / (torch.exp(torch.tensor(1.0)) - 1)
                # output_mc= 2 / (1 + torch.exp(-mc.mul(torch.tensor(2.0)))) - 1
                sigmc_after = torch.nn.Sigmoid()
                output_mc = sigmc_after(M_c)

                # print("after ",M_c)

                toc_batch = time.time()
                if i_batch % 100 == 0:
                    print('Epoch.{} Batch.{}:\nBatch Likelihood per event: {:5f} nats\nTrain Time: {:2f} s\n '
                          .format(epoch,i_batch,likelihood2 / batch_event_num2,toc_batch - tic_batch),)
                # print('Epoch.{} Batch.{}:\nBatch Likelihood per event: {:5f} nats\nTrain Time: {:2f} s\n msissing rate:'
                #       .format(epoch, i_batch, likelihood2 / batch_event_num2, toc_batch - tic_batch), mc)
                # epoch_train_loss += batch_loss2
                # train_event_num += batch_event_num2
            #########################################################

            #TODO modify the output

        train_loss_history.append(float(-epoch_train_loss / train_event_num))
        toc_train = time.time()
        print('---\nEpoch.{} Training set\nTrain Likelihood per event: {:5f} nats\nTrainig Time:{:2f} s'.format(epoch, -epoch_train_loss/train_event_num, toc_train-tic_train))

        #TODO validition
        ################################################################################################################
        tic_eval = time.time()

        for i_batch, sample_batched in enumerate(dev_dataloader):

            event_seqs, time_seqs, total_time_seqs, seqs_length = utils.pad_bos(sample_batched, model.type_size)
            sim_time_seqs, sim_index_seqs = utils.generate_sim_time_seqs(time_seqs, seqs_length)
            model.forward(event_seqs, time_seqs)
            likelihood = model.log_likelihood(event_seqs, sim_time_seqs, sim_index_seqs, total_time_seqs,
                                                  seqs_length)
            dev_event_num += torch.sum(seqs_length)
            epoch_dev_loss -= likelihood
        val_loss_history.append(float(-epoch_dev_loss / dev_event_num))
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
            test_loss_history.append(float(-epoch_test_loss / test_event_num))
            toc_test = time.time()

            print('Epoch.{} Test set\nTest Likelihood per event: {:5f} nats\nTest Time:{:2f}s.\n'.format(epoch,
                                                                                                         -epoch_test_loss / test_event_num,
                                                                                                         toc_test - tic_test))

        toc_epoch = time.time()
        #TODO output
        with open("loss_{}_rl_iptv_top1.txt".format(current_date), 'a') as l:
            if epoch==0:
                l.write("Batch_size_real : {}\nBatch_size_fake :{}\nnetwork_lr :{}\nmc_lr :{}\nbetas :{}\n".format(batch_size_real,
                                                                                            batch_size,lr,lr2,betas))
            l.write("Epoch {}:\n".format(epoch))
            # l.write("MC_not_sigmoid {}:\n".format(M_c))
            l.write("MC {}:\n".format(output_mc))
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
    name1 = 'plot_result/rl1_{}.jpg'.format(current_date)
    name2 = 'plot_result/rl2_{}.jpg'.format(current_date)
    utils.loss_plot(train_loss_history, test_loss_history, val_loss_history, epoch_num, name1, name2)

    torch.save(model.state_dict(),'model_rl_houston.pkl')

    return


if __name__ == "__main__":


    settings = {
        'hidden_size': 32,
        'type_size': 7,
        # 'train_path': 'data/data_retweet/incomp_train.npy',
        # 'dev_path': 'data/data_retweet/incomp_dev.npy',
        # 'test_path': 'data/data_retweet/incomp_test.npy',
        # 'train_path': 'data/chicago/train.npy',
        # 'dev_path': 'data/chicago/dev.npy',
        # 'test_path': 'data/chicago/test.npy',
        # 'train_path': 'data/houston/train_test.npy',
        # 'dev_path': 'data/houston/dev_test.npy',
        # 'test_path': 'data/houston/test_test.npy',
        # 'train_path': 'data/dallas/train.npy',
        # 'dev_path': 'data/dallas/dev.npy',
        # 'test_path': 'data/dallas/test.npy',
        # 'train_path': 'data/iptv/train.npy',
        # 'dev_path': 'data/iptv/dev.npy',
        # 'test_path': 'data/iptv/test.npy',
        # 'train_path': 'data/houston/train_top1.npy',
        # 'dev_path': 'data/houston/dev_top1.npy',
        # 'test_path': 'data/houston/test_top1.npy',
        'train_path': 'data/simulation/simulation.npy',
        'dev_path': 'data/simulation/simulation.npy',
        'test_path': 'data/simulation/simulation.npy',
        'batch_size': 1,
        'batch_size_real':1,
        'epoch_num': 200,
        'current_date': datetime.date.today(),
        'mc':np.random.rand(7),
        'gamma': 0.1
    }


    train(settings)

