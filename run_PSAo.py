import sys
import argparse
import glob
from loader import *
from model import *
from utils import *
import numpy as np
import os
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler


def minmax(cam):
    cam_min = np.min(cam)
    cam = cam - cam_min
    cam_max = np.max(cam)
    cam = cam / (1e-7 + cam_max)
    return cam

def get_network(opt, network, savedmodelname=None):
    parallel = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if parallel:
        network = nn.DataParallel(network).to(device)
        if opt.pretrained_weight:
            print("Model is using pretrained weights from the paper")
            pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
            pretrained_dir = './model_weights'
            pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
            print(pretrained_path)
            assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check.\n" \
                                                "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
            network.load_state_dict(torch.load(pretrained_path))
        else:
            if savedmodelname is not None:
                network.load_state_dict(torch.load(savedmodelname))

    # else:
    #     if opt.pretrained_weight:
    #         print("Model is using pretrained weights from the paper")
    #         pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
    #         pretrained_dir = './model_weights'
    #         pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
    #         assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check.\n" \
    #                                             "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
    #         state_dict = torch.load(pretrained_path)
    #         # remap to handle w/o DataParallel:  a new state_dict by removing 'module.' prefix
    #         new_state_dict = {}
    #         for key in state_dict.keys():
    #             if key.startswith("module."):
    #                 new_key = key.replace('module.', '')  # Remove 'module.' from the keys

    #             new_state_dict[new_key] = state_dict[key]

    #         # Load the updated state_dict into your model
    #         network.load_state_dict(new_state_dict)
    #     else:
    #         network.load_state_dict(torch.load(savedmodelname))

    #     if cuda:
    #         network = network.cuda()

    return network

def train(network, opt):
    cuda = True
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # parallel = True
    # device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(f"{opt.output_fullname}/", exist_ok=True)

    # if parallel:
    #     network = nn.DataParallel(network).to(device)
    #     if opt.pretrained_weight:
    #         print("Model is using pretrained weights from the paper")
    #         pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
    #         pretrained_dir = './model_weights'
    #         pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
    #         assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check. \n" \
    #                                                 "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
    #         network.load_state_dict(torch.load(pretrained_path))
    # else:
    #     if opt.pretrained_weight:
    #         print("Model is using pretrained weights from the paper")
    #         pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
    #         pretrained_dir = './model_weights'
    #         pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
    #         assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check.\n" \
    #                                                 "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
    #         state_dict = torch.load(pretrained_path)
    #         # remap to handle w/o DataParallel:  a new state_dict by removing 'module.' prefix
    #         new_state_dict = {}
    #         for key in state_dict.keys():
    #             if key.startswith("module."):
    #                 new_key = key.replace('module.', '')  # Remove 'module.' from the keys
    #             new_state_dict[new_key] = state_dict[key]
    #     network = network.cuda()

    if opt.path_pretrained_model is None:
        network = get_network(opt, network)
    else:
        network = get_network(opt, network, opt.path_pretrained_model)
        print("!!! Continue training from ", opt.path_pretrained_model)

    if opt.epoch > 0:
        if len(glob.glob(f"{opt.output_fullname}/epoch{opt.epoch - 1}*.pth")) > 0:
            lastpointname = glob.glob(f"{opt.output_fullname}/epoch{opt.epoch - 1}*.pth")[0]
            network.load_state_dict(torch.load(lastpointname))
        else:
            bestepoch = np.loadtxt(os.path.join(f'' + opt.output_fullname, 'best.info'))
            bestpointname = glob.glob(f"{opt.output_fullname}/model_best.pth")[0]
            network.load_state_dict(torch.load(bestpointname))
            opt.epoch = int(bestepoch)

    optimizer = torch.optim.Adam(network.parameters(), lr=opt.lr, betas=(0.9, 0.999))
    if opt.lrscheduler is not None: 
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lrscheduler[0], gamma=opt.lrscheduler[1])
        print('lr_scheduler.StepLR is set: ', opt.lrscheduler)
    steps_per_epoch = opt.save_epoch_num
    writer = SummaryWriter(log_dir=f"{opt.output_fullname}")
    prev_time = time.time()
    prev_val_loss = 1e+100
    earlystoppingcount = 0

    loader_train = torch.utils.data.DataLoader(args.train_loader,
                                               batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers,
                                               drop_last=True)

    loader_val = torch.utils.data.DataLoader(args.val_loader,
                                             batch_size=opt.batchsize, shuffle=True, num_workers=opt.num_workers,
                                             drop_last=True)

    for epoch in range(opt.epoch, opt.max_epoch):

        if earlystoppingcount > opt.earlystopping:
            break

        epoch_total_loss = []
        epoch_step_time = []
        epoch_total_loss_o, epoch_total_loss_psa = [], []

        for step, batch in enumerate(loader_train):
            step_start_time = time.time()

            # if len(args.optional_meta) > 0:
            #     I1, I2 = batch
            #     input1, target1, meta1 = I1
            #     input2, target2, meta2 = I2
            #     predicted1, predicted2 = network(input2.type(Tensor), input1.type(Tensor),
            #                         meta = [meta2.type(Tensor), meta1.type(Tensor)])

            # else:
            I1, I2 = batch
            input1, target_o_1, target_psa_1  = I1
            input2, target_o_2, target_psa_2 = I2
            predicted1, predicted2 = network(input2.type(Tensor), input1.type(Tensor))

            targetdiff_o = (target_o_2 - target_o_1)[:, None].type(Tensor)
            targetdiff_psa = torch.zeros(targetdiff_o.shape).type(Tensor)
            for xx in range(targetdiff_psa.shape[0]):
                if target_psa_1[xx] == 1 and target_psa_2[xx] == 1: targetdiff_psa[xx] = 1
                elif target_psa_1[xx] == 0 and target_psa_2[xx] == 0: targetdiff_psa[xx] = 0
                else:
                    print('target_psa_1, target_psa_2: ', target_psa_1, target_psa_2)
                    exit()
          
            if opt.task_option == 'o':
                targetdiff_o[targetdiff_o > 0] = 1
                targetdiff_o[targetdiff_o == 0] = 0.5
                targetdiff_o[targetdiff_o < 0] = 0
                
            # Loss
            optimizer.zero_grad()
            loss_o = args.loss(predicted1, targetdiff_o)
            loss_psa = args.loss(predicted2, targetdiff_psa)
            if args.alpha_oloss is None:
                loss = loss_o + loss_psa
            else:
                loss = loss_o * args.alpha_oloss + loss_psa * (1-args.alpha_oloss)
            loss.backward()
            optimizer.step()
            epoch_total_loss.append(loss.item())
            epoch_total_loss_o.append(loss_o.item())
            epoch_total_loss_psa.append(loss_psa.item())
            
            # Log Progress
            batches_done = epoch * len(loader_train) + step
            batches_left = opt.max_epoch * len(loader_train) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [ loss: %f, loss_o: %f , loss_psa: %f] ETA: %s"
                % (
                    epoch,
                    opt.max_epoch,
                    step,
                    len(loader_train),
                    loss.item(),
                    loss_o.item(),
                    loss_psa.item(),
                    time_left,
                )
            )
            epoch_step_time.append(time.time() - step_start_time)

        if opt.lrscheduler is not None: 
            prev_lr = scheduler.get_last_lr()
            scheduler.step()
            print('\nLr is updated by lr_scheduler.StepLR. {}->{}'.format(prev_lr, scheduler.get_last_lr()))

        if ((epoch + 1) % steps_per_epoch == 0):  # (step != 0) &
            # print epoch info
            epoch_info = '\nValidating... Step %d/%d / Epoch %d/%d' % (
                step, len(loader_train), epoch, opt.max_epoch)
            time_info = '%.4f sec/step' % np.mean(epoch_step_time)
            loss_info = 'train loss: %.4e ' % (np.mean(epoch_total_loss))

            log_stats([np.mean(epoch_total_loss)], ['loss/train'], epoch, writer)
            log_stats([np.mean(epoch_total_loss_o)], ['loss/train_o'], epoch, writer)
            log_stats([np.mean(epoch_total_loss_psa)], ['loss/train_psa'], epoch, writer)

            network.eval()
            valloss_total = []
            valloss_total_o, valloss_total_psa = [], []
            for valstep, batch in enumerate(loader_val):
                # if len(args.optional_meta) > 0:
                #     I1, I2 = batch
                #     input1, target1, meta1 = I1
                #     input2, target2, meta2 = I2
                #     predicted = network(input2.type(Tensor), input1.type(Tensor),
                #                         meta = [meta2.type(Tensor), meta1.type(Tensor)])

                # else:
                I1, I2 = batch
                input1, target_o_1, target_psa_1  = I1
                input2, target_o_2, target_psa_2 = I2
                predicted1, predicted2 = network(input2.type(Tensor), input1.type(Tensor))

                targetdiff_o = (target_o_2 - target_o_1)[:, None].type(Tensor)
                targetdiff_psa = torch.zeros(targetdiff_o.shape).type(Tensor)
                for xx in range(targetdiff_psa.shape[0]):
                    if target_psa_1[xx] == 1 and target_psa_2[xx] == 1: targetdiff_psa[xx] = 1
                    elif target_psa_1[xx] == 0 and target_psa_2[xx] == 0: targetdiff_psa[xx] = 0
                    else:
                        print('target_psa_1, target_psa_2: ', target_psa_1, target_psa_2)
                        exit()
                if opt.task_option == 'o':
                    targetdiff_o[targetdiff_o > 0] = 1
                    targetdiff_o[targetdiff_o == 0] = 0.5
                    targetdiff_o[targetdiff_o < 0] = 0

                valloss_o = args.loss(predicted1, targetdiff_o)
                valloss_psa = args.loss(predicted2, targetdiff_psa)
                if args.alpha_oloss is None:
                    valloss = valloss_o + valloss_psa
                else:
                    valloss = valloss_o * args.alpha_oloss + valloss_psa * (1-args.alpha_oloss)
                valloss_total.append(valloss.item())
                valloss_total_o.append(valloss_o.item())
                valloss_total_psa.append(valloss_psa.item())

            log_stats([np.mean(valloss_total)], ['loss/val'], epoch, writer)
            log_stats([np.mean(valloss_total_o)], ['loss/val_o'], epoch, writer)
            log_stats([np.mean(valloss_total_psa)], ['loss/val_psa'], epoch, writer)

            val_loss_info = 'val loss: %.4e, o: %.4e, psa: %.4e' % (np.mean(valloss_total), np.mean(valloss_total_o), np.mean(valloss_total_psa))
            print(' - '.join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)
            curr_val_loss = np.mean(valloss_total)
            if prev_val_loss > curr_val_loss:
                torch.save(network.state_dict(),
                           f"{opt.output_fullname}/model_best.pth")
                np.savetxt(f"{opt.output_fullname}/model_best.info", np.array([epoch]))
                prev_val_loss = curr_val_loss
                earlystoppingcount = 0  # New bottom
            else:
                earlystoppingcount += 1
                print(f'Early stopping count: {earlystoppingcount}')

            network.train()

    torch.save(network.state_dict(), f"{opt.output_fullname}/model_epoch{epoch}.pth")
    network.eval()


def get_score(opt, resultfilename):
    import sklearn.metrics as metrics
    from scipy.stats import pearsonr
    sigmoid = nn.Sigmoid()

    # def rmse(a, b):
    #     return metrics.mean_squared_error(a, b, squared=False)

    dict_metric = {'auc': metrics.roc_auc_score, 'pearson': pearsonr,
                   'rmse': metrics.root_mean_squared_error, 'loss': opt.loss, 'mse':metrics.mean_squared_error}
    dict_task_metrics = {'o': ['loss', 'auc', 'acc'],
                         't': ['loss', 'rmse', 'mse'],
                         's': ['loss', 'rmse']}

    result = pd.read_csv(resultfilename)
    target_diff = np.array(result['target'])
    feature_diff = np.array(result['predicted'])
    target1 = np.array(result['target1'])
    target2 = np.array(result['target2'])
    
    for dtm in dict_task_metrics[args.task_option]:
        if dtm == 'auc' and args.task_option == 'o':
            print(f'warning: {dtm.upper()} calculated only for binary pairs')
            feature_diff_auc = sigmoid(torch.tensor(feature_diff)).numpy()
            print(f'{dtm.upper()}: {dict_metric[dtm](target_diff[target_diff != 0.5], feature_diff_auc[target_diff != 0.5]):.3}')

            # ### AUC, target interval
            # sig_interval_dict = {}
            # for t1, t2, f in zip(target1, target2, feature_diff_auc):
            #     interval = (t2-t1).item()

            #     if interval == 0: continue

            #     if interval not in sig_interval_dict: 
            #         sig_interval_dict[interval]=[]
                
            #     sig_interval_dict[interval].append(f.item())
            
            # for interval in sorted(sig_interval_dict.keys()):
            #     sig_interval = np.array(sig_interval_dict[interval])
            #     print(sig_interval)

            #     if interval < 0: target_class = np.array([0 for x in range(len(sig_interval))])
            #     elif interval == 0: target_class = np.array([0.5 for x in range(len(sig_interval))])
            #     elif interval > 0: target_class = np.array([1 for x in range(len(sig_interval))])

            #     print(f'interval {str(interval)}: \
            #           auc={dict_metric[dtm](target_class[target_class != 0.5], sig_interval[target_class != 0.5]):.3} \
            #           , num={len(sig_interval[target_class != 0.5])}')

        elif dtm == 'acc' and args.task_option == 'o':
            ### Accuracy
            pred_class = []
            for f in feature_diff:
                if f < 0: pred_class.append(0) #false
                elif f == 0: pred_class.append(2) #same
                elif f > 0: pred_class.append(1) #true
            pred_class = np.array(pred_class)
            print('warning: ACC calculated only for binary/positive pairs')
            #print(f'ACC: {accuracy_score(target_diff[target_diff != 0.5], pred_class[target_diff != 0.5]):.3}')
            print(f'ACC: {accuracy_score(target_diff[target_diff == 1], pred_class[target_diff == 1]):.3}')

            # ### Confusion matrix
            # cm = confusion_matrix(target_diff[target_diff != 0.5], pred_class[target_diff != 0.5])
            # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1]) # Adjust display_labels for your classes
            # fig = disp.plot().figure_
            # fig.savefig(os.path.join(opt.output_fullname, 'confusion_matrix'+'.jpg'))
            # plt.close()

            # #check
            # right_list = []
            # for t,p in zip(target_diff, pred_class):
            #     if t != 0.5:
            #         if t==p: right_list.append(1)
            #         else: right_list.append(0)
            # print(sum(right_list)/len(right_list))

            ### Accuracy, target interval
            class_interval = {}
            #class_interval_case = {}
            class_interval_fromone = {}
            for t1, t2, f in zip(target1, target2, feature_diff):
                interval = (t2-t1).item()
                if interval == 0: continue
                elif interval < 0: continue

                if interval not in class_interval: 
                    class_interval[interval]=[]
                if interval not in class_interval_fromone:
                    class_interval_fromone[interval] = []
            
                if f < 0: class_interval[interval].append(0) #false
                elif f == 0: class_interval[interval].append(2) #same
                elif f > 0: class_interval[interval].append(1) #true

                if t1 == 1:
                    if f < 0: class_interval_fromone[interval].append(0) #false
                    elif f == 0: class_interval_fromone[interval].append(2) #same
                    elif f > 0: class_interval_fromone[interval].append(1) #true

                # case = str(int(t1))+'-'+str(int(t2))
                # if case not in class_interval_case:
                #     class_interval_case[case] = []
                
                # if f < 0: class_interval_case[case].append(0) #false
                # elif f == 0: class_interval_case[case].append(2) #same
                # elif f > 0: class_interval_case[case].append(1) #true

            acc_interval_dict = {}
            for interval in sorted(class_interval.keys()):
                pred_class = np.array(class_interval[interval])
                if interval < 0: target_class = np.array([0 for x in range(len(pred_class))])
                elif interval == 0: target_class = np.array([0.5 for x in range(len(pred_class))])
                elif interval > 0: target_class = np.array([1 for x in range(len(pred_class))])

                acc_interval = accuracy_score(target_class[target_class != 0.5], pred_class[target_class != 0.5])
                acc_interval_dict[interval] = [acc_interval, len(pred_class[target_class != 0.5])]

                print(f'interval {str(interval)}: \
                      acc={acc_interval:.3} \
                      , num={len(pred_class[target_class != 0.5])}')
                # print(pred_class)
                # print(target_class)
            
            acc_intervalfromone_dict = {}
            for interval in sorted(class_interval_fromone.keys()):
                pred_class = np.array(class_interval_fromone[interval])
                if interval < 0: target_class = np.array([0 for x in range(len(pred_class))])
                elif interval == 0: target_class = np.array([0.5 for x in range(len(pred_class))])
                elif interval > 0: target_class = np.array([1 for x in range(len(pred_class))])

                acc_interval = accuracy_score(target_class[target_class != 0.5], pred_class[target_class != 0.5])
                acc_intervalfromone_dict[interval] = [acc_interval, len(pred_class[target_class != 0.5])]

                print(f'interval from one {str(interval)}: \
                      acc={acc_interval:.3} \
                      , num={len(pred_class[target_class != 0.5])}')
                # print(pred_class)
                # print(target_class)
            
            # acc_intervalcase_dict = {}
            # for intervalcase in class_interval_case.keys():
            #     pred_class = np.array(class_interval_case[intervalcase])

            #     splt = intervalcase.split('-')
            #     interval = int(float(splt[1]))-int(float(splt[0]))
            #     #print(splt, interval)

            #     if interval < 0: target_class = np.array([0 for x in range(len(pred_class))])
            #     elif interval == 0: target_class = np.array([0.5 for x in range(len(pred_class))])
            #     elif interval > 0: target_class = np.array([1 for x in range(len(pred_class))])

            #     acc_interval = accuracy_score(target_class[target_class != 0.5], pred_class[target_class != 0.5])
            #     acc_intervalcase_dict[intervalcase] = acc_interval

            #     print(f'interval case {intervalcase}: \
            #           acc={acc_interval:.3} \
            #           , num={len(pred_class[target_class != 0.5])}')
            #     # print(pred_class)
            #     # print(target_class)
            
            def plot_acc_bar(acc_interval_dict, name):
                interval_pos = [x for x in acc_interval_dict.keys() if x>0] 
                interval_pos.sort()
                acc_pos = [acc_interval_dict[x][0] for x in interval_pos]
                num_list = [acc_interval_dict[x][1] for x in interval_pos]
                fig = plt.figure()
                bar = plt.bar(interval_pos, acc_pos, edgecolor='black', facecolor='grey')
                for i, rect in enumerate(bar):
                    height = rect.get_height()
                    plt.text(rect.get_x() + rect.get_width() / 2.0, height, f'{num_list[i]}', ha='center', va='bottom')
                if name == 'acc_interval': plt.xlabel('Image interval')
                elif name == 'acc_intervalfromone': plt.xlabel('Image interval from the first image')
                plt.ylabel('Accuracy')
                plt.xticks(interval_pos)
                fig.savefig(os.path.join(opt.output_fullname, name+'.jpg'))
                fig.savefig(os.path.join(opt.output_fullname, name+'.pdf'))
                plt.close()

            plot_acc_bar(acc_interval_dict, 'acc_interval')
            plot_acc_bar(acc_intervalfromone_dict, 'acc_intervalfromone')
            
            # fig = plt.figure()
            # plt.bar(acc_intervalcase_dict.keys(), acc_intervalcase_dict.values(), edgecolor='black', facecolor='grey')
            # plt.xlabel('Imaging pair')
            # plt.ylabel('Accuracy')
            # plt.xticks(rotation=45)
            # fig.savefig(os.path.join(opt.output_fullname, 'acc_intervalcase'+'.jpg'))
            # plt.close()

            ### Plot histogram of pred_positive
            feature_diff_sig = sigmoid(torch.tensor(feature_diff)).numpy()
            hist_width = 0.1
            for case, pred in {'positive':feature_diff_sig[target_diff==1], 'negative':feature_diff_sig[target_diff==0]}.items():
                fig = plt.figure()
                plt.hist(pred, edgecolor='black', facecolor='grey', \
                            bins=np.arange(start=np.floor(min(pred) / hist_width) * hist_width, 
                                        stop=np.ceil(max(pred) / hist_width) * hist_width + hist_width, step=hist_width))
                plt.xlabel('Prediction')
                plt.ylabel('Frequency')
                fig.savefig(os.path.join(opt.output_fullname, 'pred_hist_'+case+'.jpg'))
                plt.close()

        else:
            if dtm == 'loss':
                print(f'{dtm.upper()}: {opt.loss(torch.tensor(feature_diff), torch.tensor(target_diff)).item():.3f}')
            else:
                print(f'{dtm.upper()}: {dict_metric[dtm](target_diff, feature_diff):.3f}')




def test(network,opt, overwrite = False):
    # import sklearn.metrics as metrics
    # from scipy.stats import pearsonr
    # sigmoid = nn.Sigmoid()

    # def rmse(a, b):
    #     return metrics.mean_squared_error(a, b, squared=False)

    savedmodelname = f"{opt.output_fullname}/model_best.pth"

    # dict_metric = {'auc': metrics.roc_auc_score, 'pearson': pearsonr,
    #                'rmse': rmse, 'loss': opt.loss}
    # dict_task_metrics = {'o': ['loss', 'auc', 'acc'],
    #                      't': ['loss', 'rmse'],
    #                      's': ['loss', 'rmse']}

    if opt.gradcam:
        visualize_gradcam_pair(network, opt, visualization=True)
        return

    resultname = f'prediction-testset'
    run = False
    resultfilename = os.path.join(f'' + opt.output_fullname, f'{resultname}.csv')
    if os.path.exists(resultfilename):
        print(f"result exists: {resultfilename}")

        get_score(opt, resultfilename)

        # result = pd.read_csv(resultfilename)
        # target_diff = np.array(result['target'])
        # feature_diff = np.array(result['predicted'])

        # for dtm in dict_task_metrics[args.task_option]:
        #     if dtm == 'auc' and args.task_option == 'o':
        #         print(f'warning: {dtm.upper()} calculated only for binary pairs')
        #         feature_diff_auc = sigmoid(torch.tensor(feature_diff)).numpy()
        #         print(f'{dtm.upper()}: {dict_metric[dtm](target_diff[target_diff != 0.5], feature_diff_auc[target_diff != 0.5]):.3}')
        #     elif dtm == 'acc' and args.task_option == 'o':
        #         pred_class = []
        #         for f in feature_diff:
        #             if f < 0: pred_class.append(0) #false
        #             elif f == 0: pred_class.append(2) #same
        #             elif f > 0: pred_class.append(1) #true
        #         pred_class = np.array(pred_class)
        #         print('warning: ACC calculated only for binary pairs')
        #         print(f'ACC: {accuracy_score(target_diff[target_diff != 0.5], pred_class[target_diff != 0.5]):.3}')

        #         # #check
        #         # right_list = []
        #         # for t,p in zip(target_diff, pred_class):
        #         #     if t != 0.5:
        #         #         if t==p: right_list.append(1)
        #         #         else: right_list.append(0)
        #         # print(sum(right_list)/len(right_list))

        #         # Plot histogram of pred_positive
        #         feature_diff_sig = sigmoid(torch.tensor(feature_diff)).numpy()
        #         hist_width = 0.1
        #         for case, pred in {'positive':feature_diff_sig[target_diff==1], 'negative':feature_diff_sig[target_diff==0]}.items():
        #             fig = plt.figure()
        #             plt.hist(pred, edgecolor='black', facecolor='grey', \
        #                      bins=np.arange(start=np.floor(min(pred) / hist_width) * hist_width, 
        #                                     stop=np.ceil(max(pred) / hist_width) * hist_width + hist_width, step=hist_width))
        #             plt.xlabel('Prediction')
        #             plt.ylabel('Frequency')
        #             fig.savefig(os.path.join(opt.output_fullname, 'pred_hist_'+case+'.jpg'))
        #             plt.close()

            # else:
            #     if dtm == 'loss':
            #         print(f'{dtm.upper()}: {opt.loss(torch.tensor(feature_diff), torch.tensor(target_diff)).item():.3f}')
            #     else:
            #         print(f'{dtm.upper()}: {dict_metric[dtm](target_diff, feature_diff):.3f}')

    if not os.path.exists(resultfilename) or overwrite:
        run = True

    if run:
        print("RUN TEST")
        cuda = True
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

        # parallel = True
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if parallel:
        #     network = nn.DataParallel(network).to(device)
        #     if opt.pretrained_weight:
        #         print("Model is using pretrained weights from the paper")
        #         pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
        #         pretrained_dir = './model_weights'
        #         pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
        #         print(pretrained_path)
        #         assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check.\n" \
        #                                             "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
        #         network.load_state_dict(torch.load(pretrained_path))
        #     else:
        #         network.load_state_dict(torch.load(savedmodelname))

        # else:
        #     if opt.pretrained_weight:
        #         print("Model is using pretrained weights from the paper")
        #         pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
        #         pretrained_dir = './model_weights'
        #         pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
        #         assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check.\n" \
        #                                             "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
        #         state_dict = torch.load(pretrained_path)
        #         # remap to handle w/o DataParallel:  a new state_dict by removing 'module.' prefix
        #         new_state_dict = {}
        #         for key in state_dict.keys():
        #             if key.startswith("module."):
        #                 new_key = key.replace('module.', '')  # Remove 'module.' from the keys

        #             new_state_dict[new_key] = state_dict[key]

        #         # Load the updated state_dict into your model
        #         network.load_state_dict(new_state_dict)
        #     else:
        #         network.load_state_dict(torch.load(savedmodelname))

        #     if cuda:
        #         network = network.cuda()
        network = get_network(opt, network, savedmodelname)

        network.eval()

        loader_test = torch.utils.data.DataLoader(opt.test_loader,
                                                  batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)

        tmp_stack_target_o = np.empty((0, 1))
        tmp_stack_predicted1 = np.empty((0, 1))
        tmp_stack_target1_o = np.empty((0, 1))
        tmp_stack_target2_o = np.empty((0, 1))
        tmp_stack_target_psa = np.empty((0, 1))
        tmp_stack_predicted2 = np.empty((0, 1))
        tmp_stack_target1_psa = np.empty((0, 1))
        tmp_stack_target2_psa = np.empty((0, 1))

        # moved this to test subjectid key problem
        result = pd.DataFrame()
        result['subject'] = np.array(
            loader_test.dataset.demo['subject'].iloc[loader_test.dataset.index_combination[:, 0]])

        for teststep, batch in enumerate(loader_test):
            sys.stdout.write(
                "\r [Batch %d/%d] "  # [ target diff: %d ]
                % (teststep,
                   len(loader_test),
                   )
            )

            # if len(opt.optional_meta) > 0:
            #     I1, I2 = batch
            #     input1, target1, meta1 = I1
            #     input2, target2, meta2 = I2
            #     predicted = network(input2.type(Tensor), input1.type(Tensor),
            #                         meta = [meta2.type(Tensor), meta1.type(Tensor)])

            # else:
            I1, I2 = batch
            input1, target_o_1, target_psa_1  = I1
            input2, target_o_2, target_psa_2 = I2
            predicted1, predicted2 = network(input2.type(Tensor), input1.type(Tensor))

            targetdiff_o = (target_o_2 - target_o_1)[:, None].type(Tensor)
            targetdiff_psa = torch.zeros(targetdiff_o.shape).type(Tensor)
            for xx in range(targetdiff_psa.shape[0]):
                if target_psa_1[xx] == 1 and target_psa_2[xx] == 1: targetdiff_psa[xx] = 1
                elif target_psa_1[xx] == 0 and target_psa_2[xx] == 0: targetdiff_psa[xx] = 0
                else:
                    print('target_psa_1, target_psa_2: ', target_psa_1, target_psa_2)
                    exit()
            if opt.task_option == 'o':
                targetdiff_o[targetdiff_o > 0] = 1
                targetdiff_o[targetdiff_o == 0] = 0.5
                targetdiff_o[targetdiff_o < 0] = 0
                
            tmp_stack_predicted1 = np.append(tmp_stack_predicted1,
                                            np.array((predicted1).cpu().detach()),
                                            axis=0)
            tmp_stack_target_o = np.append(tmp_stack_target_o,
                                         targetdiff_o.cpu().detach(), axis=0)
            tmp_stack_target1_o = np.append(tmp_stack_target1_o, np.array(target_o_1)[:, None], axis=0)
            tmp_stack_target2_o = np.append(tmp_stack_target2_o, np.array(target_o_2)[:, None], axis=0)

            tmp_stack_predicted2 = np.append(tmp_stack_predicted2,
                                            np.array((predicted2).cpu().detach()),
                                            axis=0)
            tmp_stack_target_psa = np.append(tmp_stack_target_psa,
                                         targetdiff_psa.cpu().detach(), axis=0)
            tmp_stack_target1_psa = np.append(tmp_stack_target1_psa, np.array(target_psa_1)[:, None], axis=0)
            tmp_stack_target2_psa = np.append(tmp_stack_target2_psa, np.array(target_psa_2)[:, None], axis=0)

        result['target_o'] = tmp_stack_target_o
        result['predicted_o'] = tmp_stack_predicted1
        result['target1_o'] = tmp_stack_target1_o
        result['target2_o'] = tmp_stack_target2_o
        result['target_psa'] = tmp_stack_target_psa
        result['predicted_psa'] = tmp_stack_predicted2
        result['target1_psa'] = tmp_stack_target1_psa
        result['target2_psa'] = tmp_stack_target2_psa
        result.to_csv(resultfilename)

        # result = pd.read_csv(resultfilename)
        # target_diff = np.array(result['target'])
        # feature_diff = np.array(result['predicted'])

        print('\nSaved filename: ', resultfilename)

        get_score(opt, resultfilename)
        # for dtm in dict_task_metrics[args.task_option]:
        #     if dtm == 'auc' and args.task_option == 'o':
        #         print(f'warning: {dtm.upper()} calculated only for binary pairs')
        #         feature_diff_auc = sigmoid(torch.tensor(feature_diff)).numpy()
        #         print(f'{dtm.upper()}: {dict_metric[dtm](target_diff[target_diff != 0.5], feature_diff_auc[target_diff != 0.5]):.3}')
        #     elif dtm == 'acc' and args.task_option == 'o':
        #         pred_class = []
        #         for f in feature_diff:
        #             if f < 0: pred_class.append(0) #false
        #             elif f == 0: pred_class.append(2) #same
        #             elif f > 0: pred_class.append(1) #true
        #         pred_class = np.array(pred_class)
        #         print('warning: ACC calculated only for binary pairs')
        #         print(f'ACC: {accuracy_score(target_diff[target_diff != 0.5], pred_class[target_diff != 0.5]):.3}')

        #         # #check
        #         # right_list = []
        #         # for t,p in zip(target_diff, pred_class):
        #         #     if t != 0.5:
        #         #         if t==p: right_list.append(1)
        #         #         else: right_list.append(0)
        #         # print(sum(right_list)/len(right_list))

        #         # Plot histogram of pred_positive
        #         feature_diff_sig = sigmoid(torch.tensor(feature_diff)).numpy()
        #         hist_width = 0.1
        #         for case, pred in {'positive':feature_diff_sig[target_diff==1], 'negative':feature_diff_sig[target_diff==0]}.items():
        #             fig = plt.figure()
        #             plt.hist(pred, edgecolor='black', facecolor='grey', \
        #                      bins=np.arange(start=np.floor(min(pred) / hist_width) * hist_width, 
        #                                     stop=np.ceil(max(pred) / hist_width) * hist_width + hist_width, step=hist_width))
        #             plt.xlabel('Prediction')
        #             plt.ylabel('Frequency')
        #             fig.savefig(os.path.join(opt.output_fullname, 'pred_hist_'+case+'.jpg'))
        #             plt.close()

        #     else:
        #         if dtm == 'loss':
        #             print(f'{dtm.upper()}: {opt.loss(torch.tensor(feature_diff), torch.tensor(target_diff)).item():.3f}')
        #         else:
        #             print(f'{dtm.upper()}: {dict_metric[dtm](target_diff, feature_diff)}:.3f')

    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobname', default='lilac', type=str, help="name of job")  # , required=True)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=0, type=int)

    parser.add_argument('--earlystopping', default=10, type=int, help="early stopping criterion")
    parser.add_argument('--batchsize', default=16, type=int)
    parser.add_argument('--max_epoch', default=300, type=int, help="max epoch")
    parser.add_argument('--epoch', default=0, type=int, help="starting epoch")
    parser.add_argument('--save_epoch_num', default=1, type=int, help="validate and save every N epoch")

    parser.add_argument('--image_directory', default='./datasets', type=str)  # , required=True)
    parser.add_argument('--csv_file_train', default='./datasets/demo_oasis_train.csv', type=str,
                        help="csv file for training set")  # , required=True)
    parser.add_argument('--csv_file_val', default='./datasets/demo_oasis_val.csv', type=str,
                        help="csv file for validation set")  # , required=True)
    parser.add_argument('--csv_file_test', default='./datasets/demo_oasis_test.csv', type=str,
                        help="csv file for testing set")  # , required=True)
    parser.add_argument('--output_directory', default='./output', type=str,
                        help="directory path for saving model and outputs")  # , required=True)

    parser.add_argument('--image_size', default="128, 128, 128", type=str, help="w,h for 2D and w,h,d for 3D")
    parser.add_argument('--image_channel', default=1, type=int)
    parser.add_argument('--task_option', default='o', choices=['o', 't', 's'],
                        type=str, help="o: temporal 'o'rdering\n "
                                       "t: regression for 't'ime interval\n "
                                       "s: regression with optional meta for a 's'pecific target variable\n ")
    parser.add_argument('--targetname', default='age', type=str)
    parser.add_argument('--optional_meta', default='', type=str,
                        help='list optional meta names to be used (e.g., ["AGE", "AGE_x_SEX"]). csv files should include the meta data name')
    parser.add_argument('--backbone_name', default='cnn_3D', type=str,
                        help="implemented models: cnn_3D, cnn_2D, resnet50_2D, resnet18_2D, resnet18_3D")

    parser.add_argument('--run_mode', default='train', choices=['train', 'eval'], help="select mode")  # required=True,
    parser.add_argument('--pretrained_weight', default=False, action='store_true')

    parser.add_argument('--inter_num_ch', default=16, type=int, help='Number of output channels from CNN')
    parser.add_argument('--num_block', default=4, type=int, help='Number of CNN blocks')
    parser.add_argument('--exclude_sametarget', default=False, action='store_true', help='Exclude same target combinatinos (target=0.5) in o task')
    parser.add_argument('--lrscheduler', default=None, type=float, nargs=2, help='StepLR. Input: step_size gamma')
    parser.add_argument('--gradcam', default=False, action='store_true')
    parser.add_argument('--path_pretrained_model', default=None, type=str, help='Input pth path. Continue training from the model')

    parser.add_argument('--alpha_oloss', type=float)

    args = parser.parse_args()

    return args


def run_setup(args):
    if args.gradcam and args.run_mode=='train':
        print('args.gradcam and args.run_mode==train')
        exit()

    dict_loss = {'o': nn.BCEWithLogitsLoss(), 't': nn.MSELoss(), 's': nn.MSELoss()}
    dict_task = {'o': 'temporal_ordering', 't': 'regression', 's': 'regression'}

    args.loss = dict_loss[args.task_option]

    if args.optional_meta == '':
        path_pref = args.jobname + '-' + dict_task[args.task_option] + '-' + \
                    'backbone_' + args.backbone_name + '-lr' + str(args.lr) + '-seed' + str(args.seed) + '-batch' + str(
            args.batchsize)
    elif ',' in args.optional_meta:
        path_pref = args.jobname + '-' + dict_task[args.task_option] + '-' + 'meta' + '_'.join(
            args.optional_meta) + '-' + \
                    'backbone_' + args.backbone_name + '-lr' + str(args.lr) + '-seed' + str(args.seed) + '-batch' + str(
            args.batchsize)
    else:
        path_pref = args.jobname + '-' + dict_task[args.task_option] + '-' + 'meta' + '_' + \
            args.optional_meta + '-' + \
                    'backbone_' + args.backbone_name + '-lr' + str(args.lr) + '-seed' + str(args.seed) + '-batch' + str(
            args.batchsize)


    args.output_fullname = os.path.join(args.output_directory, path_pref)
    os.makedirs(args.output_fullname, exist_ok=True)

    # check path
    assert os.path.exists(args.image_directory), "incorrect image directory path"

    # set up seed
    set_manual_seed(args.seed)

    # set up GPU
    if torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")
        print("!! NO GPU AVAILABLE !!")

    # string to list
    image_size = [int(item) for item in args.image_size.split(',')]
    args.image_size = image_size
    if len(args.optional_meta) > 0 and ',' in args.optional_meta:
        optiona_meta_names = [item for item in args.optional_meta.split(',')]
        args.optional_meta = optiona_meta_names
    elif len(args.optional_meta) > 0 and not(',' in args.optional_meta):
        args.optional_meta = [args.optional_meta]
    else:
        args.optional_meta = []

    if len(args.image_size) == 2:
        if args.run_mode == 'train':
            args.train_loader = loader2D(args, trainvaltest='train')
            args.val_loader = loader2D(args, trainvaltest='val')
        if args.run_mode == 'eval':
            args.test_loader = loader2D(args, trainvaltest='test')
    elif len(args.image_size) == 3:
        if args.run_mode == 'train':
            args.train_loader = loader3D(args, trainvaltest='train')
            args.val_loader = loader3D(args, trainvaltest='val')
        if args.run_mode == 'eval':
            args.test_loader = loader3D(args, trainvaltest='test')
    else:
        raise NotImplementedError

    print(' ----------------- Run Setup Summary -----------------')
    print(f'JOB NAME: {args.jobname}')
    print(f'TASK: {dict_task[args.task_option] }')
    print(f'Target Attribute: {args.targetname}')
    if len(args.optional_meta)>0:
        print(f'Optional Meta: {args.optional_meta}')
    print(f'BACKBONE: {args.backbone_name}')
    print(f'RUN MODE: {args.run_mode}')
    print(f"Num of GPUs: {torch.cuda.device_count()}")

def visualize_gradcam_pair(network, opt, visualization=False):
    '''
    https://github.com/heejong-kim/learning-to-compare-longitudinal-images-3d/blob/978e7ae07acf3eb6299b50f3f6b18a5b89b7c3eb/train-aging.py#L1419
    '''
    dir_cam = os.path.join(opt.output_fullname, 'gradcam')
    # if os.path.isdir(dir_cam):
    #     print('dir_cam exists.')
    #     exit()
    if os.path.isdir(dir_cam) is False:
        os.mkdir(dir_cam)
    
    dir_arr = os.path.join(opt.output_fullname, 'gradcam_arr')
    if os.path.isdir(dir_arr) is False:
        os.mkdir(dir_arr)
    
    # if visualization: fname_cam_summary = os.path.join(dir_cam, 'gradcam-summary_visualization.txt')
    # else: fname_cam_summary = os.path.join(dir_cam, 'gradcam-summary.txt')

    import torchio as tio
    def write_imgs_iterate(img_dict, save_dir, save_name, num_rows, num_cols):
        os.makedirs(save_dir, exist_ok=True)

        # num_rows = 3
        # num_cols = 4
        f = plt.figure(figsize=(20, 16))
        plt.subplot(num_rows, num_cols, 1)
        # plt.imshow(t)
        # plt.title('ground truth')
        # plt.axis('off')

        i = 1
        for item in img_dict:
            plt.subplot(num_rows, num_cols, i)
            plt.imshow(item["image"])
            plt.title(item["title"])
            plt.axis('off')
            i += 1

        plt.savefig(os.path.join(save_dir, save_name))

        plt.clf()
        plt.close('all')
    
    def write_imgs_sb(img_dict, save_dir, save_name):
        os.makedirs(save_dir, exist_ok=True)

        for item in img_dict:
            fig = plt.figure(figsize=(10,5))
            plt.imshow(item["image"])
            plt.title(item["title"])
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, save_name+'_'+item["title"]+'.png'))
            plt.close()

    def tensor_hook(grad):
        grads['gradient']['difference'] = (grad[0].cpu().detach())
        # print(f'backward hook: {grad.size()}')

    def tensor_hook_input1(grad):
        grads['gradient']['activation1'] = (grad[0].cpu().detach())
        # print(f'backward hook1: {grad.size()}')

    def tensor_hook_input2(grad):
        grads['gradient']['activation2'] = (grad[0].cpu().detach())
        # print(f'backward hook2: {grad.size()}')

    def blend(image, heatmap, use_mask, x, y, z):
        t = np.clip(image, 0, 1)

        heatmap -= np.min(heatmap)

        if heatmap.max() != torch.tensor(0.):
            heatmap /= heatmap.max()

        ## 3d into 2ds
        image2d = np.concatenate((image[x, :, :].squeeze(), image[:, y, :].squeeze(), image[:, :, z].squeeze()), 1)
        heatmap2d = np.concatenate((heatmap[x, :, :].squeeze(), heatmap[:, y, :].squeeze(), heatmap[:, :, z].squeeze()), 1)

        def blend_image_and_heatmap(img_cv, heatmap, use_mask=False, image_weight=0.5):
            blended_img_mask = None
            image_size = img_cv.shape
            if use_mask:
                score = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
                heatmap_cv = np.uint8(score)
                blended_img_mask = np.uint8((np.repeat(score.reshape(image_size), 3, axis=2) * img_cv))

            heatmap = np.max(heatmap) - heatmap
            if np.max(heatmap) < 255.:
                heatmap *= 255

            score = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
            heatmap_cv = np.uint8(score)
            heatmap_cv = cv2.applyColorMap(heatmap_cv, cv2.COLORMAP_JET)

            blended_img = heatmap_cv * image_weight + img_cv
            blended_img = cv2.normalize(blended_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
            blended_img[blended_img < 0] = 0

            return blended_img, score, heatmap_cv, blended_img_mask, img_cv

        im, score, heatmap_cv, blended_img_mask, img_cv = blend_image_and_heatmap(image2d[:, :, None], heatmap2d, use_mask=use_mask)
        # blended_img, score, heatmap_cv, blended_img_mask, img_cv
        return minmax(image2d), im, heatmap_cv, blended_img_mask, image, score, heatmap

    def handle_image_saving(img_dict, orig_im, blended_im, blended_img_mask, label, operation, save_image=False,
                            save_mask=False):
        im_to_save = blended_im
        if save_mask:
            im_to_save = blended_img_mask

        im_to_save = np.concatenate((orig_im, im_to_save), 1)

        if save_image:
            title = label #f'label: {label}'
            img_dict.append({"image": im_to_save, "title": title})

    def blend_concat(im1, im2, hm1, hm2, x, y, z):

        t1, blended_im1, heatmap_cv, blended_img_mask1, image, score, heatmap = blend(im1.squeeze(), hm1,
                                                                                      True, x, y, z)
        t2, blended_im2, heatmap_cv, blended_img_mask2, image, score, heatmap = blend(im2.squeeze(), hm2,
                                                                                      True, x, y, z)
        orig_pair = np.repeat(np.concatenate((t1, t2))[:, :, None], 3, 2)
        blended_im_concat = np.concatenate((blended_im1, blended_im2))
        blended_img_mask_concat = np.concatenate((blended_img_mask1, blended_img_mask2))
        return orig_pair, blended_im_concat, blended_img_mask_concat

    def get_all_maps(activation, gradient):

        resize = tio.transforms.Resize(tuple(im1.shape))

        AM = resize((activation).sum(0).squeeze()[None, :])

        ## gradCAM elementwise (attention map * gradient)
        gradcam_activation_elementwise = (activation * gradient).sum(0)
        print('gradcam_activation_elementwise:' , gradcam_activation_elementwise.shape)
        gradCAMelement = gradcam_activation_elementwise
        gradCAMelement[gradCAMelement < 0] = 0
        gradCAMelement = resize(gradCAMelement.squeeze()[None,:])
        print('gradCAMelement: ', gradCAMelement.shape)

        # gradCAM avgpool (attention map * avgpool(gradient)) = original gradcam
        pooled_grads = gradient.sum(axis=tuple([1, 2, 3]))
        gradcam_activation = activation
        for i in range(len(pooled_grads)):
            gradcam_activation[i, :, :] *= pooled_grads[i]

        gradCAM = gradcam_activation.sum(0)
        gradCAM[gradCAM < 0] = 0
        gradCAM = resize(gradCAM.squeeze()[None, :])

        return AM.squeeze(), gradCAMelement.squeeze(), gradCAM.squeeze()

    import cv2

    savedmodelname = f"{opt.output_fullname}/model_best.pth"
    print('savedmodelname: ', savedmodelname)

    opt.batchsize = 1
    cuda = True
    #parallel = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    network = get_network(opt, network, savedmodelname)

    resultname = f'prediction-testset'
    result_pred = pd.read_csv(os.path.join(f'' + opt.output_fullname, f'{resultname}.csv'))

    loader_test = torch.utils.data.DataLoader(
            opt.test_loader,
            batch_size=opt.batchsize, shuffle=False, num_workers = 1)
    # loader_test = torch.utils.data.DataLoader(
    #         loader(root=opt.image_dir, trainvaltest='test', transform=False, opt=opt),
    #         batch_size=opt.batchsize, shuffle=False, num_workers = 1)
    # # from test()
    # loader_test = torch.utils.data.DataLoader(args.test_loader,
    #                                               batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)

    network.eval()
    network.zero_grad()

    # subject-wise saving
    subject_names = np.unique(result_pred.subject)
    for subject_name in subject_names:
        #if not os.path.exists(os.path.join(dir_cam, f'{subject_name}-GCEdiff.png')):

        print(f'{subject_name}')

        # data_index = np.where(np.logical_and(result_pred.subject == subject_name,
        #                                      result_pred['gt-target'] > 0))[0]
        data_index = np.where(np.logical_and(result_pred.subject == subject_name,
                                                result_pred['target'] > 0))[0]

        num_rows = int(np.ceil(np.sqrt(len(data_index))))
        if num_rows == 0:
            print(len(data_index))
            num_rows = 1
            num_cols = 1
        else:
            num_cols = int(np.ceil(len(data_index) / num_rows))


        gradcamelement_diff_dict = []

        for i in data_index:
            network.eval()
            network.zero_grad()
            batch = loader_test.dataset.__getitem__(i)

            # get matched ROI map

            grads = {'activation':[],
                        'gradient':{}}  # an empty dictionary

            if len(opt.optional_meta) > 0:
                I1, I2 = batch
                input1, target1, meta1 = I1
                input2, target2, meta2 = I2
                # predicted = network(input2.type(Tensor), input1.type(Tensor),
                #                     meta = [meta2.type(Tensor), meta1.type(Tensor)])
            else:
                I1, I2 = batch
                input1, target1 = I1
                input2, target2 = I2
                #predicted = network(input2.type(Tensor), input1.type(Tensor))

            #############################################################
            if target1==target2: continue
            
            input1 = torch.tensor(input1)[None, :]
            input2 = torch.tensor(input2)[None, :]
            # print('input1: ', input1.shape) #torch.Size([1, 1, 80, 80, 80])
            # print('input2: ', input2.shape) #torch.Size([1, 1, 80, 80, 80])
            
            if opt.backbone_name == "resnet18_3D":
                feature_tensor1 = network.module.backbone.conv1(input1.type(Tensor).to(device))
                print('conv1: ', feature_tensor1.shape)
                feature_tensor1 = network.module.backbone.bn1(feature_tensor1)
                print('bn1: ', feature_tensor1.shape)
                feature_tensor1 = network.module.backbone.relu(feature_tensor1)
                print('relu: ', feature_tensor1.shape)

                feature_tensor1 = network.module.backbone.maxpool(feature_tensor1)
                print('maxpool: ', feature_tensor1.shape)

                feature_tensor1 = network.module.backbone.layer1(feature_tensor1)
                print('layer1: ', feature_tensor1.shape)

                feature_tensor1 = network.module.backbone.layer2(feature_tensor1)
                print('layer2: ', feature_tensor1.shape)

                feature_tensor1 = network.module.backbone.layer3(feature_tensor1)
                print('layer3: ', feature_tensor1.shape)

                feature_tensor1 = network.module.backbone.layer4(feature_tensor1)
                print('layer4: ', feature_tensor1.shape)

                feature_tensor2 = network.module.backbone.conv1(input2.type(Tensor).to(device))
                feature_tensor2 = network.module.backbone.bn1(feature_tensor2)
                feature_tensor2 = network.module.backbone.relu(feature_tensor2)
                feature_tensor2 = network.module.backbone.maxpool(feature_tensor2)
                feature_tensor2 = network.module.backbone.layer1(feature_tensor2)
                feature_tensor2 = network.module.backbone.layer2(feature_tensor2)
                feature_tensor2 = network.module.backbone.layer3(feature_tensor2)
                feature_tensor2 = network.module.backbone.layer4(feature_tensor2)
            else:
                feature_tensor1 = network.module.backbone.encoder(input1.type(Tensor).to(device))
                feature_tensor2 = network.module.backbone.encoder(input2.type(Tensor).to(device))
            #feature_tensor = feature_tensor2 + feature_tensor1
            feature_tensor = feature_tensor2 - feature_tensor1
            print('feature_tensor: ', feature_tensor.shape) #torch.Size([1, 16, 5, 5, 5])

            activation_diff = feature_tensor.cpu().detach().squeeze().numpy()
            activation1 = feature_tensor1.cpu().detach().squeeze().numpy()
            activation2 = feature_tensor2.cpu().detach().squeeze().numpy()
            print("activation_diff: ", activation_diff.shape) #(16, 5, 5, 5)
            print("activation1: ", activation1.shape)

            handle_tensor = feature_tensor.register_hook(tensor_hook)
            handle_tensor1 = feature_tensor1.register_hook(tensor_hook_input1)
            handle_tensor2 = feature_tensor2.register_hook(tensor_hook_input2)

            feature_tensor = torch.flatten(feature_tensor, 1)
            predicted = network.module.linear(feature_tensor) 

            predicted.backward()
            handle_tensor.remove()
            handle_tensor1.remove()
            handle_tensor2.remove()

            gradient_diff = grads['gradient']['difference'].squeeze().numpy()
            gradient1 = grads['gradient']['activation1'].squeeze().numpy()
            gradient2 = grads['gradient']['activation2'].squeeze().numpy()
            print("gradient_diff: ", gradient_diff.shape) 
            print("gradient1: ", gradient1.shape)

            im1 = (input1.squeeze().numpy()) # 3d
            im2 = (input2.squeeze().numpy())
            AM_diff, gradCAMelement_diff, gradCAM_diff = get_all_maps(activation_diff, gradient_diff)
            AM1, gradCAMelement1, gradCAM1 = get_all_maps(activation1, gradient1)
            AM2, gradCAMelement2, gradCAM2 = get_all_maps(activation2, gradient2)
            #print('gradCAMelement_diff: ', gradCAMelement_diff.shape) #(80, 80, 80)

            label = f'{subject_name}_t1_{int(target1)}_t2_{int(target2)}_pred_{predicted.cpu().detach().item():.2f}'
            np.save(os.path.join(dir_arr, label+'_elementdiff'+'.npy'), gradCAMelement_diff)
            np.save(os.path.join(dir_arr, label+'_elementf1'+'.npy'), gradCAMelement1)
            np.save(os.path.join(dir_arr, label+'_elementf2'+'.npy'), gradCAMelement2)

            
            if visualization:

                # TODO:
                def find_xyz(heatmap):
                    x,y,z = np.where(heatmap == heatmap.max())
                    return x,y,z

                x, y, z = find_xyz(gradCAMelement_diff)
                if len(x)>1:
                    x, y, z = x[-1], y[-1], z[-1]

                # AM_diff_xyz = find_xyz(AM_diff)
                # gradCAMelement_diff_xyz = find_xyz(gradCAMelement_diff)
                # gradCAMelement_xyz = find_xyz(gradCAMelement1+gradCAMelement2)
                # gradCAM_diff_xyz =find_xyz(gradCAM_diff)
                # gradCAM_xyz = find_xyz(gradCAM1+gradCAM2)


                # orig_pair, blended_im_concat, blended_img_mask_concat = blend_concat(im1, im2, AM_diff, AM_diff)
                # handle_image_saving(activation_diff_dict, orig_pair, blended_im_concat, blended_img_mask_concat, label, 'AMdiff', save_image=True, save_mask=False)
                # orig_pair, blended_im_concat, blended_img_mask_concat = blend_concat(im1, im2, AM1, AM2)
                # handle_image_saving(activation_separate_dict, orig_pair, blended_im_concat, blended_img_mask_concat, label, 'AMseparate', save_image=True, save_mask=False)

                orig_pair, blended_im_concat, blended_img_mask_concat = blend_concat(im1, im2, gradCAMelement_diff, gradCAMelement_diff, x, y, z)
                handle_image_saving(gradcamelement_diff_dict, orig_pair, blended_im_concat, blended_img_mask_concat, label, 'GCEdiff', save_image=True, save_mask=False)
                # orig_pair, blended_im_concat, blended_img_mask_concat = blend_concat(im1, im2, gradCAMelement1, gradCAMelement2)
                # handle_image_saving(gradcamelement_separate_dict, orig_pair, blended_im_concat, blended_img_mask_concat, label, 'GCEseparate', save_image=True, save_mask=False)
                #
                # orig_pair, blended_im_concat, blended_img_mask_concat = blend_concat(im1, im2, gradCAM_diff, gradCAM_diff)
                # handle_image_saving(gradcam_diff_dict, orig_pair, blended_im_concat, blended_img_mask_concat, label, 'GCdiff', save_image=True, save_mask=False)
                # orig_pair, blended_im_concat, blended_img_mask_concat = blend_concat(im1, im2, gradCAM1, gradCAM2)
                # handle_image_saving(gradcam_separate_dict, orig_pair, blended_im_concat, blended_img_mask_concat, label, 'GCseparate', save_image=True, save_mask=False)

            torch.cuda.empty_cache()

        if visualization:
            # write_imgs_iterate(activation_diff_dict, f'{dir_cam}', f'{subject_name}-AMdiff', num_rows, num_cols)
            # write_imgs_iterate(activation_separate_dict, f'{dir_cam}', f'{subject_name}-AMseparate', num_rows, num_cols)
            write_imgs_iterate(gradcamelement_diff_dict, f'{dir_cam}', f'{subject_name}-GCEdiff', num_rows, num_cols)
            # write_imgs_iterate(gradcamelement_separate_dict, f'{dir_cam}', f'{subject_name}-GCEseparate', num_rows, num_cols)
            # write_imgs_iterate(gradcam_diff_dict, f'{dir_cam}', f'{subject_name}-GCdiff', num_rows, num_cols)
            # write_imgs_iterate(gradcam_separate_dict, f'{dir_cam}', f'{subject_name}-GCseparate', num_rows, num_cols)

            write_imgs_sb(gradcamelement_diff_dict, f'{dir_cam}', f'{subject_name}-GCEdiff')
            

if __name__ == "__main__":

    args = parse_args()
    run_setup(args)

    #model = LILAC(args)
    model = LILAC_PSAo(args)
    print("Num of Model Parameter:", count_parameters(model))

    if args.run_mode == 'eval':
        print(' ----------------- Testing initiated -----------------')
        test(model, args)

    else:
        assert args.run_mode == 'train', "check run_mode"
        print(' ----------------- Training initiated -----------------')
        train(model, args)

