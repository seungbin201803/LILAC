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
from sklearn.metrics import accuracy_score, roc_auc_score, root_mean_squared_error, mean_squared_error
import cv2

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler
import torchio as tio


def minmax(cam):
    cam_min = np.min(cam)
    cam = cam - cam_min
    cam_max = np.max(cam)
    cam = cam / (1e-7 + cam_max)
    return cam



def train(network, opt):
    cuda = True
    parallel = True
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    os.makedirs(f"{opt.output_fullname}/", exist_ok=True)
    if parallel:
        network = nn.DataParallel(network).to(device)
        if opt.pretrained_weight:
            print("Model is using pretrained weights from the paper")
            pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
            pretrained_dir = './model_weights'
            pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
            assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check. \n" \
                                                    "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
            network.load_state_dict(torch.load(pretrained_path))
    else:
        if opt.pretrained_weight:
            print("Model is using pretrained weights from the paper")
            pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
            pretrained_dir = './model_weights'
            pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
            assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check.\n" \
                                                    "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
            state_dict = torch.load(pretrained_path)
            # remap to handle w/o DataParallel:  a new state_dict by removing 'module.' prefix
            new_state_dict = {}
            for key in state_dict.keys():
                if key.startswith("module."):
                    new_key = key.replace('module.', '')  # Remove 'module.' from the keys
                new_state_dict[new_key] = state_dict[key]
        network = network.cuda()

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

        for step, batch in enumerate(loader_train):
            step_start_time = time.time()

            if len(args.optional_meta) > 0:
                I1, I2 = batch
                input1, target1, meta1 = I1
                input2, target2, meta2 = I2
                predicted = network(input2.type(Tensor), input1.type(Tensor),
                                    meta = [meta2.type(Tensor), meta1.type(Tensor)])

            else:
                I1, I2 = batch
                input1, target1 = I1
                input2, target2 = I2
                predicted = network(input2.type(Tensor), input1.type(Tensor))

            targetdiff = (target2 - target1)[:, None].type(Tensor)
            if opt.task_option == 'o':
                targetdiff[targetdiff > 0] = 1
                targetdiff[targetdiff == 0] = 0.5
                targetdiff[targetdiff < 0] = 0

            # Loss
            optimizer.zero_grad()
            loss = args.loss(predicted, targetdiff)
            loss.backward()
            optimizer.step()
            epoch_total_loss.append(loss.item())

            # Log Progress
            batches_done = epoch * len(loader_train) + step
            batches_left = opt.max_epoch * len(loader_train) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [ loss: %f ] ETA: %s"
                % (
                    epoch,
                    opt.max_epoch,
                    step,
                    len(loader_train),
                    loss.item(),
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

            network.eval()
            valloss_total = []
            for valstep, batch in enumerate(loader_val):
                if len(args.optional_meta) > 0:
                    I1, I2 = batch
                    input1, target1, meta1 = I1
                    input2, target2, meta2 = I2
                    predicted = network(input2.type(Tensor), input1.type(Tensor),
                                        meta = [meta2.type(Tensor), meta1.type(Tensor)])

                else:
                    I1, I2 = batch
                    input1, target1 = I1
                    input2, target2 = I2
                    predicted = network(input2.type(Tensor), input1.type(Tensor))

                targetdiff = (target2 - target1)[:, None].type(Tensor)
                if opt.task_option == 'o':
                    targetdiff[targetdiff > 0] = 1
                    targetdiff[targetdiff == 0] = 0.5
                    targetdiff[targetdiff < 0] = 0

                valloss = args.loss(predicted, targetdiff)
                valloss_total.append(valloss.item())

            log_stats([np.mean(valloss_total)], ['loss/val'], epoch, writer)
            val_loss_info = 'val loss: %.4e' % (np.mean(valloss_total))
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
    sigmoid = nn.Sigmoid()
    dict_metric = {'auc': roc_auc_score, 
                   'rmse': root_mean_squared_error, 
                   'loss': opt.loss, 
                   'mse':mean_squared_error}
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
        elif dtm == 'acc' and args.task_option == 'o':
            pred_class = []
            for f in feature_diff:
                if f < 0: pred_class.append(0) #false
                elif f == 0: pred_class.append(2) #same
                elif f > 0: pred_class.append(1) #true
            pred_class = np.array(pred_class)
            print('warning: ACC calculated only for binary/positive pairs')
            print(f'ACC: {accuracy_score(target_diff[target_diff == 1], pred_class[target_diff == 1]):.3}')
            #print(f'ACC: {accuracy_score(target_diff[target_diff != 0.5], pred_class[target_diff != 0.5]):.3}')

            ### Accuracy, different interval
            class_interval = {}
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
    savedmodelname = f"{opt.output_fullname}/model_best.pth"

    if opt.gradcam:
        visualize_gradcam_pair(network, opt, visualization=True)
        return

    resultname = f'prediction-testset'
    run = False
    resultfilename = os.path.join(f'' + opt.output_fullname, f'{resultname}.csv')
    if os.path.exists(resultfilename):
        print(f"result exists: {resultfilename}")
        get_score(opt, resultfilename)

    if not os.path.exists(resultfilename) or overwrite:
        run = True

    if run:
        print("RUN TEST")
        cuda = True
        parallel = True
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

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
                network.load_state_dict(torch.load(savedmodelname))

        else:
            if opt.pretrained_weight:
                print("Model is using pretrained weights from the paper")
                pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
                pretrained_dir = './model_weights'
                pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
                assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check.\n" \
                                                    "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
                state_dict = torch.load(pretrained_path)
                # remap to handle w/o DataParallel:  a new state_dict by removing 'module.' prefix
                new_state_dict = {}
                for key in state_dict.keys():
                    if key.startswith("module."):
                        new_key = key.replace('module.', '')  # Remove 'module.' from the keys

                    new_state_dict[new_key] = state_dict[key]

                # Load the updated state_dict into your model
                network.load_state_dict(new_state_dict)
            else:
                network.load_state_dict(torch.load(savedmodelname))

            if cuda:
                network = network.cuda()


        network.eval()

        loader_test = torch.utils.data.DataLoader(opt.test_loader,
                                                  batch_size=opt.batchsize, shuffle=False, num_workers=opt.num_workers)

        tmp_stack_target = np.empty((0, 1))
        tmp_stack_predicted = np.empty((0, 1))
        tmp_stack_target1 = np.empty((0, 1))
        tmp_stack_target2 = np.empty((0, 1))

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

            if len(opt.optional_meta) > 0:
                I1, I2 = batch
                input1, target1, meta1 = I1
                input2, target2, meta2 = I2
                predicted = network(input2.type(Tensor), input1.type(Tensor),
                                    meta = [meta2.type(Tensor), meta1.type(Tensor)])

            else:
                I1, I2 = batch
                input1, target1 = I1
                input2, target2 = I2
                predicted = network(input2.type(Tensor), input1.type(Tensor))

            targetdiff = (target2 - target1)[:, None].type(Tensor)
            if opt.task_option == 'o':
                targetdiff[targetdiff > 0] = 1
                targetdiff[targetdiff == 0] = 0.5
                targetdiff[targetdiff < 0] = 0

            tmp_stack_predicted = np.append(tmp_stack_predicted,
                                            np.array((predicted).cpu().detach()),
                                            axis=0)
            tmp_stack_target = np.append(tmp_stack_target,
                                         targetdiff.cpu().detach(), axis=0)
            tmp_stack_target1 = np.append(tmp_stack_target1, np.array(target1)[:, None], axis=0)
            tmp_stack_target2 = np.append(tmp_stack_target2, np.array(target2)[:, None], axis=0)

        result['target'] = tmp_stack_target
        result['predicted'] = tmp_stack_predicted
        result['target1'] = tmp_stack_target1
        result['target2'] = tmp_stack_target2
        result.to_csv(resultfilename)

        print('\nSaved filename: ', resultfilename)
        get_score(opt, resultfilename)

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

    parser.add_argument('--lrscheduler', default=None, type=float, nargs=2, help='StepLR. Input: step_size gamma')
    parser.add_argument('--gradcam', default=False, action='store_true')
    parser.add_argument('--path_pretrained_model', default=None, type=str, help='Input pth path. Continue training from the model')


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
    dir_cam = os.path.join(opt.output_fullname, 'gradcam')
    # if os.path.isdir(dir_cam):
    #     print('dir_cam exists.')
    #     exit()
    if os.path.isdir(dir_cam) is False:
        os.mkdir(dir_cam)
    
    dir_arr = os.path.join(opt.output_fullname, 'gradcam_arr')
    if os.path.isdir(dir_arr) is False:
        os.mkdir(dir_arr)
   
    def write_imgs_iterate(img_dict, save_dir, save_name, num_rows, num_cols):
        os.makedirs(save_dir, exist_ok=True)

        f = plt.figure(figsize=(20, 16))
        plt.subplot(num_rows, num_cols, 1)
       
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
        gradCAMelement = gradcam_activation_elementwise
        gradCAMelement[gradCAMelement < 0] = 0
        gradCAMelement = resize(gradCAMelement.squeeze()[None,:])

        # gradCAM avgpool (attention map * avgpool(gradient)) = original gradcam
        pooled_grads = gradient.sum(axis=tuple([1, 2, 3]))
        gradcam_activation = activation
        for i in range(len(pooled_grads)):
            gradcam_activation[i, :, :] *= pooled_grads[i]

        gradCAM = gradcam_activation.sum(0)
        gradCAM[gradCAM < 0] = 0
        gradCAM = resize(gradCAM.squeeze()[None, :])

        return AM.squeeze(), gradCAMelement.squeeze(), gradCAM.squeeze()


    savedmodelname = f"{opt.output_fullname}/model_best.pth"
    print('savedmodelname: ', savedmodelname)

    opt.batchsize = 1
    cuda = True
    parallel = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    if parallel:
        network = nn.DataParallel(network).to(device)
        if opt.pretrained_weight:
            print("Model is using pretrained weights from the paper")
            pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
            pretrained_dir = './model_weights'
            pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
            assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check.\n" \
                                                "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
            network.load_state_dict(torch.load(pretrained_path))
        else:
            if savedmodelname is not None:
                network.load_state_dict(torch.load(savedmodelname))
    else:
        if opt.pretrained_weight:
            print("Model is using pretrained weights from the paper")
            pretrained_filename = opt.output_fullname.split('/')[-1] + '.pth'
            pretrained_dir = './model_weights'
            pretrained_path = os.path.join(pretrained_dir, pretrained_filename)
            assert os.path.exists(pretrained_path), "Pretrained weight does not exist. Please check.\n" \
                                                "Download: wget https://zenodo.org/records/14713287/files/lilac_model_weights.tar.gz"
            state_dict = torch.load(pretrained_path)
            # remap to handle w/o DataParallel:  a new state_dict by removing 'module.' prefix
            new_state_dict = {}
            for key in state_dict.keys():
                if key.startswith("module."):
                    new_key = key.replace('module.', '')  # Remove 'module.' from the keys

                new_state_dict[new_key] = state_dict[key]

            # Load the updated state_dict into your model
            network.load_state_dict(new_state_dict)
        else:
            network.load_state_dict(torch.load(savedmodelname))

        if cuda:
            network = network.cuda()

    resultname = f'prediction-testset'
    result_pred = pd.read_csv(os.path.join(f'' + opt.output_fullname, f'{resultname}.csv'))

    loader_test = torch.utils.data.DataLoader(
            opt.test_loader,
            batch_size=opt.batchsize, shuffle=False, num_workers = 1)

    network.eval()
    network.zero_grad()

    # subject-wise saving
    subject_names = np.unique(result_pred.subject)
    for subject_name in subject_names:
        data_index = np.where(np.logical_and(result_pred.subject == subject_name,
                                                result_pred['target'] > 0))[0]
        num_rows = int(np.ceil(np.sqrt(len(data_index))))
        if num_rows == 0:
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
            else:
                I1, I2 = batch
                input1, target1 = I1
                input2, target2 = I2

            #############################################################
            if target1==target2: continue
            
            input1 = torch.tensor(input1)[None, :]
            input2 = torch.tensor(input2)[None, :]
            
            if opt.backbone_name == "resnet18_3D":
                feature_tensor1 = network.module.backbone.conv1(input1.type(Tensor).to(device))
                feature_tensor1 = network.module.backbone.bn1(feature_tensor1)
                feature_tensor1 = network.module.backbone.relu(feature_tensor1)
                feature_tensor1 = network.module.backbone.maxpool(feature_tensor1)
                feature_tensor1 = network.module.backbone.layer1(feature_tensor1)
                feature_tensor1 = network.module.backbone.layer2(feature_tensor1)
                feature_tensor1 = network.module.backbone.layer3(feature_tensor1)
                feature_tensor1 = network.module.backbone.layer4(feature_tensor1)

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

            activation_diff = feature_tensor.cpu().detach().squeeze().numpy()
            activation1 = feature_tensor1.cpu().detach().squeeze().numpy()
            activation2 = feature_tensor2.cpu().detach().squeeze().numpy()

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

            im1 = (input1.squeeze().numpy()) # 3d
            im2 = (input2.squeeze().numpy())
            AM_diff, gradCAMelement_diff, gradCAM_diff = get_all_maps(activation_diff, gradient_diff)
            AM1, gradCAMelement1, gradCAM1 = get_all_maps(activation1, gradient1)
            AM2, gradCAMelement2, gradCAM2 = get_all_maps(activation2, gradient2)

            label = f'{subject_name}_t1_{int(target1)}_t2_{int(target2)}_pred_{predicted.cpu().detach().item():.2f}'
            np.save(os.path.join(dir_arr, label+'_elementdiff'+'.npy'), gradCAMelement_diff)
            np.save(os.path.join(dir_arr, label+'_elementf1'+'.npy'), gradCAMelement1)
            np.save(os.path.join(dir_arr, label+'_elementf2'+'.npy'), gradCAMelement2)

            if visualization:
                def find_xyz(heatmap):
                    x,y,z = np.where(heatmap == heatmap.max())
                    return x,y,z

                x, y, z = find_xyz(gradCAMelement_diff)
                if len(x)>1:
                    x, y, z = x[-1], y[-1], z[-1]

                orig_pair, blended_im_concat, blended_img_mask_concat = blend_concat(im1, im2, gradCAMelement_diff, gradCAMelement_diff, x, y, z)
                handle_image_saving(gradcamelement_diff_dict, orig_pair, blended_im_concat, blended_img_mask_concat, label, 'GCEdiff', save_image=True, save_mask=False)

            torch.cuda.empty_cache()

        if visualization:
            write_imgs_iterate(gradcamelement_diff_dict, f'{dir_cam}', f'{subject_name}-GCEdiff', num_rows, num_cols)
            write_imgs_sb(gradcamelement_diff_dict, f'{dir_cam}', f'{subject_name}-GCEdiff')
            

if __name__ == "__main__":

    args = parse_args()
    run_setup(args)

    model = LILAC(args)
    print("Num of Model Parameter:", count_parameters(model))

    if args.run_mode == 'eval':
        print(' ----------------- Testing initiated -----------------')
        test(model, args)

    else:
        assert args.run_mode == 'train', "check run_mode"
        print(' ----------------- Training initiated -----------------')
        train(model, args)
