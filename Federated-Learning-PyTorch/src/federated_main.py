#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details, average_bn_statistics, average_bn_statistics1
from collections import defaultdict


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(args.gpu)
    #device = 'cuda' if args.gpu else 'cpu'
    device = 'cuda'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar()

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    #rm_dict, rv_dict = {}, {}
    rm_dict, rv_dict = defaultdict(list), defaultdict(list) 
    
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_accuracies, local_bn_rm, local_bn_rv = [], [], [], [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        #idxs_users = [76, 54, 80, 33, 85, 84,  5, 73, 12, 91]

        num_users = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
        idxs_users = np.random.choice(range(30), m, replace=False)

        for idx in idxs_users:
            print("======================================================== client id : ", idxs_users)
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w, loss, accuracy, list_rm, list_rv = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch)
            #print("---------------------------", loss)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            local_accuracies.append(copy.deepcopy(accuracy))
            
            '''
            print(list_rm.count(None))
            if(list_rm.count(None) == 0):
                local_BN_Statistics.append(list_rm)
            
            rm_dict[idx] = list_rm
            rv_dict[idx] = list_rv
            '''
            rm_dict[idx].append(list_rm)
            rv_dict[idx].append(list_rv)            

            local_bn_rm.append(list_rm)
            local_bn_rv.append(list_rv)
        

        #update BN statistics
        #average_rm, average_rv = average_bn_statistics(local_bn_rm, local_bn_rv)
        average_rm, average_rv = average_bn_statistics1(rm_dict, rv_dict)
        #print("-------------------------------------------------", global_BN_Statistics)
        '''
        f = open("/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics/global", "a")
        f.write("TRAIN")
        f.write(str(global_BN_Statistics))
        f.close()
        '''
        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)


        loss_avg = sum(local_losses) / len(local_losses)
        accuracy_avg = sum(local_accuracies)/len(local_accuracies)
        train_loss.append(loss_avg)
        train_accuracy.append(accuracy_avg)

        '''
        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))
        '''
        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))



        #print("DICTIONARY     :       ", rm_dict)
        # I moved this section inside
        # Test inference after completion of training
        test_acc, test_l = test_inference(args, model=global_model, test_dataset=test_dataset, avg_rm=average_rm, avg_rv=average_rv)
        test_accuracy.append(test_acc)
        test_loss.append(test_l)
        print(f' \n Results after {args.epochs} global rounds of training:')
        #print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        print("|---- Avg Train Accuracy: {:.2f}%".format(100*np.mean(np.array(train_accuracy))))
        print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    

    file_directory = "/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/save/objects"
    file_name = os.path.join(file_directory, '{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs))

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    
    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))
    
    # Plot test Accuracy and loss vs Communication rounds
    plt.figure()
    plt.title('Test Accuracy vs Communication rounds')
    plt.plot(range(len(test_accuracy)), test_accuracy, color='k')
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Communication Rounds')
    plt.savefig('/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_test_acc.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))

    plt.figure()
    plt.title('Test Loss vs Communication rounds')
    plt.plot(range(len(test_loss)), test_loss, color='r')
    plt.ylabel('Testing Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_test_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.iid, args.local_ep, args.local_bs))