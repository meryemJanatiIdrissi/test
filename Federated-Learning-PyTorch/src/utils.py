#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset == 'mnist' or 'fmnist':
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_bn_statistics1(rm_dict, rv_dict):
    global_avg_rm, global_avg_rv = [], []

    if rm_dict and rv_dict:
        L_rm, L_rv = [], [] 
        for key1, key2 in zip(rm_dict, rv_dict):
            if(len(rm_dict[key1]) > 1):
                l = []
                rmList = list(zip(*rm_dict[key1]))
                for i in range(len(rmList)):
                    l.append(torch.mean(torch.stack(rmList[i]), dim=0)) 
                L_rm.append(l)
            else:
                L_rm.append(rm_dict[key1][0])

            if(len(rv_dict[key2]) > 1):
                l = []
                rvList = list(zip(*rv_dict[key1]))
                for i in range(len(rvList)):
                    l.append(torch.mean(torch.stack(rvList[i]), dim=0)) 
                L_rv.append(l)
            else:
                L_rv.append(rv_dict[key2][0])
            
            
        final_rm, final_rv = list(zip(*L_rm)), list(zip(*L_rv))

        for i in range(len(final_rm)):
            global_avg_rm.append(torch.mean(torch.stack(final_rm[i]), dim=0))
            global_avg_rv.append(torch.mean(torch.stack(final_rv[i]), dim=0))

        return global_avg_rm, global_avg_rv

            

    



prev_rm, prev_rv = [], []
def average_bn_statistics(local_bn_rm, local_bn_rv):
    """
    return the average of the BN statistics
    """
    global prev_rm, prev_rv
    
        
    average_rm, average_rv, global_average_rm, global_average_rv = [], [], [], []
    rmList = list(zip(*local_bn_rm))
    rvList = list(zip(*local_bn_rv))
    for i in range(len(rmList)):
        average_rm.append(torch.mean(torch.stack(rmList[i]), dim=0))
        average_rv.append(torch.mean(torch.stack(rvList[i]), dim=0))
    '''
    f = open("/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics/global", "a")
    f.write("average_statistics")
    f.write(str(average_statistics))
    f.close()
    '''

    if len(prev_rm) != 0:
        combined_rm = [prev_rm, average_rm]
        combined_rv = [prev_rv, average_rv]
        rmList = list(zip(*combined_rm))
        rvList = list(zip(*combined_rv))
        for i in range(len(rmList)):
            global_average_rm.append(torch.mean(torch.stack(rmList[i]), dim=0))
            global_average_rv.append(torch.mean(torch.stack(rvList[i]), dim=0))
        f = open("/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics/global", "a")
        f.write("global_average_statistics")
        f.write(str(global_average_rm))
        f.close()
        prev_rm = copy.deepcopy(average_rm)
        prev_rv = copy.deepcopy(average_rv)

        return global_average_rm, global_average_rv

    else:
        prev_rm = copy.deepcopy(average_rm)
        prev_rv = copy.deepcopy(average_rv)
        return average_rm, average_rv  




'''
old_statistics = []
def average_bn_statistics(global_statisctics):
    """
    return the average of the BN statistics
    """
    global old_statistics
    
        
    average_statistics, global_average_statistics = [], []
    vlist = list(zip(*global_statisctics))
    for i in range(len(vlist)):
        average_statistics.append(torch.mean(torch.stack(vlist[i]), dim=0))
    
    f = open("/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics/global", "a")
    f.write("average_statistics")
    f.write(str(average_statistics))
    f.close()
    

    if len(old_statistics) != 0:
        combined = [old_statistics, average_statistics]
        vlist = list(zip(*combined))
        for i in range(len(vlist)):
            global_average_statistics.append(torch.mean(torch.stack(vlist[i]), dim=0))
        f = open("/home/janati/Desktop/github/FL-achwin/Federated-Learning-PyTorch/src/batchStatistics/global", "a")
        f.write("global_average_statistics")
        f.write(str(global_average_statistics))
        f.close()
        old_statistics = copy.deepcopy(average_statistics)

        return global_average_statistics

    else:
        old_statistics = copy.deepcopy(average_statistics)
        return average_statistics        
'''    

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
