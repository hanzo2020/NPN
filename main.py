# coding=utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import logging
import argparse
from runner.BaseRunner import BaseRunner
from runner.OtherRunner import OtherRunner
import torch.nn.functional as F
from Flatten import Flatten
from NPN import NPN
from LeNet5 import LeNet5
from dataset.dataset_img import DatasetImg

# print('ok')

# flag = torch.cuda.is_available()
# if flag:
#     print('可用')
#     print(torch.cuda.get_device_name(0))
# else:
#     print('不可')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--dataset", choices=["shape", "nine-circles"],
                        help="Use kandinsky patterns dataset")
    parser.add_argument('--model_name', type=str, default='NPN',
                             help='Choose model to run.')
    parser.add_argument('--device', default='cpu',
                        help='cuda device, i.e. 0 or cpu')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Run on CPU instead of GPU (not recommended)")
    parser.add_argument("--random_seed", type=int, default=1,
                        help="Batch size to infer with")
    args = parser.parse_args()
    return args

def get_data_loader(args, shuffle=True):
    dataset_train = DatasetImg(
        args.dataset, 'train', args.model_name
    )
    dataset_val = DatasetImg(
        args.dataset, 'val', args.model_name
    )
    dataset_test = DatasetImg(
        args.dataset, 'test', args.model_name
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        shuffle=shuffle,
        batch_size=args.batch_size,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        shuffle=False,
        batch_size=args.batch_size,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        shuffle=False,
        batch_size=args.batch_size,
    )
    return train_loader, val_loader, test_loader

def main():
    args = get_args()
    print('args ', args)
    if args.no_cuda:
        device = torch.device('cpu')
    elif len(args.device.split(',')) > 1:
        # multi gpu
        device = torch.device('cuda')
    else:
        device = torch.device('cuda:' + args.device)
    if args.random_seed != 1:
        torch.manual_seed(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)
    model_name = eval(args.model_name)
    print('device:', device)
    train_path = 'dataset/train'
    print(train_path)

    train_loader, val_loader, test_loader = get_data_loader(args)
    net = model_name(device).to(device)
    # train(net, epoch=10, args=args, data_loader=train_loader, device=args.device)
    if args.model_name == 'NPN':
        run = BaseRunner()
    else:
        run = OtherRunner()
    run.train(net, data_loader=train_loader, device=device, val_loader=val_loader, test_loader=test_loader)

if __name__ == "__main__":
    main()