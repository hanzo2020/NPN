# coding=utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import logging
import torchvision.models as models
import argparse
from runner.BaseRunner import BaseRunner
from runner.OtherRunner import OtherRunner
from runner.MultiRunner import MultiRunner
from runner.NPNMultiRunner import NPNMultiRunner
import torch.nn.functional as F
from Flatten import Flatten
from NPN import NPN
from LeNet5 import LeNet5
from AlexNet import AlexNet
from VGG16 import VGG16
from VGG13 import VGG13
from VGG10 import VGG10
from NPN224 import NPN224
from NPNCCS import NPNCCS
from ResNet18 import ResNet18
from dataset.dataset_img import DatasetImg
from dataset.dataset_multi_img import DatasetMImg
# print('ok')

# flag = torch.cuda.is_available()
# if flag:
#     print('可用')
#     print(torch.cuda.get_device_name(0))
# else:
#     print('不可')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size to infer with")
    parser.add_argument("--class_num", type=int, default=2,
                        help="Batch size to infer with")
    parser.add_argument("--img_size", type=int, default=28,
                        help="Batch size to infer with")
    parser.add_argument("--dataset", choices=["shape", "nine-circles", "OBC", "ChineseStyle", "CCS", "three"],
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
    if args.class_num == 2:
        dataset_train = DatasetImg(
            args.dataset, 'train', args.model_name, img_size=args.img_size
        )
        dataset_val = DatasetImg(
            args.dataset, 'val', args.model_name, img_size=args.img_size
        )
        dataset_test = DatasetImg(
            args.dataset, 'test', args.model_name, img_size=args.img_size
        )
    else:
        dataset_train = DatasetMImg(
            args.dataset, 'train', args.model_name, img_size=args.img_size, class_num=args.class_num
        )
        dataset_val = DatasetMImg(
            args.dataset, 'val', args.model_name, img_size=args.img_size, class_num=args.class_num
        )
        dataset_test = DatasetMImg(
            args.dataset, 'test', args.model_name, img_size=args.img_size, class_num=args.class_num
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
    if args.model_name == 'NPN' or args.model_name == 'NPN224':
        run = BaseRunner()
    elif args.model_name == 'NPNCCS':
        run = NPNMultiRunner(args.batch_size)
    elif args.class_num > 2:
        run = MultiRunner(args.batch_size)
    else:
        run = OtherRunner()
    run.train(net, data_loader=train_loader, device=device, val_loader=val_loader, test_loader=test_loader)

if __name__ == "__main__":
    main()