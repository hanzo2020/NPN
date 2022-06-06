# coding=utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import os
import logging
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from Flatten import Flatten
from NPN import NPN
from dataset.dataset_img import DatasetImg


class MultiRunner(object):
    def __init__(self, batch_size, lr):
        self.epoch = 100
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.batch_size = batch_size
        self.lr = lr


    def fit(self, model, data_loader, device):
        # optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.02)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        train_loss = 0
        train_dict = {}
        count = 0
        acc = 0
        for j, data in tqdm(enumerate(data_loader), leave=False, ncols=100, mininterval=1):
            imgs, target_set = map(lambda x: x.to(device), data)
            y_pred = model(imgs)
            _, pred = torch.max(y_pred.data, 1)
            loss = self.criterion(y_pred, target_set.to(torch.int64))
            train_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # label = target_set.cpu().detach().numpy()
            acc += torch.sum(pred == target_set.data)
            count += self.batch_size
        Acc = acc / count
        train_dict['Acc'] = Acc
        print('Train Loss: {:.6f}'.format(train_loss / count) + 'Train Acc: {:.6f}'.format(Acc))
        return train_dict

    def train(self, model, data_loader, device, val_loader, test_loader):
        train_best = {'epoch': 0, 'train_Acc': 0, 'val_Acc': 0, 'test_Acc': 0}
        val_best = {'epoch': 0, 'train_Acc': 0, 'val_Acc': 0, 'test_Acc': 0}
        test_best = {'epoch': 0, 'train_Acc': 0, 'val_Acc': 0, 'test_Acc': 0}
        for i in range(self.epoch):
            print('epoch {}'.format(i + 1))
            train_dict = self.fit(model=model, data_loader=data_loader, device=device)
            val_dict = self.val(model=model, data_loader=val_loader, device=device)
            test_dict = self.predict(model=model, data_loader=test_loader, device=device)
            if train_dict['Acc'] > train_best['train_Acc']:
                train_best['train_Acc'] = train_dict['Acc']
                train_best['epoch'] = i + 1
                train_best['val_Acc'] = val_dict['Acc']
                train_best['test_Acc'] = test_dict['Acc']
            if val_dict['Acc'] > val_best['val_Acc']:
                val_best['train_Acc'] = train_dict['Acc']
                val_best['epoch'] = i + 1
                val_best['val_Acc'] = val_dict['Acc']
                val_best['test_Acc'] = test_dict['Acc']
            if test_dict['Acc'] > test_best['test_Acc']:
                test_best['train_Acc'] = train_dict['Acc']
                test_best['epoch'] = i + 1
                test_best['val_Acc'] = val_dict['Acc']
                test_best['test_Acc'] = test_dict['Acc']

        print(
            'best of train: epoch:' + str(train_best['epoch']) + 'train_Acc:{:.6f}'.format(train_best['train_Acc']) +
            'val_Acc:{:.6f}'.format(train_best['val_Acc']) + 'test_Acc:{:.6f}'.format(train_best['test_Acc']))
        print(
            'best of val__: epoch:' + str(val_best['epoch']) + 'train_Acc:{:.6f}'.format(val_best['train_Acc']) +
            'val_Acc:{:.6f}'.format(val_best['val_Acc']) + 'test_Acc:{:.6f}'.format(val_best['test_Acc']))
        print(
            'best of test: epoch:' + str(test_best['epoch']) + 'train_Acc:{:.6f}'.format(test_best['train_Acc']) +
            'val_Acc:{:.6f}'.format(test_best['val_Acc']) + 'test_Acc:{:.6f}'.format(test_best['test_Acc']))



    def val(self, model, data_loader, device):
        gc.collect()
        val_dict = {}
        test_loss = 0
        count = 0
        acc = 0
        for j, data in tqdm(list(enumerate(data_loader)), leave=False, ncols=100, mininterval=1):
            imgs, target_set = map(lambda x: x.to(device), data)
            y_pred = model(imgs)
            _, pred = torch.max(y_pred.data, 1)
            loss = self.criterion(y_pred, target_set.to(torch.int64))
            test_loss += loss.data
            acc += torch.sum(pred == target_set.data)
            count += self.batch_size
        Acc = acc / count
        val_dict['Acc'] = Acc
        print('Val Loss: {:.6f}'.format(test_loss / count) + 'Val Acc: {:.6f}'.format(val_dict['Acc']))
        return val_dict

    def predict(self, model, data_loader, device):
        gc.collect()
        test_loss = 0
        test_dict = {}
        count = 0
        acc = 0
        for j, data in tqdm(list(enumerate(data_loader)), leave=False, ncols=100, mininterval=1):
            imgs, target_set = map(lambda x: x.to(device), data)
            y_pred = model(imgs)
            _, pred = torch.max(y_pred.data, 1)
            loss = self.criterion(y_pred, target_set.to(torch.int64))
            test_loss += loss.data
            acc += torch.sum(pred == target_set.data)
            count += self.batch_size
        Acc = acc / count
        test_dict['Acc'] = Acc
        print('Test Loss: {:.6f}'.format(test_loss / count) + 'Test Acc: {:.6f}'.format(test_dict['Acc']))
        return test_dict
