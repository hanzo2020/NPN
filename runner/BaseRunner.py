# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import gc
import os
import logging
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from Flatten import Flatten
from NPN import NPN
from dataset.dataset_img import DatasetImg

class BaseRunner(object):
    def __init__(self, epoch=100):
        self.epoch = epoch
        self.criterion = nn.BCELoss(reduction='sum')#二分类损失函数, log的底数为e


    def fit(self, model, data_loader, device):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
        train_loss = 0
        train_dict = {}
        y = 0
        label = 0
        count = 0
        up = 0
        acc = 0
        down = 0
        for j, data in tqdm(enumerate(data_loader)):
            imgs, target_set = map(lambda x: x.to(device), data)
            y_pred = model(imgs)
            npy = y_pred.cpu().detach().numpy()
            for y in range(len(npy)):
                if npy[y] > 0.5:
                    npy[y] = 1
                else:
                    npy[y] = 0
            # y = y_pred.item()
            loss = self.criterion(y_pred, target_set)
            label = target_set.cpu().detach().numpy()
            train_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc += accuracy_score(npy, label)
            # if (y > 0.5 and target_set == 1) or (y < 0.5 and target_set == 0):
            #     up += 1
            # else:
            #     down += 1
            count = count + 1
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
            if test_dict['Acc'] > test_best['val_Acc']:
                test_best['train_Acc'] = train_dict['Acc']
                test_best['epoch'] = i + 1
                test_best['val_Acc'] = val_dict['Acc']
                test_best['test_Acc'] = test_dict['Acc']

        print('best of train: epoch:' + str(train_best['epoch']) + 'train_Acc:' + str(train_best['train_Acc']) + 'val_Acc:' +
              str(train_best['val_Acc']) + 'test_Acc:' + str(train_best['test_Acc']))
        print('best of val__: epoch:' + str(val_best['epoch']) + 'train_Acc:' + str(val_best['train_Acc']) + 'val_Acc:' +
              str(val_best['val_Acc']) + 'test_Acc:' + str(val_best['test_Acc']))
        print('best of test: epoch:' + str(test_best['epoch']) + 'train_Acc:' + str(test_best['train_Acc']) + 'val_Acc:' +
              str(test_best['val_Acc']) + 'test_Acc:' + str(test_best['test_Acc']))


    def val(self, model, data_loader, device):
        gc.collect()
        criterion = nn.BCELoss(reduction='sum')
        val_dict = {}
        test_loss = 0
        count = 0
        up = 0
        acc = 0
        down = 0
        for i, data in tqdm(enumerate(data_loader)):
            imgs, target_set = map(lambda x: x.to(device), data)
            y_pred = model(imgs)
            npy = y_pred.cpu().detach().numpy()
            for y in range(len(npy)):
                if npy[y] > 0.5:
                    npy[y] = 1
                else:
                    npy[y] = 0
            loss = self.criterion(y_pred, target_set)
            label = target_set.cpu().detach().numpy()
            test_loss += loss.data
            acc += accuracy_score(npy, label)
            count += 1
        Acc = acc / count
        val_dict['Acc'] = Acc
        print('Val Loss: {:.6f}'.format(test_loss / count) + 'Val Acc: {:.6f}'.format(Acc))
        return val_dict

    def predict(self, model, data_loader, device):
        gc.collect()
        criterion = nn.BCELoss(reduction='sum')
        test_dict = {}
        test_loss = 0
        count = 0
        acc = 0
        up = 0
        down = 0
        for i, data in tqdm(enumerate(data_loader)):
            imgs, target_set = map(lambda x: x.to(device), data)
            y_pred = model(imgs)
            npy = y_pred.cpu().detach().numpy()
            for y in range(len(npy)):
                if npy[y] > 0.5:
                    npy[y] = 1
                else:
                    npy[y] = 0
            loss = self.criterion(y_pred, target_set)
            label = target_set.cpu().detach().numpy()
            test_loss += loss.data
            acc += accuracy_score(npy, label)
            count += 1
        Acc = acc / count
        test_dict['Acc'] = Acc
        print('Test Loss: {:.6f}'.format(test_loss / count) + 'Test Acc: {:.6f}'.format(Acc))
        return test_dict
