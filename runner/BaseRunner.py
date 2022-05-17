# coding=utf-8

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
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
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.02)
        train_loss = 0
        train_dict = {}
        count = 0
        acc = 0
        pres = []
        labels = []
        for j, data in tqdm(list(enumerate(data_loader))):
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
            pres.extend(list(npy))
            labels.extend(list(label))
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
        Recall = recall_score(np.array(pres),np.array(labels))
        Precision = precision_score(np.array(pres),np.array(labels))
        Acc = acc / count
        train_dict['Acc'] = Acc
        train_dict['Recall'] = Recall
        train_dict['Precision'] = Precision
        print('Train Loss: {:.6f}'.format(train_loss / count) + 'Train Acc: {:.6f}'.format(Acc) + 'Train Rec: {:.6f}'.format(Recall) +
        'Train Pre: {:.6f}'.format(Precision))
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
                train_best['Recall'] = train_dict['Recall']
                train_best['Precision'] = train_dict['Precision']
            if val_dict['Acc'] > val_best['val_Acc']:
                val_best['train_Acc'] = train_dict['Acc']
                val_best['epoch'] = i + 1
                val_best['val_Acc'] = val_dict['Acc']
                val_best['test_Acc'] = test_dict['Acc']
                val_best['val_Recall'] = val_dict['Recall']
                val_best['test_Recall'] = test_dict['Recall']
                val_best['val_Pre'] = val_dict['Precision']
                val_best['test_Pre'] = test_dict['Precision']
            if test_dict['Acc'] > test_best['val_Acc']:
                test_best['train_Acc'] = train_dict['Acc']
                test_best['epoch'] = i + 1
                test_best['val_Acc'] = val_dict['Acc']
                test_best['test_Acc'] = test_dict['Acc']
                test_best['test_Recall'] = test_dict['Recall']
                test_best['val_Recall'] = val_dict['Recall']
                test_best['test_Pre'] = test_dict['Precision']
                test_best['val_Pre'] = val_dict['Precision']

        print(
            'best of train: epoch:' + str(train_best['epoch']) + 'train_Acc:{:.6f}'.format(train_best['train_Acc']) +
            'val_Acc:{:.6f}'.format(train_best['val_Acc']) + 'test_Acc:{:.6f}'.format(train_best['test_Acc']) +
            'train_Recall:{:.6f}'.format(train_best['Recall']) + 'train_Pre:{:.6f}'.format(train_best['Precision']))
        print(
            'best of val__: epoch:' + str(val_best['epoch']) + 'train_Acc:{:.6f}'.format(val_best['train_Acc']) +
            'val_Acc:{:.6f}'.format(val_best['val_Acc']) + 'test_Acc:{:.6f}'.format(val_best['test_Acc']) +
            'val_Recall:{:.6f}'.format(val_best['val_Recall']) + 'test_Recall:{:.6f}'.format(val_best['test_Recall']) +
            'val_Pre:{:.6f}'.format(val_best['val_Pre']) + 'test_Pre:{:.6f}'.format(val_best['test_Pre']))
        print(
            'best of test: epoch:' + str(test_best['epoch']) + 'train_Acc:{:.6f}'.format(test_best['train_Acc']) +
            'val_Acc:{:.6f}'.format(test_best['val_Acc']) + 'test_Acc:{:.6f}'.format(test_best['test_Acc']) +
            'val_Recall:{:.6f}'.format(val_best['val_Recall']) + 'test_Recall:{:.6f}'.format(val_best['test_Recall']) +
            'val_Pre:{:.6f}'.format(val_best['val_Pre']) + 'test_Pre:{:.6f}'.format(val_best['test_Pre']))


    def val(self, model, data_loader, device):
        gc.collect()
        val_dict = {}
        test_loss = 0
        count = 0
        acc = 0
        pres = []
        labels = []
        for i, data in tqdm(list(enumerate(data_loader))):
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
            pres.extend(list(npy))
            labels.extend(list(label))
            test_loss += loss.data
            acc += accuracy_score(npy, label)
            count += 1
        Recall = recall_score(np.array(pres), np.array(labels))
        Precision = precision_score(np.array(pres), np.array(labels))
        Acc = acc / count
        val_dict['Acc'] = Acc
        val_dict['Recall'] = Recall
        val_dict['Precision'] = Precision
        print('Val Loss: {:.6f}'.format(test_loss / count) + 'Val Acc: {:.6f}'.format(Acc) + 'Val Rec: {:.6f}'.format(Recall))
        return val_dict

    def predict(self, model, data_loader, device):
        gc.collect()
        test_dict = {}
        test_loss = 0
        count = 0
        acc = 0
        pres = []
        labels = []
        for i, data in tqdm(list(enumerate(data_loader))):
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
            pres.extend(list(npy))
            labels.extend(list(label))
            test_loss += loss.data
            acc += accuracy_score(npy, label)
            count += 1
        Recall = recall_score(np.array(pres), np.array(labels))
        Precision = precision_score(np.array(pres), np.array(labels))
        Acc = acc / count
        test_dict['Acc'] = Acc
        test_dict['Recall'] = Recall
        test_dict['Precision'] = Precision
        print('Test Loss: {:.6f}'.format(test_loss / count) + 'Test Acc: {:.6f}'.format(Acc) + 'Test Rec: {:.6f}'.format(Recall))
        return test_dict
