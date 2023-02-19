# coding=utf-8

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import gc
from sklearn.metrics import multilabel_confusion_matrix, cohen_kappa_score
import os
import logging
import argparse
from tqdm import tqdm
import torch.nn.functional as F
from Flatten import Flatten
from NPN import NPN
from dataset.dataset_img import DatasetImg


class NPNMultiRunner(object):
    def __init__(self, batch_size, class_num, lr):
        self.epoch = 5
        # self.criterion = nn.BCELoss(reduction='mean')#二分类损失函数, log的底数为e
        self.criterion = nn.CrossEntropyLoss(reduction='sum')  # 二分类损失函数, log的底数为e
        self.softmax = nn.Softmax(dim=1)
        self.batch_size = batch_size
        self.class_num = class_num
        self.lr = lr


    def fit(self, model, data_loader, device, epoch):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.02)
        train_loss = 0
        train_dict = {}
        count = 0
        acc = 0
        #matrix = np.zeros((self.class_num, 2, 2), dtype=np.int64)
        # model.train()
        for j, data in tqdm(enumerate(data_loader), ncols=100, mininterval=1):
            imgs, target_set = map(lambda x: x.to(device), data)
            y_pred = model(imgs)
            _, pred = torch.max(self.softmax(y_pred), 1)
            loss = self.criterion(y_pred, target_set.to(torch.int64))
            train_loss += loss.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc += torch.sum(pred == target_set.data)
            count += len(pred)
            #cm = multilabel_confusion_matrix(target_set.cpu().detach(), pred.cpu(), labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
            #matrix = matrix + cm
        Acc = acc / count
        train_dict['Acc'] = Acc
        #train_dict['matrix'] = matrix
        print('Train Loss: {:.6f}'.format(train_loss / count) + 'Train Acc: {:.6f}'.format(Acc))
        return train_dict

    def train(self, model, data_loader, device, val_loader, test_loader):
        train_best = {'epoch': 0, 'train_Acc': 0, 'val_Acc': 0, 'test_Acc': 0}
        val_best = {'epoch': 0, 'train_Acc': 0, 'val_Acc': 0, 'test_Acc': 0}
        test_best = {'epoch': 0, 'train_Acc': 0, 'val_Acc': 0, 'test_Acc': 0}
        for i in range(self.epoch):
            print('epoch {}'.format(i + 1))
            train_dict = self.fit(model=model, data_loader=data_loader, device=device, epoch=i)
            val_dict = self.val(model=model, data_loader=val_loader, device=device)
            test_dict = self.predict(model=model, data_loader=test_loader, device=device)
            if train_dict['Acc'] > train_best['train_Acc']:
                train_best['train_Acc'] = train_dict['Acc']
                train_best['epoch'] = i + 1
                train_best['val_Acc'] = val_dict['Acc']
                train_best['test_Acc'] = test_dict['Acc']
                #train_best['train_matrix'] = train_dict['matrix']
                #train_best['val_matrix'] = val_dict['matrix']
                #train_best['test_matrix'] = test_dict['matrix']
            if val_dict['Acc'] > val_best['val_Acc']:
                val_best['train_Acc'] = train_dict['Acc']
                val_best['epoch'] = i + 1
                val_best['val_Acc'] = val_dict['Acc']
                val_best['test_Acc'] = test_dict['Acc']
                #val_best['train_matrix'] = train_dict['matrix']
                #val_best['val_matrix'] = val_dict['matrix']
                val_best['test_matrix'] = test_dict['matrix']
                val_best['test_kappa'] = test_dict['kappa']
                torch.save(model, 'net.pth')
                print('save model, epoch=' + str(val_best['epoch']))
            if test_dict['Acc'] > test_best['test_Acc']:
                test_best['train_Acc'] = train_dict['Acc']
                test_best['epoch'] = i + 1
                test_best['val_Acc'] = val_dict['Acc']
                test_best['test_Acc'] = test_dict['Acc']
                #test_best['train_matrix'] = train_dict['matrix']
                #test_best['val_matrix'] = val_dict['matrix']
                #test_best['test_matrix'] = test_dict['matrix']
        #torch.save(model, 'net.pth')
        print(
            'best of train: epoch:' + str(train_best['epoch']) + 'train_Acc:{:.6f}'.format(train_best['train_Acc']) +
            'val_Acc:{:.6f}'.format(train_best['val_Acc']) + 'test_Acc:{:.6f}'.format(train_best['test_Acc']))
        print(
            'best of val__: epoch:' + str(val_best['epoch']) + 'train_Acc:{:.6f}'.format(val_best['train_Acc']) +
            'val_Acc:{:.6f}'.format(val_best['val_Acc']) + 'test_Acc:{:.6f}'.format(val_best['test_Acc']))
        print(
            'best of test: epoch:' + str(test_best['epoch']) + 'train_Acc:{:.6f}'.format(test_best['train_Acc']) +
            'val_Acc:{:.6f}'.format(test_best['val_Acc']) + 'test_Acc:{:.6f}'.format(test_best['test_Acc']))
        cy = val_best['test_matrix']
        print('best val epoch in test detail:')
        print('best val epoch in test kappa:{:.6f}'.format(val_best['test_kappa']))
        for i in range(cy.shape[0]):
            acc = (cy[i, 0, 0] + cy[i, 1, 1]) / (cy[i, 0, 0] + cy[i, 1, 1] + cy[i, 1, 1] + cy[i, 0, 1])
            pre = cy[i, 0, 0] / (cy[i, 0, 0] + cy[i, 0, 1])
            recall = cy[i, 0, 0] / (cy[i, 0, 0] + cy[i, 1, 0])
            f1 = (2 * (pre * recall)) / (pre + recall)
            print('class' + str(i) + ':' + 'acc:{:.6f}'.format(acc) + 'precision:{:.6f}'.format(pre) + 'recall:{:.6f}'.format(recall)
                  + 'f1:{:.6f}'.format(f1))



    def val(self, model, data_loader, device):
        gc.collect()
        val_dict = {}
        val_loss = 0
        count = 0
        acc = 0
        #matrix = np.zeros((self.class_num, 2, 2), dtype=np.int64)
        with torch.no_grad():
            for j, data in tqdm(enumerate(data_loader), leave=False, ncols=100, mininterval=1):
                imgs, target_set = map(lambda x: x.to(device), data)
                y_pred = model(imgs)
                _, pred = torch.max(self.softmax(y_pred), 1)
                loss = self.criterion(y_pred, target_set.to(torch.int64))
                val_loss += loss.data
                acc += torch.sum(pred == target_set.data)
                count += len(pred)
                #cm = multilabel_confusion_matrix(target_set.cpu().detach(), pred.cpu(), labels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
                #matrix = matrix + cm
        Acc = acc / count
        val_dict['Acc'] = Acc
        #val_dict['matrix'] = matrix
        print('Val Loss: {:.6f}'.format(val_loss / count) + 'Val Acc: {:.6f}'.format(val_dict['Acc']))
        return val_dict

    def predict(self, model, data_loader, device):
        gc.collect()
        test_loss = 0
        test_dict = {}
        total_pred = torch.tensor([])
        total_target = torch.tensor([])
        count = 0
        acc = 0
        matrix = np.zeros((self.class_num, 2, 2), dtype=np.int64)
        # model.eval()
        with torch.no_grad():
            for j, data in tqdm(enumerate(data_loader), leave=False, ncols=100, mininterval=1):
                imgs, target_set = map(lambda x: x.to(device), data)
                y_pred = model(imgs)
                _, pred = torch.max(self.softmax(y_pred), 1)
                loss = self.criterion(y_pred, target_set.to(torch.int64))
                test_loss += loss.data
                acc += torch.sum(pred == target_set.data)
                count += len(pred)
                total_pred = torch.cat((total_pred, pred.cpu()))
                total_target = torch.cat((total_target, target_set.cpu().detach()))
                #cm = multilabel_confusion_matrix(target_set.cpu().detach(), pred.cpu())
                #kappa = cohen_kappa_score(target_set.cpu().detach(), pred)
                #matrix = matrix + cm
        cm = multilabel_confusion_matrix(total_target, total_pred)
        kappa = cohen_kappa_score(total_target, total_pred)
        Acc = acc / count
        test_dict['Acc'] = Acc
        test_dict['matrix'] = cm
        test_dict['kappa'] = kappa
        print('Test Loss: {:.6f}'.format(test_loss / count) + 'Test Acc: {:.6f}'.format(test_dict['Acc']))
        return test_dict
