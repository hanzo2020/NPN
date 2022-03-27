import torch
import torch.nn as nn
import torch.nn.functional as F
from Flatten import Flatten
from utils import utils
import numpy as np



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.net = nn.Sequential(#28X28
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#14X14
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#7X7
            Flatten(),

        )
        self.Linear1 = nn.Linear(294, 128)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(64, 2)
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.net(x)
        x = self.Linear1(x)
        x = self.sigmoid(x)
        x = self.Linear2(x)
        # prediction = self.softmax(x)

        return x