import torch
import torch.nn as nn
import torch.nn.functional as F
from Flatten import Flatten
from utils import utils
import numpy as np



class LeNet5(nn.Module):
    def __init__(self, device):
        super(LeNet5, self).__init__()
        self.device = device
        self.net = nn.Sequential(#28X28
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),#14X14
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),#7X7
            Flatten(),

        )
        self.Linear1 = nn.Linear(16*5*5, 120)
        self.sigmoid = nn.Sigmoid()
        self.Linear2 = nn.Linear(120, 2)
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.net(x)
        x = self.Linear1(x)
        x = self.sigmoid(x)
        x = self.Linear2(x)
        # prediction = self.softmax(x)

        return x