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
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0),#26X26
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),#13X13
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0),#11X11
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),#5X5
            Flatten(),#400
            nn.Linear(400, 256),
            nn.Sigmoid(),
            nn.Linear(256, 2),
        )
        # self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = self.net(x)
        # prediction = self.softmax(x)

        return x