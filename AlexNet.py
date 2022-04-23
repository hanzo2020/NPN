import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from Flatten import Flatten
import torchvision.models as models

class AlexNet(nn.Module):#input_size == 224X224
    def __init__(self, device):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=1),
            nn.ReLU()
        )

        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU()
        )

        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.max_pool3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(6400, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4)
        )


    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool3(x)
        x = self.fc(x)
        return x


