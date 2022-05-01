import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from Flatten import Flatten

class VGG16(nn.Module):#input_size == 224X224
    def __init__(self, device):
        super().__init__()
        # 3 * 224 * 224
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),  # 64 * 222 * 222
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),# 64 * 222* 222
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1)# pooling 64 * 112 * 112
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # 128 * 110 * 110
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # 128 * 110 * 110
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1)  # pooling 128 * 56 * 56
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # 256 * 54 * 54
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 256 * 54 * 54
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 256 * 54 * 54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1)  # pooling 256 * 28 * 28
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1),  # 512 * 26 * 26,这里本来变512
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 512 * 26 * 26
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 512 * 26 * 26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1)  # pooling 512 * 14 * 14
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1),  # 512 * 12 * 12
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 512 * 12 * 12
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 512 * 12 * 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1),  # pooling 512 * 7 * 7
            Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 40)
        )

    def forward(self, x):
        out = self.conv1(x)  # 112
        out = self.conv2(out)  # 56
        out = self.conv3(out)  # 28
        out = self.conv4(out)  # 14
        out = self.conv5(out)  # 7
        out = self.fc(out)

        return out
