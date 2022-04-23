import torch
from torch import nn
import numpy as np
from torch.autograd import Variable
from Flatten import Flatten

class VGG10(nn.Module):#input_size == 224X224
    def __init__(self, device):
        super().__init__()
        # 3 * 96 * 96
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  # 32 * 96 * 96
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)# pooling 32 * 48 * 48
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 64 * 48 * 48
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # pooling 64 * 24 * 24
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 128 * 24 * 24
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # pooling 128 * 12 * 12
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 256 * 12 * 12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # pooling 256 * 6 * 6
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),  # 512 * 6 * 6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # pooling 512 * 3 * 3
            Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            # nn.ReLU(),
            # nn.Dropout(p=0.5),
            # nn.Linear(512, 8)
        )

    def forward(self, x):
        out = self.conv1(x)  # 112
        out = self.conv2(out)  # 56
        out = self.conv3(out)  # 28
        out = self.conv4(out)  # 14
        out = self.conv5(out)  # 7
        out = self.fc(out)

        return out
