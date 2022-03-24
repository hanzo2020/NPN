import torch
import torch.nn as nn
import torch.nn.functional as F
from Flatten import Flatten
from utils import utils
import numpy as np



class NPN(nn.Module):
    def __init__(self):
        super(NPN, self).__init__()
        self.square = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, 32]).astype(np.float32)), requires_grad=False)
        self.sim_scale = 5
        self.net = nn.Sequential(#28X28
            nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#14X14
            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#7X7
            Flatten(),#784
            nn.Linear(441, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            # nn.ReLU(),
            # nn.Linear(64, 1),
            # nn.Sigmoid()
        )
    def forward(self, x):
        prediction = self.predict(x)

        return prediction

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * self.sim_scale
        if sigmoid:
            return result.sigmoid()
        return result

    def predict(self, x):
        d = self.net(x)
        prediction = self.similarity(d, self.square, sigmoid=True)
        return prediction


