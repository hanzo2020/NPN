import torch
import torch.nn as nn
import torch.nn.functional as F
from Flatten import Flatten
from utils import utils
import numpy as np



class NPN(nn.Module):
    def __init__(self, device):
        super(NPN, self).__init__()
        self.device = device
        self.triangle = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, 64]).astype(np.float32)), requires_grad=False)
        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, 64]).astype(np.float32)), requires_grad=False)
        self.sim_scale = 10
        self.net = nn.Sequential(#28X28
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#14X14
            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#7X7
            Flatten(),#784
            nn.Linear(294, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 1),
            # nn.Sigmoid()
        )
        self.belong_to = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    # def uniform_size(self, vector1, vector2, train=True):
    #     if len(vector1.size()) < len(vector2.size()):
    #         vector1 = vector1.expand_as(vector2)
    #     else:
    #         vector2 = vector2.expand_as(vector1)
    #     if train:
    #         r12 = torch.Tensor(vector1.size()[:-1]).uniform_(0, 1).bernoulli()
    #         # r12 = utils.tensor_to_gpu(r12).unsqueeze(-1)
    #         r12 = r12.unsqueeze(-1).to(self.device)
    #         new_v1 = r12 * vector1 + (-r12 + 1) * vector2
    #         new_v2 = r12 * vector2 + (-r12 + 1) * vector1
    #         return new_v1, new_v2
    #     return vector1, vector2

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
        triangle = self.triangle.expand_as(d)
        # d, triangle = self.uniform_size(d, self.triangle, train=True)
        vector = torch.cat((d, triangle), dim=-1)
        d = self.belong_to(vector)
        prediction = self.similarity(d, self.true, sigmoid=True)
        return prediction


