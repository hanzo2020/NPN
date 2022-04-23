import torch
import torch.nn as nn
import torch.nn.functional as F
from Flatten import Flatten
from utils import utils
import numpy as np



class NPNCCS(nn.Module):
    def __init__(self, device, class_num = 4):
        super(NPNCCS, self).__init__()
        self.device = device
        self.class_num = class_num
        self.concept_embeddings = torch.nn.Embedding(self.class_num, 32)
        self.concept = torch.tensor([0,1,2,3])
        # self.zero = torch.nn.Parameter(utils.numpy_to_torch(
        #     np.random.uniform(0, 1, size=[1, 64]).astype(np.float32)), requires_grad=False)
        # self.one = torch.nn.Parameter(utils.numpy_to_torch(
        #     np.random.uniform(0, 1, size=[1, 64]).astype(np.float32)), requires_grad=False)
        # self.two = torch.nn.Parameter(utils.numpy_to_torch(
        #     np.random.uniform(0, 1, size=[1, 64]).astype(np.float32)), requires_grad=False)
        # self.three = torch.nn.Parameter(utils.numpy_to_torch(
        #     np.random.uniform(0, 1, size=[1, 64]).astype(np.float32)), requires_grad=False)
        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, 32]).astype(np.float32)), requires_grad=False)
        self.sim_scale = 10
        self.net = nn.Sequential(#224X224
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#112X112
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#56X56
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#28X28
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#64X14X14
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),#128X7X7
            Flatten(),#1568
            nn.Linear(256 * 7 * 7, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.1),
            nn.Linear(4096, 2048),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.1),
            nn.Linear(2048, 32),
            # nn.Sigmoid()
        )
        self.belong_to = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32)
        )


    def forward(self, x):
        prediction, zero_pre, one_pre, two_pre, three_pre = self.predict(x)
        return prediction, zero_pre, one_pre, two_pre, three_pre

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * self.sim_scale
        if sigmoid:
            return result.sigmoid()
            # return result
        return result

    def predict(self, x):
        d = self.net(x)
        concepts = self.concept_embeddings(self.concept.to(self.device))
        #
        zero = concepts[0]
        zero = zero.expand_as(d)
        vector = torch.cat((d, zero), dim=-1)
        zero = self.belong_to(vector)
        zero_pre = self.similarity(zero, self.true, sigmoid=True)
        #
        one = concepts[1]
        one = one.expand_as(d)
        vector = torch.cat((d, one), dim=-1)
        one = self.belong_to(vector)
        one_pre = self.similarity(one, self.true, sigmoid=True)
        #
        two = concepts[2]
        two = two.expand_as(d)
        vector = torch.cat((d, two), dim=-1)
        two = self.belong_to(vector)
        two_pre = self.similarity(two, self.true, sigmoid=True)
        #
        three = concepts[3]
        three = three.expand_as(d)
        vector = torch.cat((d, three), dim=-1)
        three = self.belong_to(vector)
        three_pre = self.similarity(three, self.true, sigmoid=True)
        prediction = torch.stack((zero_pre,one_pre,two_pre,three_pre), dim=1)
        return prediction, zero_pre, one_pre, two_pre, three_pre


