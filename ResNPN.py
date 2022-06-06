import torch
import torch.nn as nn
import torch.nn.functional as F
from Flatten import Flatten
from utils import utils
import numpy as np
from NPNCCS import NPNCCS

class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class ResNPN(NPNCCS):
    def __init__(self, device, class_num, batch_size):
        super(NPNCCS, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.class_num = class_num
        self.obj_dimension = 1024
        self.concept_embeddings = torch.nn.Embedding(self.class_num, self.obj_dimension)
        self.concept = torch.arange(0, self.class_num, dtype=int)
        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, self.obj_dimension]).astype(np.float32)), requires_grad=False)
        self.sim_scale = 10
        self.belong_to = nn.Sequential(
            nn.Linear(2 * self.obj_dimension, 2 * self.obj_dimension),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(2 * self.obj_dimension, self.obj_dimension)
        )
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))

        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Sequential(
            Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            # nn.ReLU(),
            # nn.Dropout(p=0.1),
            # nn.Linear(4096, 2048),
            # nn.LeakyReLU(0.1),
            # nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, self.obj_dimension),
        )

    def forward(self, x):
        # prediction, zero_pre, one_pre, two_pre, three_pre, concepts = self.predict(x)
        # return prediction, zero_pre, one_pre, two_pre, three_pre, concepts
        pre = self.predict(x)
        return pre

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * self.sim_scale
        if sigmoid:
            return result.sigmoid()
            # return result
        return result

    def predict(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        # out = out.reshape(x.shape[0], -1)
        d = self.fc(out)
        concepts = self.concept_embeddings(self.concept.to(self.device))
        # ________________________________
        concepts1 = concepts.expand(d.shape[0], self.class_num, self.obj_dimension)
        d1 = d.expand(self.class_num, d.shape[0], d.shape[1])
        d2 = d1.transpose(0, 1)#4X64X64 to 64X4X64
        concepts2 = torch.cat((d2, concepts1), dim=-1)
        # print(concepts2.shape)
        concepts3 = self.belong_to(concepts2)
        pre = self.similarity(concepts3, self.true, sigmoid=True)
        return pre