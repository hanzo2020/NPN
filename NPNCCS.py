import torch
import torch.nn as nn
import torch.nn.functional as F
from Flatten import Flatten
from utils import utils
import numpy as np



class NPNCCS(nn.Module):
    def __init__(self, device, class_num, batch_size):
        super(NPNCCS, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.class_num = class_num
        self.obj_dimension = 64
        self.concept_embeddings = torch.nn.Embedding(self.class_num, self.obj_dimension)
        self.concept = torch.arange(0, self.class_num, dtype=int)
        self.true = torch.nn.Parameter(utils.numpy_to_torch(
            np.random.uniform(0, 1, size=[1, self.obj_dimension]).astype(np.float32)), requires_grad=False)
        self.sim_scale = 10
        self.net = nn.Sequential(#224X224
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(),#1568
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.2),
            nn.Linear(2048, self.obj_dimension),
        )
        self.belong_to = nn.Sequential(
            nn.Linear(2 * self.obj_dimension, 2 * self.obj_dimension),
            # nn.ReLU(True),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(2 * self.obj_dimension, self.obj_dimension)
        )


    def forward(self, x):
        pre = self.predict(x)
        return pre

    def similarity(self, vector1, vector2, sigmoid=True):
        result = F.cosine_similarity(vector1, vector2, dim=-1)
        result = result * self.sim_scale
        if sigmoid:
            return result.sigmoid()
        return result

    def predict(self, x):
        d = self.net(x)
        concepts = self.concept_embeddings(self.concept.to(self.device))
        concepts1 = concepts.expand(d.shape[0], self.class_num, self.obj_dimension)
        d1 = d.expand(self.class_num, d.shape[0], d.shape[1])
        d2 = d1.transpose(0, 1)#4X64X64 to 64X4X64
        concepts2 = torch.cat((d2, concepts1), dim=-1)
        concepts3 = self.belong_to(concepts2)
        pre = self.similarity(concepts3, self.true, sigmoid=True)
        return pre




