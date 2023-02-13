import torch
import torch.nn as nn
from torchvision import models

class TestModel(nn.Module):
    def __init__(self, class_num):
        super(TestModel, self).__init__()
        self.class_num = class_num
        self.feature = nn.Sequential(
            # models.vgg11_bn(num_classes = self.class_num)
            # models.resnet34(num_classes = self.class_num)
            # models.vgg16_bn()
            models.regnet_y_400mf()
        )
    def forward(self, input):
        out = self.feature(input)
        return out