import torch
import torch.nn as nn
from torchvision import models

class ResNet34(nn.Module):
    def __init__(self, class_num):
        super(ResNet34, self).__init__()
        self.class_num = class_num
        self.feature = nn.Sequential(
            # models.vgg11_bn(num_classes = self.class_num)
            models.resnet34(num_classes = self.class_num)
            # models.vgg16_bn()
        )
    def forward(self, input):
        out = self.feature(input)
        return out