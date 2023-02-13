import torch
import torch.nn as nn
from torchvision import models

class ResNet50(nn.Module):
    def __init__(self, class_num):
        super(ResNet50, self).__init__()
        self.class_num = class_num
        self.feature = nn.Sequential(
            # models.vgg11_bn(num_classes = self.class_num)
            # models.resnet34(num_classes = self.class_num)
            # models.vgg16_bn()
            models.resnet50(num_classes = self.class_num)
        )
    def forward(self, input):
        out = self.feature(input)
        return out