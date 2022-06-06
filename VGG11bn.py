
import torch.nn as nn
from Flatten import Flatten


class VGG11git(nn.Module):
    # implement a simple version of vgg11 (https://arxiv.org/pdf/1409.1556.pdf)
    # the shape of image in CIFAR10 is 32x32x3, much smaller than 224x224x3,
    # the number of channels and hidden units are decreased compared to the architecture in paper
    def __init__(self, class_num):
        super(VGG11git, self).__init__()
        self.class_num = class_num
        self.conv = nn.Sequential(
            # Stage 1
            # TODO: convolutional layer, input channels 3, output channels 8, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 2
            # TODO: convolutional layer, input channels 8, output channels 16, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 3
            # TODO: convolutional layer, input channels 16, output channels 32, filter size 3
            # TODO: convolutional layer, input channels 32, output channels 32, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 4
            # TODO: convolutional layer, input channels 32, output channels 64, filter size 3
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2),
            # Stage 5
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            # TODO: convolutional layer, input channels 64, output channels 64, filter size 3
            # TODO: max-pooling layer, size 2
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc = nn.Sequential(

            # TODO: fully-connected layer (64->64)
            # TODO: fully-connected layer (64->10)
            Flatten(),
            nn.Linear(in_features=3136, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=self.class_num),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x