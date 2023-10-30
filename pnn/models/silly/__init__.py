from torch import nn
from .. import BaseModel

from ..utils import input_size

class SillyModel(BaseModel):
    def __init__(self, channels=256, residual=False):
        super().__init__()
        self.residual = residual
        self.layer1 = self.conv_module(3, channels)
        self.layer2 = self.conv_module(channels, channels)
        self.layer3 = self.conv_module(channels, channels)
        self.layer4 = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=0)


    def conv_module(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=0),
            #nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            #nn.MaxPool2d(kernel_size=2, stride=2)
        )


    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    def input_size(self, s):
        return input_size(input_size(input_size(input_size(s, 3), 5), 5), 5)
