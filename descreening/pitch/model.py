from torch import nn
from torch.nn import Module


class PitchModel(Module):
    def __init__(self, channels=96, residual=False):
        super().__init__()
        self.residual = residual
        self.layer1 = self.conv_module(1, channels)
        self.layer2 = self.conv_module(channels, channels)
        self.layer3 = self.conv_module(channels, channels)
        self.layer4 = nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=0)
        self.layer5 = nn.LazyLinear(1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.layer5(x)
        x = x.view(-1,)  # Flatten
        return x

    def conv_module(self, in_channels, out_channels, kernel_size=9):
        return nn.Sequential(
            # nn.InstanceNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            # nn.InstanceNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            #nn.MaxPool2d(kernel_size=2, stride=2),
        )
