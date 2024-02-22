import torch
import torch.nn as nn

from ..utilities.array import fit_to_smaller

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, activation):
        super(ResidualBlock, self).__init__()
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=9, groups=in_channels, padding=0)
        self.activation1 = activation
        self.full_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
        self.activation2 = activation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)

    def forward(self, x):
        residual = x
        out = self.depthwise_conv(x)
        out = self.activation1(out)
        out = self.full_conv(out)
        out = self.activation2(out)
        out = self.conv(out)
        r, out = fit_to_smaller(residual, out)
        return out + r


class TopLevelModel(nn.Module):
    def __init__(self, internal_channels, activation, N):
        super(TopLevelModel, self).__init__()
        in_channels = out_channels = 3
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=3, padding=0)
        self.blocks = nn.ModuleList([ResidualBlock(internal_channels, activation) for _ in range(N)])
        self.conv2 = nn.Conv2d(internal_channels, out_channels, kernel_size=3, padding=0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        for block in self.blocks:
            out = block(out)
        out = self.conv2(out)
        r, out = fit_to_smaller(residual, out)
        return out + r
