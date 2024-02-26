import torch
import torch.nn as nn

from ..utilities.array import fit_to_smaller
from .abs import AbsModel



def input_size(output_size, kernel_size, stride=1, padding=0):
    return ((output_size - 1) * stride) + kernel_size - 2 * padding


def output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1


class ResidualBlock(AbsModel):
    def __init__(self, in_channels, ksize, activation):
        super(ResidualBlock, self).__init__()
        self.ksize = ksize
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=ksize, groups=in_channels, padding=0)
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

    def input_size(self, output_size):
        return input_size(input_size(input_size(output_size, 3), 3), self.ksize)

    def output_size(self, input_size):
        return output_size(output_size(output_size(input_size, self.ksize), 3), 3)


class TopLevelModel(AbsModel):
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

    def input_size(self, output_size):
        n = 1024 - self.output_size(1024)
        return output_size + n

    def output_size(self, input_size):
        mock_input = torch.zeros((1, 3, input_size, input_size), device=next(self.parameters()).device)
        mock_output = self(mock_input)
        _, _, height, width = mock_output.size()
        return height
