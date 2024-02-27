import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import AbsModule
from .utils import input_size, output_size
from ..utilities.array import fit_to_smaller_add


def lanczos(x, n):
    return 0.0 if abs(x) > n else np.sinc(x) * np.sinc(x / n)


class Lanczos2xUpsampler(AbsModule):
    def __init__(self, n=3, pad=True):
        super().__init__()
        start = np.array([lanczos(i + 0.25, n) for i in range(-n, n)])
        end = np.array([lanczos(i + 0.75, n) for i in range(-n, n)])
        s = start / np.sum(start)
        e = end / np.sum(end)
        k1 = np.pad(s.reshape(1, n * 2) * s.reshape(n * 2, 1), ((0, 1), (0, 1)))
        k2 = np.pad(e.reshape(1, n * 2) * s.reshape(n * 2, 1), ((0, 1), (1, 0)))
        k3 = np.pad(s.reshape(1, n * 2) * e.reshape(n * 2, 1), ((1, 0), (0, 1)))
        k4 = np.pad(e.reshape(1, n * 2) * e.reshape(n * 2, 1), ((1, 0), (1, 0)))
        w = torch.tensor(np.array([[k1], [k2], [k3], [k4]], dtype=np.float32))
        self.register_buffer("w", w)
        self.n = n
        self.pad = pad

    def forward(self, x):
        b, c, h, w = x.shape
        h1 = x.view(b * c, 1, h, w)
        if self.pad:
            h2 = F.pad(h1, (self.n, self.n, self.n, self.n), mode="reflect")
        else:
            h2 = h1
        h3 = F.conv2d(h2, self.w)
        h4 = F.pixel_shuffle(h3, 2)
        if self.pad:
            return h4.view(b, c, h * 2, w * 2)
        else:
            return h4.view(b, c, (h - 2 * self.n) * 2, (w - 2 * self.n) * 2)

    def input_size(self, output_size):
        if self.pad:
            return output_size
        else:
            return output_size // 2 + (self.n * 2)

    def output_size(self, input_size):
        return (input_size - (self.n * 2)) * 2


class ResidualBlock(AbsModule):
    def __init__(self, in_channels, ksize, activation):
        super(ResidualBlock, self).__init__()
        self.ksize = ksize
        self.pointwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.activation0 = activation
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=ksize, groups=in_channels, padding=0)
        self.activation1 = activation
        self.full_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
        self.activation2 = activation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        residual = x
        out = self.pointwise_conv(x)
        out = self.activation0(out)
        out = self.depthwise_conv(out)
        out = self.activation1(out)
        out = self.full_conv(out)
        out = self.activation2(out)
        out = self.conv(out)
        return fit_to_smaller_add(residual, out)

    def input_size(self, output_size):
        return input_size(input_size(input_size(output_size, 3), 3), self.ksize)

    def output_size(self, input_size):
        return output_size(output_size(output_size(input_size, self.ksize), 3), 3)
