from ..abs import AbsModel

from torch import nn


def input_size(output_size, kernel_size, stride=1, padding=0):
    return ((output_size - 1) * stride) + kernel_size - 2 * padding


def output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size - kernel_size + 2 * padding) // stride + 1


def fit_to_smaller(x, y):
    b, c, h1, w1 = x.shape
    _, _, h2, w2 = y.shape

    h = min(h1, h2)
    w = min(w1, w2)

    h1_start = (h1 - h) // 2
    h1_end = h1_start + h

    w1_start = (w1 - w) // 2
    w1_end = w1_start + w

    h2_start = (h2 - h) // 2
    h2_end = h2_start + h

    w2_start = (w2 - w) // 2
    w2_end = w2_start + w

    x = x[:, :, h1_start:h1_end, w1_start:w1_end]
    y = y[:, :, h2_start:h2_end, w2_start:w2_end]

    return x, y



class UNetLikeModelLevel(AbsModel):
    def __init__(self, channels=256, bottom=False):
        super().__init__()
        self.bottom = bottom
        if not bottom:
            self.up = Lanczos2xUpsampler(n=2, pad=False)
        self.conv1 = nn.Conv2d(3 if bottom else 3 + channels, channels, kernel_size=3, stride=1, padding=0)
        self.a1 = nn.LeakyReLU(0.1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=0)
        self.a2 = nn.LeakyReLU(0.1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=0)

    def forward(self, x, y=None):
        if not self.bottom:
            h1 = torch.cat(fit_to_smaller(x, self.up(y)), dim=1)
        else:
            h1 = x
            assert y is None
        h2 = self.a1(self.conv1(h1))
        h3 = self.a2(self.conv2(self.bn1(h2)))
        h4 = self.conv3(self.bn2(h3))
        h2_, h4_ = fit_to_smaller(h2, h4)
        return h2_ + h4_

    def input_size(self, output_size):
        if self.bottom:
            return input_size(input_size(input_size(output_size, 5), 5), 3)
        else:
            return input_size(input_size(input_size(output_size, 5), 5), 3) // 2 + 4

    def output_size(self, input_size):
        if self.bottom:
            return output_size(output_size(output_size(input_size, 3), 5), 5)
        else:
            return output_size(output_size(output_size((input_size - 4) * 2, 3), 5), 5)






class UNetLikeModel(AbsModel):
    def __init__(self, channels=128, residual=False):
        super().__init__()
        self.residual = residual
        self.block3 = UNetLikeModelLevel(channels)
        self.block2 = UNetLikeModelLevel(channels)
        self.block1 = UNetLikeModelLevel(channels, bottom=True)
        self.av1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.av2 = nn.AvgPool2d(kernel_size=2, stride=2)
        #self.av3 = nn.AvgPool2d(kernel_size=2, stride=2)
        #self.up1 = Lanczos2xUpsampler(n=2, pad=False)
        #self.up2 = Lanczos2xUpsampler(n=2, pad=False)
        #self.up3 = Lanczos2xUpsampler(n=2, pad=False)
        self.layer4 = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=0)


    def forward(self, x):
        #m1 = self.av1(x)
        m1 = x
        m2 = self.av1(m1)
        m3 = self.av2(m2)
        h1 = self.block1(m3)
        h2 = self.block2(m2, h1)
        h3 = self.block3(m1, h2)
        _h4, r = fit_to_smaller(self.layer4(h3), x)
        return _h4 + r

    def input_size(self, s):
        return self.block1.input_size(self.block2.input_size(self.block3.input_size(input_size(s, 3)))) * 4

    def output_size(self, s):
        s = s // 4
        return output_size(self.block3.output_size(self.block2.output_size(self.block1.output_size(s))), 3)


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def lanczos(x, n):
    return 0.0 if abs(x) > n else np.sinc(x) * np.sinc(x / n)

class Lanczos2xUpsampler(nn.Module):
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
        self.register_buffer('w', w)
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
