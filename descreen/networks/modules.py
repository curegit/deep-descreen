import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from descreen.networks import AbsModule
from descreen.networks.utils import input_size, output_size
from descreen.utilities.array import fit_to_smaller_add


class Lanczos2xUpsampler(AbsModule):
    def __init__(self, n: int = 3, pad: bool = True) -> None:
        super().__init__()
        start = np.array([self.lanczos(i + 0.25, n) for i in range(-n, n)])
        end = np.array([self.lanczos(i + 0.75, n) for i in range(-n, n)])
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

    def forward(self, x: Tensor) -> Tensor:
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

    def input_size_unchecked(self, output_size: int) -> int:
        if self.pad:
            return output_size
        else:
            return output_size // 2 + (self.n * 2)

    def output_size_unchecked(self, input_size: int) -> int:
        return (input_size - (self.n * 2)) * 2

    @staticmethod
    def lanczos(x: float, n: int) -> float:
        return 0.0 if abs(x) > n else (np.sinc(x) * np.sinc(x / n)).item()


class ResidualBlock(AbsModule):
    def __init__(self, channels, ksize, activation) -> None:
        super().__init__()
        self.ksize = ksize
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.activation0 = activation
        self.depthwise_conv = nn.Conv2d(channels, channels, kernel_size=ksize, groups=channels, padding=0)
        self.activation1 = activation
        # self.full_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        # self.activation2 = activation
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.a = nn.ReLU()  # nn.LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.pointwise_conv(x)
        out = self.activation0(out)
        out = self.depthwise_conv(out)
        out = self.activation1(out)
        out = self.conv(out)
        return self.a(fit_to_smaller_add(residual, out))

    def input_size_unchecked(self, output_size: int) -> int:
        return input_size(input_size(input_size(output_size, 1), 1), self.ksize)

    def output_size_unchecked(self, input_size: int) -> int:
        return output_size(output_size(output_size(input_size, self.ksize), 1), 1)


class SimpleResidualBlock(AbsModule):
    def __init__(self, channels, activation) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.activation = activation
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)
        self.a = nn.ReLU()  # nn.LeakyReLU(0.2)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.conv2(out)
        return self.a(fit_to_smaller_add(residual, out))

    def input_size_unchecked(self, output_size: int) -> int:
        return input_size(input_size(output_size, 3), 3)

    def output_size_unchecked(self, input_size: int) -> int:
        return output_size(output_size(input_size, 3), 3)
