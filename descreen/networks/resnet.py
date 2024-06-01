
import torch.nn as nn
from torch import Tensor
from . import AbsModule
from .modules import SimpleResidualBlock
from .utils import input_size, output_size

class RepeatedResidualBlock(AbsModule):
    def __init__(self, in_channels, out_channels, inner_channels, activation=nn.ReLU(), n=8) -> None:
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, inner_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(inner_channels, out_channels, kernel_size=1)
        self.blocks = nn.ModuleList([SimpleResidualBlock(inner_channels, activation) for _ in range(n)])

    def forward(self, x: Tensor) -> Tensor:
        out = self.in_conv(x)
        for block in self.blocks:
            out = block(out)
        out = self.out_conv(out)
        return out

    def input_size_unchecked(self, output_size: int) -> int:
        size = output_size
        for _ in range(len(self.blocks)):
            size = input_size(size, 3)
        return size

    def output_size_unchecked(self, input_size: int) -> int:
        size = input_size
        for _ in range(len(self.blocks)):
            size = output_size(size, 3)
        return size

