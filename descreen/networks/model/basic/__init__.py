import torch
import torch.nn as nn
from descreen.networks.model import DescreenModel
from descreen.networks.resnet import RepeatedResidualBlock
from descreen.networks.modules import ResidualBlock
from descreen.utilities.array import fit_to_smaller, fit_to_smaller_add
from descreen.networks.utils import input_size, output_size


class BasicModel(DescreenModel):
    def __init__(self, channels=(128, 128), ns=(8, 8), k=25):
        super().__init__()
        internal_channels, l_c = channels
        N, M = ns
        self.N = N
        self.k = k
        in_channels = out_channels = 3
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=1, padding=0)
        self.blocks = nn.ModuleList([ResidualBlock(internal_channels, k, nn.ReLU()) for _ in range(N)])
        self.conv2 = nn.Conv2d(internal_channels, out_channels, kernel_size=1, padding=0)
        self.resnet = RepeatedResidualBlock(3 + internal_channels, 3, l_c, n=M)

    def forward_t(self, x):
        residual = x
        out = self.conv1(x)
        for block in self.blocks:
            out = block(out)
        innerout = out
        out = self.conv2(out)
        z, _ = fit_to_smaller(x, innerout)
        r = self.resnet(torch.cat((innerout, z), dim=1))
        h = fit_to_smaller_add(out, r)
        m, ff = fit_to_smaller(out, h)
        return m, ff

    @classmethod
    def alias(cls) -> str:
        return "basic"

    @property
    def multiple_of(self) -> int:
        return 1

    def input_size_unchecked(self, output_size):
        n = 1024 - self.output_size(1024)
        return output_size + n

    def output_size_unchecked(self, input_size):
        i = repeated(lambda x: output_size(x, self.k), self.N)(input_size)
        return self.resnet.output_size(i)
        # mock_input = torch.zeros((1, 3, input_size, input_size), device=next(self.parameters()).device)
        # mock_output = self(mock_input)
        # _, _, height, width = mock_output.size()
        # return height


def repeated(f, n):
    def rfun(p):
        acc = p
        for _ in range(n):
            acc = f(acc)
        return acc

    return rfun
