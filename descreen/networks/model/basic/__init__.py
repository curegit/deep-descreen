import torch
import torch.nn as nn
from .. import DescreenModel
from ...resnet import RepeatedResidualBlock
from ...modules import ResidualBlock
from ....utilities.array import fit_to_smaller, fit_to_smaller_add


class BasicModel(DescreenModel):
    def __init__(self, channels=(128, 128), ns=(8, 8), k=25):
        super().__init__()
        internal_channels, l_c = channels
        N, _ = ns
        in_channels = out_channels = 3
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=3, padding=0)
        self.blocks = nn.ModuleList([ResidualBlock(internal_channels, k, nn.ReLU()) for _ in range(N)])
        self.conv2 = nn.Conv2d(internal_channels, out_channels, kernel_size=3, padding=0)
        self.resnet = RepeatedResidualBlock(6, 3, l_c)

    def forward_t(self, x):
        residual = x
        out = self.conv1(x)
        for block in self.blocks:
            out = block(out)
        out = self.conv2(out)
        z, _ = fit_to_smaller(x, out)
        r = self.resnet(torch.cat((out, z), dim=1))
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
        n = 512 - self.output_size(512)
        return output_size + n

    def output_size_unchecked(self, input_size):
        mock_input = torch.zeros((1, 3, input_size, input_size), device=next(self.parameters()).device)
        mock_output = self(mock_input)
        _, _, height, width = mock_output.size()
        return height
