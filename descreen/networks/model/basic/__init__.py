import torch
import torch.nn as nn
from ... import AbsModule
from ...modules import ResidualBlock
from ....utilities.array import fit_to_smaller_add


class TopLevelModel(AbsModule):
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
        return fit_to_smaller_add(residual, out)

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
        mock_input = torch.zeros((1, 3, input_size, input_size), device=next(self.parameters()).device)
        mock_output = self(mock_input)
        _, _, height, width = mock_output.size()
        return height
