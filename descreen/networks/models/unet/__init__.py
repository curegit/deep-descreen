from torch import nn, cat
from ... import AbsModule
from ...modules import ResidualBlock, Lanczos2xUpsampler
from ...utils import input_size, output_size
from ....utilities.array import fit_to_smaller, fit_to_smaller_add


class UNetLikeModelLevel(AbsModule):
    def __init__(self, channels=256, N=4, large_k=13, bottom=False):
        super().__init__()
        self.bottom = bottom
        if not self.bottom:
            self.lanczos_n = 3
            self.up = Lanczos2xUpsampler(n=self.lanczos_n, pad=False)
        self.conv1 = nn.Conv2d(3 if bottom else 3 + channels, channels, kernel_size=3, stride=1, padding=0)
        # self.a1 = nn.LeakyReLU(0.1)
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=0)
        # self.a2 = nn.LeakyReLU(0.1)
        # self.conv3 = nn.Conv2d(channels, channels, kernel_size=5, stride=1, padding=0)
        self.blocks = nn.ModuleList([ResidualBlock(channels, ksize=large_k, activation=nn.LeakyReLU(0.1)) for _ in range(N)])
        # self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=0)

    def forward(self, x, y=None):
        if not self.bottom:
            h1 = cat(fit_to_smaller(x, self.up(y)), dim=1)
        else:
            h1 = x
            assert y is None
        out = self.conv1(h1)
        for block in self.blocks:
            out = block(out)
        return out

    @property
    def multiple_of(self) -> int:
        return 2

    def input_size(self, output_size):
        for block in self.blocks:
            output_size = block.input_size(output_size)
        s = input_size(output_size, 3)
        if not self.bottom:
            s = self.up.input_size(s)
        return s

    def output_size(self, input_size):
        if not self.bottom:
            input_size = self.up.output_size(input_size)
        hn = output_size(input_size, 3)
        for block in self.blocks:
            hn = block.output_size(hn)
        return hn


class UNetLikeModel(AbsModule):
    def __init__(self, channels=128):
        super().__init__()
        # self.residual = residual
        self.upper_block = UNetLikeModelLevel(channels)
        self.lower_block = UNetLikeModelLevel(channels, bottom=True)
        self.down = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.block1 = UNetLikeModelLevel(channels)
        # self.av2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.av3 = nn.AvgPool2d(kernel_size=2, stride=2)
        # self.up1 = Lanczos2xUpsampler(n=2, pad=False)
        # self.up2 = Lanczos2xUpsampler(n=2, pad=False)
        # self.up3 = Lanczos2xUpsampler(n=2, pad=False)
        self.out = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        z = self.down(x)
        h1 = self.lower_block(z)
        h2 = self.upper_block(x, h1)
        return fit_to_smaller_add(x, self.out(h2))

    def input_size(self, s):
        return self.lower_block.input_size(self.upper_block.input_size(input_size(s, 3))) * 2

    def output_size(self, s):
        return output_size(self.upper_block.output_size(self.lower_block.output_size(s // 2)), 3)
