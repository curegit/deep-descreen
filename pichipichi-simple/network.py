from math import sqrt
from chainer import Link, Chain, Sequential
from chainer.links import Convolution2D, BatchNormalization
from chainer.functions import relu
from chainer.initializers import Normal

class EqualizedConvolution2D(Chain):

	def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, nobias=False, initial_bias=None, gain=sqrt(2)):
		super().__init__()
		self.c = gain * sqrt(1 / (in_channels * ksize ** 2))
		with self.init_scope():
			self.conv = Convolution2D(in_channels, out_channels, ksize, stride, pad, nobias=nobias, initialW=Normal(1.0), initial_bias=initial_bias)

	def __call__(self, x):
		return self.conv(self.c * x)

class Relu(Link):

	def __init__(self):
		super().__init__()

	def __call__(self, x):
		return relu(x)

class ResidualBlock(Chain):

    def __init__(self, channels):
        super().__init__()
        with self.init_scope():
            self.c1 = EqualizedConvolution2D(channels, channels, ksize=3)
            self.b1 = BatchNormalization(channels)
            self.a1 = Relu()
            self.c2 = EqualizedConvolution2D(channels, channels, ksize=3)
            self.b2 = BatchNormalization(channels)

    def __call__(self, x):
        return x[:,:,2:-2,2:-2] + self.b2(self.c2(self.a1(self.b1(self.c1(x)))))

class PichiPichiNetwork(Chain):

    def __init__(self, n, channels):
        super().__init__()
        with self.init_scope():
            self.input = Sequential(EqualizedConvolution2D(3, channels, ksize=3, gain=1), Relu())
            self.blocks = Sequential(*[ResidualBlock(channels) for i in range(n)])
            self.output = Sequential(EqualizedConvolution2D(channels, channels, ksize=3), Relu(), EqualizedConvolution2D(channels, 3, ksize=1, gain=1))

    def __call__(self, x):
        return self.output(self.blocks(self.input(x)))





class Encoder():

