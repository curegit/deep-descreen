from math import sqrt
from chainer import Link, Chain, Sequential
from chainer.links import Convolution2D, BatchNormalization
from chainer.functions import leaky_relu

class LeakyReluLink(Link):

	def __init__(self, a=0.2):
		super().__init__()
		self.a = a

	def __call__(self, x):
		return leaky_relu(x, self.a)

class ResidualBlock(Chain):

	def __init__(self, channels, ksize):
		super().__init__()
		self.ksize = ksize
		with self.init_scope():
			self.c1 = Convolution2D(channels, channels, ksize)
			self.b1 = BatchNormalization(channels)
			self.a1 = LeakyReluLink()
			self.c2 = Convolution2D(channels, channels, ksize)
			self.b2 = BatchNormalization(channels)
			self.a2 = LeakyReluLink()

	def __call__(self, x):
		n = (self.ksize - 1) // 2 * 2
		h1 = self.a1(self.b1(self.c1(x)))
		h2 = self.a2(self.b2(self.c2(h1)))
		skip = x[:,:,n:-n,n:-n]
		return (h2 + skip) / sqrt(2)

class Network(Chain):

	def __init__(self, channels, blocks, ksize):
		super().__init__()
		with self.init_scope():
			self.frgb = Convolution2D(3, channels, ksize=1)
			self.resnet = Sequential(ResidualBlock(channels, ksize)).repeat(blocks)
			self.trgb = Convolution2D(channels, 3, ksize=1)

	def __call__(self, x):
		return self.trgb(self.resnet(self.frgb(x)))
