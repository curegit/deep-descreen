from math import sqrt
import numpy as np
from chainer import Link, Chain, Sequential
from chainer.links import Convolution2D, BatchNormalization
from chainer.functions import leaky_relu, average_pooling_2d, concat, pad, convolution_2d, depth2space

def cubic(b, c):
	def k(x):
		if abs(x) < 1:
			return 1 / 6 * ((12 - 9 * b - 6 * c) * abs(x) ** 3 + (-18 + 12 * b + 6 * c) * abs(x) ** 2 + (6 - 2 * b))
		elif 1 <= abs(x) < 2:
			return 1 / 6 * ((-b - 6 * c) * abs(x) ** 3 + (6 * b + 30 * c) * abs(x) ** 2 + (-12 * b - 48 * c) * abs(x) + (8 * b + 24 * c))
		else:
			return 0
	return k

class Upsampler(Link):

	def __init__(self, b=1/3, c=1/3):
		window = cubic(b, c)
		s = np.array([window(i + 0.25) for i in range(-2, 2)])
		e = np.array([window(i + 0.75) for i in range(-2, 2)])
		k1 = np.pad(s.reshape(1, 4) * s.reshape(4, 1), ((0, 1), (0, 1)))
		k2 = np.pad(e.reshape(1, 4) * s.reshape(4, 1), ((0, 1), (1, 0)))
		k3 = np.pad(s.reshape(1, 4) * e.reshape(4, 1), ((1, 0), (0, 1)))
		k4 = np.pad(e.reshape(1, 4) * e.reshape(4, 1), ((1, 0), (1, 0)))
		self.w = np.array([[k1], [k2], [k3], [k4]], dtype=np.float32)

	def __call__(self, x):
		b, c, h, w = x.shape
		h1 = x.reshape(b * c, 1, h, w)
		h2 = h1
		h3 = convolution_2d(h2, self.xp.array(self.w))
		h4 = depth2space(h3, 2)
		return h4.reshape(b, c, (h - 1) * 2, (w - 1) * 2)

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

class Level(Chain):

	def __init__(self, channels, blocks, ksize):
		super().__init__()
		with self.init_scope():
			self.resnet = Sequential(ResidualBlock(channels, ksize)).repeat(blocks)
			self.c = Convolution2D(channels + 3, channels, ksize=3)
			self.r = LeakyReluLink()
			self.up = Upsampler()

	def __call__(self, x, y):
		h1 = self.resnet(y)
		h2 = self.up(h1)
		n = (x.shape[3] - h2.shape[3]) // 2
		h3 = concat((x[:,:,n:-n,n:-n], h2), axis=1)
		return self.r(self.c(h3))

class TailNet(Chain):

	def __init__(self, channels, tblocks, tksize):
		super().__init__()
		with self.init_scope():
			self.blocks = Sequential(ResidualBlock(channels, tksize)).repeat(tblocks)

	def __call__(self, x):
		return self.blocks(x)

class Network(Chain):

	def __init__(self, channels, blocks, ksize, tblocks, tksize):
		super().__init__()
		with self.init_scope():
			self.frgb = Convolution2D(3, channels, ksize=3)
			self.r1 = LeakyReluLink()
			self.l1 = Level(channels, blocks, ksize)
			self.l2 = Level(channels, blocks, ksize)
			self.l3 = Level(channels, blocks, ksize)
			self.l4 = Level(channels, blocks, ksize)
			self.tail = TailNet(channels, tblocks, tksize)
			self.trgb = Convolution2D(channels, 3, ksize=1)

	def __call__(self, x):
		s1 = x
		s2 = average_pooling_2d(x, ksize=2)
		s4 = average_pooling_2d(x, ksize=4)
		s8 = average_pooling_2d(x, ksize=8)
		s16 = average_pooling_2d(x, ksize=16)
		return self.trgb(self.tail(self.l4(s1, self.l3(s2, self.l2(s4, self.l1(s8, self.r1(self.frgb(s16))))))))
