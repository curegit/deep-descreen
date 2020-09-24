from chainer import Chain, Sequential
from chainer.links import Convolution2D
from chainer.functions import relu, resize_images, average_pooling_2d, concat

class Encoder(Chain):

	def __init__(self, channels, ksize, repeat):
		super().__init__()
		self.channels = channels
		self.ksize = ksize
		self.repeat = repeat
		with self.init_scope():
			self.from_rgb = Convolution2D(3, channels, ksize=1, stride=1, pad=0)
			self.convs = Sequential(Convolution2D(channels, channels, ksize=ksize, stride=1, pad=0), relu).repeat(repeat)

	def __call__(self, x):
		return self.convs(self.from_rgb(x))

class Upsampler(Chain):

	def __init__(self, channels, ksize, repeat):
		super().__init__()
		self.channels = channels
		self.ksize = ksize
		self.repeat = repeat
		with self.init_scope():
			self.convs = Sequential(Convolution2D(channels * 2, channels * 2, ksize=ksize, stride=1, pad=0), relu).repeat(repeat)
			self.convs.append(Convolution2D(channels * 2, channels, ksize=1, stride=1, pad=0))

	def __call__(self, x, y):
		x_height, x_width = x.shape[2:4]
		z_height, z_width = x_height * 2, x_width * 2
		z = resize_images(x, (z_height, z_width), mode="nearest")
		y_height, y_width = y.shape[2:4]
		height, width = min(x_height, y_height), min(x_width, y_height)
		z_width_sub = (z_width - width) // 2
		z_height_sub = (z_height - height) // 2
		y_width_sub = (y_width - width) // 2
		y_height_sub = (y_height - height) // 2
		zs = z[:, :, z_height_sub:-z_height_sub, z_width_sub:-z_width_sub]
		ys = y[:, :, y_height_sub:-y_height_sub, y_width_sub:-y_width_sub]
		return self.convs(concat((zs, ys), axis=1))

class Decoder(Chain):

	def __init__(self, channels, ksize, repeat):
		super().__init__()
		self.channels = channels
		self.ksize = ksize
		self.repeat = repeat
		with self.init_scope():
			self.convs = Sequential(Convolution2D(channels, channels, ksize=ksize, stride=1, pad=0), relu).repeat(repeat)
			self.to_rgb = Convolution2D(channels, 3, ksize=1, stride=1, pad=0)

	def __call__(self, x):
		return self.to_rgb(self.convs(x))

class Network(Chain):

	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.encoder = Encoder(64, 3, 4)
			self.upsampler = Upsampler(64, 3, 4)
			self.decoder = Decoder(64, 3, 4)

	def __call__(self, x, level):
		xs = [x] + [average_pooling_2d(x, ksize=2 ** (i + 1)) for i in range(level)]
		e_x = [self.encoder(dx) for dx in xs]
		ux = [e_x[-1]]
		for i in range(len(e_x) - 1, 0, -1):
			ux.append(self.upsampler(ux[-1], e_x[i-1]))
		d_x = [self.decoder(u) for u in ux]
		d_x.reverse()
		return d_x
