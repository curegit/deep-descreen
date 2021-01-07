import numpy as np
from random import randrange
from chainer import Variable
from chainer.reporter import report
from chainer.training import StandardUpdater
from chainer.functions import mean_squared_error

class CustomUpdater(StandardUpdater):

	def __init__(self, iterator, optimizer, resolution):
		self.resolution = resolution
		super().__init__(iterator, optimizer)

	def random_crop(self, x):
		width, height = self.resolution
		_, _, h, w = x.shape
		i = randrange(0, w - width)
		j = randrange(0, h - height)
		x = x[:,:,j:j+height,i:i+width]
		return x

	def update_core(self):
		optimizer = self.get_optimizer("main")
		model = optimizer.target

		batch = self.get_iterator("main").next()
		crop = [self.random_crop(xy) for xy in batch]
		x = Variable(model.xp.array([c[0] for c in crop], dtype="float32"))
		p = model(x)
		_, _, h1, w1 = x.shape
		b, _, h, w = p.shape
		h_, w_ = (h1 - h) // 2, (w1 - w) // 2
		y = Variable(model.xp.array([c[1][:,h_:-h_,w_:-w_] for c in crop], dtype="float32"))
		loss = mean_squared_error(p, y) / b
		optimizer.target.cleargrads()
		loss.backward()
		optimizer.update()
		report({"loss": loss})

		batch = self.get_iterator("test").next()
		crop = [self.random_crop(xy) for xy in batch]
		x = Variable(model.xp.array([c[0] for c in crop], dtype="float32"))
		p = model(x)
		_, _, h1, w1 = x.shape
		b, _, h, w = p.shape
		h_, w_ = (h1 - h) // 2, (w1 - w) // 2
		y = Variable(model.xp.array([c[1][:,h_:-h_,w_:-w_] for c in crop], dtype="float32"))
		loss = mean_squared_error(p, y) / b
		report({"loss": loss})
