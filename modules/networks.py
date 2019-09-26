from chainer import Chain
from chainer.links import Convolution2D, Deconvolution2D, PReLU

# 
class RgbNet(Chain):

	# ネットワークを定義する
	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.c1 = Convolution2D(3, 64, ksize=17, stride=1, pad=0) #64->48
			self.a1 = PReLU()
			self.c2 = Convolution2D(64, 12, ksize=1, stride=1, pad=0) #->48
			self.a2 = PReLU()
			self.c3 = Convolution2D(12, 12, ksize=3, stride=1, pad=1) #->48
			self.a3 = PReLU()
			self.c4 = Convolution2D(12, 12, ksize=7, stride=1, pad=1) #->44
			self.a4 = PReLU()
			self.c5 = Convolution2D(12, 64, ksize=4, stride=2, pad=1) #->22
			self.a5 = PReLU()
			self.d6 = Deconvolution2D(64, 3, ksize=9, stride=3, pad=4) #->64

	# 順伝播を返す
	def __call__(self, x):
		h1 = self.a1(self.c1(x))
		h2 = self.a2(self.c2(h1))
		h3 = self.a3(self.c3(h2))
		h4 = self.a4(self.c4(h3))
		h5 = self.a5(self.c5(h4))
		return self.d6(h5)
