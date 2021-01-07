import sys
import cv2
import chainer
import numpy as np
import subprocess as sp
from chainer import serializers, Variable
from network import Network

batch = 16
channels = 64
blocks = 8
ksize = 3
device = 0
patch = 256

chainer.global_config.train = True
chainer.global_config.autotune = True
chainer.global_config.cudnn_deterministic = False

model = Network(channels, blocks, ksize)
serializers.load_hdf5(sys.argv[1], model)
if device >= 0: model.to_gpu()

pad = (ksize - 1) * 2 * blocks // 2
stride = patch - pad * 2

img = cv2.imread(sys.argv[2], -1)[:,:,[2,1,0]]
img1 = (img.transpose(2, 0, 1) / 65535).astype("float32")
img1 = np.pad(img1, [(0, 0), (64, 64), (64, 64)], "edge")
h, w = img1.shape[1:3]

out = np.ones((3, h, w), dtype="float32")

for x in range(0, w, stride):
	for y in range(0, h, stride):
		i = min(w - patch, x)
		j = min(h - patch, y)
		x1, y1, x2, y2 = (i, j, i + patch, j + patch)
		x = img1[:,y1:y2,x1:x2].reshape((1, 3, patch, patch))
		print(f"{i}/{w} - {j}/{h}")
		v = Variable(model.xp.array(x))
		t = model(v)[0]
		t.to_cpu()
		out[:,y1:y1+stride,x1:x1+stride] = t.data[0]

print(out.shape)
img2 = np.clip(np.rint(out * 65535), 0, 65535).astype(np.uint16).transpose(1, 2, 0)[:,:,[2,1,0]]
cv2.imwrite(sys.argv[2] + f"-dehalf{sys.argv[1]}.png", img2)
