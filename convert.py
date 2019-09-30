from PIL import Image
from modules.networks import RgbNet
from chainer import serializers

import numpy as np
#import cupy as cp

i = Image.open("test2.bmp")
# ニューラルネットワークを作成
model = RgbNet()

# 学習結果を読み込む
serializers.load_hdf5("nn2.hdf5", model)

#model.to_gpu()


# 出力画像
dst = Image.new("RGB", (i.width, i.height), "white")

# 入力画像を分割
cur_x = 0
while cur_x <= i.size[0] - 64:
	cur_y = 0
	while cur_y <= i.size[1] - 64:
		# 画像から切りだし
		rect = (cur_x, cur_y, cur_x+64, cur_y+64)
		cropimg = i.crop(rect)

		x = (np.array(cropimg, dtype=np.uint8).transpose((2, 0, 1)) / 255).astype(np.float32).reshape((1, 3, 64, 64))

		print(x.shape)

		t = model(x)


		print(type(t.data))
		arr = np.rint(t.data.astype(np.float64).reshape((3, 64, 64)).transpose((1, 2, 0)) * 255).astype(np.uint8).flatten()

		himg = Image.frombuffer("RGB", (64, 64), arr, "raw", "RGB", 0, 1)
		#himg = Image.fromarray(bytes, 'raw')
		dst.paste(himg, (cur_x, cur_y))

		cur_y += 64
	cur_x += 64

dst.save("dehalfa4.png")
