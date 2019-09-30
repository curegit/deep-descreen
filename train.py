from glob import glob
from PIL import Image
from chainer import optimizers
from chainer import serializers
from chainer.iterators import SerialIterator
from chainer.training import Trainer, extensions
from modules.networks import RgbNet, CmykNet
from modules.training import CustomUpdater


import os
import numpy as np

import chainer
import chainer.training
from chainer import training

batch = 32
epoch = 200
device = 0
#i_folder = "dataset/input"
#t_folder = "dataset/desired"

c = 64

#i = glob("dataset/input/*.png")
i = os.listdir("dataset/input")
image_pairs = []
for f in i:
	# 画像を読み込み
    img1 = Image.open("dataset/input/" + f)
    img2 = Image.open("dataset/desired/" + f)
    cur_x = 0
    while cur_x <= img1.width - c:
        cur_y = 0
        while cur_y <= img1.height - c:
        # 画像から切りだし
            rect = (cur_x, cur_y, cur_x + c, cur_y + c)
            cropimg1 = img1.crop(rect)
            cropimg2 = img2.crop(rect)

            image_pairs.append((np.array(cropimg1, dtype=np.float32).transpose((2, 0, 1)) / 255, np.array(cropimg2, dtype=np.float32).transpose((2, 0, 1)) / 255))

            cur_y += 20
        cur_x += 20

iterator = SerialIterator(image_pairs, batch, shuffle=True)

model = RgbNet()
model = model.to_gpu()

optimizer = optimizers.Adam().setup(model)

updater = CustomUpdater(iterator, optimizer, device)

trainer = Trainer(updater, (epoch, "epoch"), out="result")
# 学習の進展を表示するようにする
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.LogReport())

# 中間結果を保存する
n_save = 0
@chainer.training.make_extension(trigger=(10, 'epoch'))
def save_model(trainer):
	# NNのデータを保存
	global n_save
	n_save = n_save+1
	chainer.serializers.save_hdf5( 'a-'+str(n_save)+'.hdf5', model )
trainer.extend(save_model)


# 機械学習を実行する
trainer.run()

model = model.to_cpu()

# 学習結果を保存する
serializers.save_hdf5( "nn2.hdf5", model)
