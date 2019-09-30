from chainer.training import StandardUpdater
from chainer.functions import mean_squared_error

import numpy as np
import cupy as cp
# 
class CustomUpdater(StandardUpdater):

	#
	def __init__(self, train_iter, optimizer, device):
		super().__init__(train_iter, optimizer, device=device)

	#
	def update_core(self):
		# データを1バッチ分取得
		batch = self.get_iterator("main").next()

		# Optimizerを取得
		optimizer = self.get_optimizer("main")
		
		# バッチ分のデータを作る
		x_batch = cp.array([x for x, y in batch], dtype=cp.float32) # 入力データ
		y_batch = cp.array([y for x, y in batch], dtype=cp.float32) # 正解データ
		# ニューラルネットワークを学習させる
		optimizer.update(optimizer.target.loss, x_batch, y_batch)
