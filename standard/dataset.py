from numpy import load
from glob import glob
from chainer.dataset import DatasetMixin

class ImageArrayDataset(DatasetMixin):

	def __init__(self, directory):
		super().__init__()
		self.images = glob(f"{directory}/**/*.npz", recursive=True)
		if not self.images:
			raise RuntimeError("empty dataset")

	def __len__(self):
		return len(self.images)

	def get_example(self, index):
		return load(self.images[index])["arr_0"]
