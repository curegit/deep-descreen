import os
import os.path
import pathlib
import glob
import random
import torch
from torch.utils.data import Dataset
from numpy import array, uint16, float32


import cv2
import numpy as np


from descreening.pitch import patch_size

from descreening.utilities import load_array


class PitchImageArrayDataset(Dataset):
    def __init__(self, npy_dir, device):
        super().__init__()
        self.img_dir = pathlib.Path(npy_dir).resolve()
        pattern = os.path.join(glob.escape(str(self.img_dir)), "**", "*" + os.extsep + "npy")
        self.files = glob.glob(pattern, recursive=True)
        self.device = device

    def __len__(self):
        return len(self.files) * 3

    def __getitem__(self, idx):
        path = self.files[idx // 3]
        array = load_array(path)
        if array.dtype != uint16:
            raise RuntimeError()
        channels, height, width = array.shape
        assert channels == 3
        assert height == patch_size
        assert width == patch_size
        i = j = 0
        #j = random.randrange(height - patch_size)
        #i = random.randrange(width - patch_size)
        arr = array[idx % 3:idx % 3 + 1, j : j + patch_size, i : i + patch_size]
        x = arr
        y = np.array(float(pathlib.Path(path).stem))
        X = torch.from_numpy((x / (2**16 - 1)).astype(float32))
        Y = torch.from_numpy(y.astype(float32))
        X = X.to(self.device)
        Y = Y.to(self.device)
        return X, Y
