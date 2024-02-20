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
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        array = load_array(path)
        height, width = array.shape
        #assert channels == 3
        assert height == patch_size
        assert width == patch_size
        assert array.dtype == uint16
        x = array.reshape((1, height, width))
        y = np.array(float(pathlib.Path(path).stem))
        X = torch.from_numpy((x / (2**16 - 1)).astype(float32))
        Y = torch.from_numpy(y.astype(float32))
        X = X.to(self.device)
        Y = Y.to(self.device)
        return X, Y
