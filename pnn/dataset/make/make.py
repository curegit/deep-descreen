import sys
import cv2
import numpy as np

import glob
from ...image import read_image





def make_pair_npy():
    x = read_image()
    y = halftone(x) sp.
    assert 


array = (np.array([x, y]) / 65535).astype("float32")
np.save(dest, array, allow_pickle=False)


def save_pair_npy():


def main():
fx = sys.argv[1]
fy = sys.argv[2]
dest = sys.argv[3]



