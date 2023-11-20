import sys
import cv2
import numpy as np
import os
import glob
#from ...image import read_image

import glob
from pathlib import Path
import subprocess as sp


for


def wide():
  src = "."
  ls = glob.glob(src + "/**/*.tiff", recursive=True)
  for l in ls:
    p = Path(l)
    sp.run(["magick", "convert", str(p), "-intent", "relative", "-profile", "../WideGamutD65.icc", "PNG48:" + str(p.with_suffix(".png"))])


def read_image(src):
    x = cv2.imread(src, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    x = x[:, :, [2, 1, 0]].transpose(2, 0, 1)
    if x.dtype != np.uint16:
        raise RuntimeError()
    return x

x_src = sys.argv[1]
y_src = sys.argv[2]
dest = sys.argv[3]
xs = sorted(glob.glob(x_src + "/**/*.png", recursive=True))
ys = sorted(glob.glob(y_src + "/**/*.png", recursive=True))

os.makedirs(dest, exist_ok=True)

for i, (x, y) in enumerate(zip(xs, ys)):
  x_arr = read_image(x)
  y_arr = read_image(y)
  array = (np.array([x_arr, y_arr]))
  assert array.shape[0] == 2
  assert array.shape[1] == 3
  np.save(dest + f"/{i}.npy", array, allow_pickle=False)
