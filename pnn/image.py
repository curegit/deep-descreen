import cv2
import numpy as np


def read_image(src):
    x = cv2.imread(src, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    x = x[:, :, [2, 1, 0]].transpose(2, 0, 1)
    if x.dtype != np.uint16:
        raise RuntimeError()
    return x
