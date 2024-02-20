import cv2
import numpy as np


def read_image(src):
    x = cv2.imread(src, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    x = x[:, :, [2, 1, 0]].transpose(2, 0, 1)
    if x.dtype != np.uint16:
        raise RuntimeError()
    return x

import sys

import torch

from .image import read_image

from PIL import Image, ImageCms
from PIL import Image
from PIL.Image import Resampling
from numpy import rint, asarray, uint8, float32

import numpy as np

#import rich.progess

def to_pil_image(array):
    srgb = rint(array * 255).clip(0, 255).astype(uint8)
    i = Image.fromarray(srgb.transpose(1, 2, 0), "RGB")
    #i.info.update(icc_profile=ImageCms.getOpenProfile(filerelpath("profiles/sGray.icc"))
    return i

def save_image(array, filepath):
    to_pil_image(array).save(filepath)



print(out.shape)
img2 = np.clip(np.rint(out * 65535), 0, 65535).astype(np.uint16).transpose(1, 2, 0)[:,:,[2,1,0]]
cv2.imwrite(f"{sys.argv[3]}.png", img2)
