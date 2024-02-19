import os
import os.path
import pathlib
import glob
import random
import torch
from torch.utils.data import Dataset
from numpy import array, uint16, float32
from numpy.lib.format import open_memmap

import cv2
import numpy as np

from PIL import Image

import torch.utils.data

import os
import os.path
import glob

def mkdirs(dirpath):
	os.makedirs(os.path.normpath(dirpath), exist_ok=True)

def alt_filepath(filepath, suffix="+"):
	while os.path.lexists(filepath):
		root, ext = os.path.splitext(filepath)
		head, tail = os.path.split(root)
		filepath = os.path.join(head, tail + suffix) + ext
	return filepath

def build_filepath(dirpath, filename, fileext, exist_ok=True, suffix="+"):
	filepath = os.path.normpath(os.path.join(dirpath, filename) + os.extsep + fileext)
	return filepath if exist_ok else alt_filepath(filepath, suffix)

def glob_recursively(dirpath, fileext):
	pattern = build_filepath(glob.escape(dirpath), os.path.join("**", "*"), glob.escape(fileext))
	return [f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f)]

def relaxed_glob_recursively(dirpath, fileext):
	lower, upper = fileext.lower(), fileext.upper()
	ls = glob_recursively(dirpath, lower)
	if lower == upper:
		return ls
	ls_upper = glob_recursively(dirpath, upper)
	case_insensitive = len(ls) == len(ls_upper) > 0 and any(os.path.samefile(f, ls_upper[0]) for f in ls)
	if case_insensitive:
		return ls
	ls += ls_upper
	cap = fileext.capitalize()
	if cap == lower or cap == upper:
		return ls
	return ls + glob_recursively(dirpath, cap)



from PIL import Image
from PIL.Image import Resampling
from numpy import rint, asarray, uint8, float32

def load_image(filepath):
	img = Image.convert("RGB")
	return (asarray(img, dtype=uint8).transpose(2, 0, 1) / 255).astype(float32)

def load_image_from_buffer(buffer):
	img = Image.open(buffer).convert("RGB")
	return (asarray(img, dtype=uint8).transpose(2, 0, 1) / 255).astype(float32)

def load_image_uint8(filepath, size):
	img = Image.open(filepath).convert("RGB").resize(size, Resampling.LANCZOS)
	return asarray(img, dtype=uint8).transpose(2, 0, 1)

def uint8_to_float(array):
	return (array / 255).astype(float32)

def from_pil_image(img):
      return (asarray(img.convert("RGB"), dtype=uint8).transpose(2, 0, 1) / 255).astype(float32)

def to_pil_image(array):
	srgb = rint(array * 255).clip(0, 255).astype(uint8)
	return Image.fromarray(srgb.transpose(1, 2, 0), "RGB")

def save_image(array, filepath):
	to_pil_image(array).save(filepath)


import subprocess as sp

class CustomImageArrayDataset_(Dataset):
    extensions = ["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp"]

    def __init__(self, dir, input_patch_size):
        super().__init__()
        self.patch_size = input_patch_size
        self.img_dir = pathlib.Path(dir).resolve()
        #pattern = os.path.join(glob.escape(str(self.img_dir)), "**", "*" + os.extsep + "npy")
        self.files = sum([relaxed_glob_recursively(dir, e) for e in self.extensions], [])
        #print(torch.utils.data.get_worker_info().seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        #mem_array = open_memmap(path, mode="r")
        #if mem_array.dtype != uint16:
        #    raise RuntimeError()

        img = Image.open(path)
        cropsize = self.patch_size + 6
        #print(random.random())
        left = random.randrange(img.width - cropsize)
        upper = random.randrange(img.height - cropsize)

        right = left + cropsize
        lower = upper + cropsize
        img = img.crop((left, upper, right, lower))
        assert img.height >= cropsize
        assert img.width >= cropsize
        import io

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        img_in_bytes = buf.getvalue()
        #try:
        cmyk_angles = [
        (15, 45, 90, 75),
        (15, 75, 30, 45),
        (15, 75, 90, 45),
        (105, 75, 90, 15),
        (165, 45, 90, 105),
        ]
        min_pitch, max_pitch = 2.49, 5.01
        pitch = random.uniform(min_pitch, max_pitch)
        angles = random.choice(cmyk_angles)
        # 角度バリエーションを増やす
        r = random.random() * 90
        angles = tuple([a + r for a in angles])
        #icc = ["-C", "JapanColor2011Coated.icc", "-r", "rel"]
        cp = sp.run(["halftonecv", "-", "-O", "-q"] + ["-p", f"{pitch:.14f}", "-a"] + [str(a) for a in angles], check=False, text=False, capture_output=True, input=img_in_bytes)
        #except:
        #print(cp.stderr.decode("utf-8"))


        halftone = Image.open(io.BytesIO(cp.stdout))

        img = img.crop((3, 3, img.width - 3, img.height - 3))
        halftone = halftone.crop((3, 3, halftone.width - 3, halftone.height - 3))

        y = from_pil_image(img)
        #img.save("test_y.png")
        x = from_pil_image(halftone)
        #halftone.save("test_x.png")
        #j = random.randrange(height - self.patch_size)
        #i = random.randrange(width - self.patch_size)
        #arr = array(mem_array[:, :, j:j + self.patch_size, i:i + self.patch_size])
        #x, y = arr
        return x, y













def read_image(src):
    x = cv2.imread(src, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    x = x[:, :, [2, 1, 0]].transpose(2, 0, 1)
    if x.dtype != np.uint16:
        raise RuntimeError()
    return x

class CustomImageArrayDataset(Dataset):
    def __init__(self, npy_dir, input_patch_size):
        super().__init__()
        self.patch_size = input_patch_size
        self.img_dir = pathlib.Path(npy_dir).resolve()
        pattern = os.path.join(glob.escape(str(self.img_dir)), "**", "*" + os.extsep + "npy")
        self.files = glob.glob(pattern, recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        mem_array = open_memmap(path, mode="r")
        if mem_array.dtype != uint16:
            raise RuntimeError()
        _, _, height, width = mem_array.shape
        assert height >= self.patch_size
        assert width >= self.patch_size

        j = random.randrange(height - self.patch_size)
        i = random.randrange(width - self.patch_size)
        arr = array(mem_array[:, :, j:j + self.patch_size, i:i + self.patch_size])
        x, y = arr
        return x, y


class CustomImageTensorDataset(Dataset):
    def __init__(self, array_dataset, reduced_padding, device):
        super().__init__()
        self.base_dataset = array_dataset
        self.unpad = reduced_padding
        self.device = "cpu" #device

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y = self.base_dataset[idx]
        X = torch.from_numpy(x.astype(float32))
        Y = torch.from_numpy(y.astype(float32))
        Y = Y[:, self.unpad:-self.unpad, self.unpad:-self.unpad]
        #X = X.to(self.device)
        #Y = Y.to(self.device)
        #print(Y.shape)
        return X, Y
