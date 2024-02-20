import cv2
import numpy as np
from pathlib import Path
from numpy import ndarray
from .utilities.filesys import resolve_path


# ファイルパス、パスオブジェクト、またはバイトを受け取り画像を配列としてロードする
def load_image(filelike: str | Path | bytes, *, transpose: bool = True, normalize: bool = True, orient: bool = True, assert16: bool = False) -> ndarray:
    match filelike:
        case str() | Path() as path:
            with open(resolve_path(path), "rb") as fp:
                buffer = fp.read()
        case bytes() as buffer:
            pass
        case _:
            raise ValueError()
    # OpenCV が ASCII パスしか扱えない問題を回避するためにバッファを経由する
    bin = np.frombuffer(buffer, np.uint8)
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH
    if not orient:
        flags |= cv2.IMREAD_IGNORE_ORIENTATION
    img = cv2.imdecode(bin, flags)
    if img.ndim != 3 or img.shape[2] != 3:
        raise RuntimeError()
    if transpose:
        # RGBxHxW にする
        img = img[:, :, [2, 1, 0]].transpose(2, 0, 1)
    match img.dtype:
        case np.uint8:
            if assert16:
                raise RuntimeError()
            if normalize:
                return (img / (2**8 - 1)).astype(np.float32)
            else:
                return img
        case np.uint16:
            if normalize:
                return (img / (2**16 - 1)).astype(np.float32)
            else:
                return img
        case _:
            raise RuntimeError()



def to_pil_image(array):
	srgb = rint(array * 255).clip(0, 255).astype(uint8)
	return Image.fromarray(srgb.transpose(1, 2, 0), "RGB")

def save_image(array, filepath):
	to_pil_image(array).save(filepath)

