import cv2
import numpy as np
from io import BufferedIOBase
from pathlib import Path
from numpy import ndarray
from descreen.utilities.filesys import resolve_path


# ファイルパス、パスオブジェクト、またはバイトを受け取り画像を配列としてロードする（アルファなし）
def load_image(filelike: str | Path | bytes, *, transpose: bool = True, normalize: bool = True, orient: bool = True, assert16: bool = False) -> ndarray:
    match filelike:
        case str() | Path() as path:
            with open(resolve_path(path, strict=True), "rb") as fp:
                buffer = fp.read()
        case bytes() as buffer:
            pass
        case _:
            raise ValueError()
    # OpenCV が ASCII パスしか扱えない問題を回避するためにバッファを経由する
    bin = np.frombuffer(buffer, np.uint8)
    # 任意深度アルファなし BGR
    flags = cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH
    if not orient:
        flags |= cv2.IMREAD_IGNORE_ORIENTATION
    img = cv2.imdecode(bin, flags)
    if img is None:
        raise RuntimeError()
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
        case np.float32:
            return img
        case _:
            raise RuntimeError()


def save_image(img: ndarray, filelike: str | Path | BufferedIOBase, *, transposed: bool = True, prefer16: bool = True, compress: bool = False) -> None:
    match img.dtype:
        case np.float32:
            if prefer16:
                qt = 2**16 - 1
                dtype = np.uint16
            else:
                qt = 2**8 - 1
                dtype = np.uint8
            arr = np.rint(img * qt).clip(0, qt).astype(dtype)
        case np.uint8 | np.uint16:
            arr = img
        case _:
            raise ValueError()
    if transposed:
        # HxWxBGR にする
        arr = arr.transpose(1, 2, 0)[:, :, [2, 1, 0]]
    ok, bin = cv2.imencode(".png", arr, [cv2.IMWRITE_PNG_COMPRESSION, 9 if compress else 0])
    if not ok:
        raise RuntimeError()
    buffer = bin.tobytes()
    match filelike:
        case str() | Path() as path:
            with open(resolve_path(path), "wb") as fp:
                fp.write(buffer)
        case BufferedIOBase() as stream:
            stream.write(buffer)
        case _:
            raise ValueError()
