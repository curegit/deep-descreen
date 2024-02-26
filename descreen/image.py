import subprocess as sp
import cv2
import numpy as np
from io import BufferedIOBase
from pathlib import Path
from numpy import ndarray
from .utilities.filesys import resolve_path, self_relpath




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
        case _:
            raise RuntimeError()



def save_image(img: ndarray, filelike: str | Path | BufferedIOBase, *, transposed: bool = True, prefer16=True):
    match img.dtype:
        case np.float32:
            if prefer16:
                q = 2 ** 16 - 1
                dtype = np.uint16
            else:
                q = 2 ** 8 - 1
                dtype = np.uint8
            print(img)
            a = np.rint(img * q).clip(0, q).astype(dtype)
        case np.uint8 | np.uint16:
            a = img
        case _:
            raise ValueError()

    if transposed:
        # HxWxBGR にする
        a = a.transpose(1, 2, 0)[:, :, [2, 1, 0]]

    ok, bin = cv2.imencode(".png", a, )
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





def halftonecv(input_img: bytes, args: list[str]) -> bytes:
    try:
        cp = sp.run(
            ["halftonecv", "-", "-O", "-q"] + args,
            check=True,
            text=False,
            capture_output=True,
            input=input_img,
        )
        return cp.stdout
    except sp.CalledProcessError as e:
        e.returncode
        match e.stderr:
            case str() as stderr:
                pass
            case bytes() as bstderr:
                bstderr.decode()
        raise


def magick_png(input_img: bytes, args: list[str], *, png48: bool = False) -> bytes:
    try:
        cp = sp.run(
            ["magick", "-"] + args + ["PNG48:-" if png48 else "PNG24:-"],
            check=True,
            text=False,
            capture_output=True,
            input=input_img,
        )
        return cp.stdout
    except sp.CalledProcessError as e:
        e.returncode
        match e.stderr:
            case str() as stderr:
                pass
            case bytes() as bstderr:
                bstderr.decode()
        raise

wide_profile = self_relpath("assets") / "WideGamutCompat-v4.icc"

def magick_wide_png(input_img: bytes, relative=True) -> bytes:

    intent = "Relative" if relative else "Perceptual"
    return magick_png(input_img, ["-intent", intent, "-black-point-compensation", "-profile", str(wide_profile)], png48=True)

def magick_srgb_png(input_img: bytes, relative=True, prefer48:bool=False, assume_wide=True) -> bytes:
    srgb_profile = self_relpath("assets") / "sRGB-v4.icc"
    intent = "Relative" if relative else "Perceptual"
    return magick_png(input_img, ["-profile", str(wide_profile), "-intent", intent, "-black-point-compensation", "-profile", str(srgb_profile)], png48=prefer48)
