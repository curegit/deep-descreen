import sys
import os
import subprocess as sp
import cv2
import numpy as np
from io import BufferedIOBase
from pathlib import Path
from numpy import ndarray
from .utilities.filesys import resolve_path, self_relpath


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


def save_image(img: ndarray, filelike: str | Path | BufferedIOBase, *, transposed: bool = True, prefer16=True, compress=True) -> None:
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

def eprint_sperr(stderr: bytes):
    assert isinstance(stderr, bytes)
    try:
        stderrmsg = stderr.decode(os.device_encoding(2))
    except Exception:
        sys.stderr.buffer.write(stderr)
        sys.stderr.buffer.flush()
    else:
        sys.stderr.write(stderrmsg)
        sys.stderr.flush()

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
        eprint_sperr(e.stderr)
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
        eprint_sperr(e.stderr)
        raise


wide_profile: Path = self_relpath("assets") / "WideGamutCompat-v4.icc"

srgb_profile: Path = self_relpath("assets") / "sRGB-v4.icc"


def magick_has_icc(input_img: bytes) -> bool:
    try:
        cp = sp.run(
            ["magick", "-", "ICC:-"],
            check=True,
            text=False,
            capture_output=True,
            input=input_img,
        )
        if len(cp.stdout) > 0:
            return True
        else:
            return False
    except sp.CalledProcessError as e:
        if e.returncode != 1:
            eprint_sperr(e.stderr)
            raise
        return False


def magick_wide_png(input_img: bytes, *, relative: bool = True, prefer48: bool = True) -> bytes:
    intent = "Relative" if relative else "Perceptual"
    cmds = ["-intent", intent, "-black-point-compensation", "-profile", str(wide_profile)]
    if not magick_has_icc(input_img):
        cmds = ["-profile", str(srgb_profile)] + cmds
    return magick_png(input_img, cmds, png48=prefer48)


def magick_srgb_png(input_img: bytes, *, relative: bool = True, prefer48: bool = False, assume_wide: bool = False) -> bytes:
    intent = "Relative" if relative else "Perceptual"
    cmds = ["-intent", intent, "-black-point-compensation", "-profile", str(srgb_profile)]
    if not magick_has_icc(input_img):
        cmds = ["-profile", str(wide_profile if assume_wide else srgb_profile)] + cmds
    return magick_png(input_img, cmds, png48=prefer48)
