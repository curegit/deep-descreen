import sys
import os
import shutil
import subprocess as sp
from functools import cache
from descreen.image.color import srgb_profile_file, wide_profile_file


@cache
def convert_cmd() -> list[str]:
    if shutil.which(cmd := "magick"):
        return [cmd]
    if shutil.which(cmd := "convert"):
        return [cmd]
    raise RuntimeError()


def magick_png(input_img: bytes, args: list[str], *, png48: bool = False) -> bytes:
    try:
        cp = sp.run(
            convert_cmd() + ["-"] + args + ["PNG48:-" if png48 else "PNG24:-"],
            check=True,
            text=False,
            capture_output=True,
            input=input_img,
        )
        return cp.stdout
    except sp.CalledProcessError as e:
        eprint_sperr(e.stderr)
        raise


def has_icc(input_img: bytes) -> bool:
    try:
        cp = sp.run(
            convert_cmd() + ["-", "ICC:-"],
            check=True,
            text=False,
            capture_output=True,
            input=input_img,
        )
        if len(cp.stdout) > 0:
            return True
        else:
            raise RuntimeError()
    except sp.CalledProcessError as e:
        if e.returncode != 1:
            eprint_sperr(e.stderr)
            raise
        return False


def magick_wide_png(input_img: bytes, *, relative: bool = True, prefer48: bool = True, fast: bool = True) -> bytes:
    with srgb_profile_file() as srgb_profile, wide_profile_file() as wide_profile:
        intent = "Relative" if relative else "Perceptual"
        cmds = ["-intent", intent, "-black-point-compensation", "-profile", str(wide_profile)]
        if not has_icc(input_img):
            cmds = ["-profile", str(srgb_profile)] + cmds
        if fast:
            cmds += ["-quality", "10"]
        return magick_png(input_img, cmds, png48=prefer48)


def magick_srgb_png(input_img: bytes, *, relative: bool = True, prefer48: bool = False, assume_wide: bool = False, radical: bool = False) -> bytes:
    with srgb_profile_file() as srgb_profile, wide_profile_file() as wide_profile:
        intent = "Relative" if relative else "Perceptual"
        cmds = ["-intent", intent, "-black-point-compensation", "-profile", str(srgb_profile)]
        if not has_icc(input_img):
            cmds = ["-profile", str(wide_profile if assume_wide else srgb_profile)] + cmds
        if radical:
            cmds += ["-quality", "98"]
        return magick_png(input_img, cmds, png48=prefer48)


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
