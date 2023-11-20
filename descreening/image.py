import os
import os.path

import subprocess as sp
import cv2
import numpy as np
from tempfile import NamedTemporaryFile, TemporaryDirectory
from descreening.config import magick, halftonecv, wide_rgb_icc
from descreening.utilities import build_filepath, glob_shallowly, relaxed_glob_recursively

def read_uint16_image(filepath):
    x = cv2.imread(filepath, cv2.IMREAD_COLOR | cv2.IMREAD_ANYDEPTH)
    x = x[:, :, [2, 1, 0]].transpose(2, 0, 1)
    if x.dtype != np.uint16:
        raise RuntimeError()
    return x



def to_pil_image(array):
    srgb = np.rint(array * 255).clip(0, 255).astype(np.uint8)
    i = Image.fromarray(srgb.transpose(1, 2, 0), "RGB")
    #i.info.update(icc_profile=ImageCms.getOpenProfile(filerelpath("profiles/sGray.icc"))
    return i

def save_image(array, filepath):
    to_pil_image(array).save(filepath)

import time

#
def cmyk_tiff_to_wide_gamut_uint16_array(filepath, wide_icc_path):
    icc_path = wide_rgb_icc
    with NamedTemporaryFile(dir="../../", suffix=".png", delete=False, mode="w+") as png_tmp:

        sp.run(magick + [filepath, "-intent", "relative", "-profile", icc_path, "PNG48:" + png_tmp.name], check=True)
        time.sleep(1)
    i = read_uint16_image(png_tmp.name)
    os.unlink(png_tmp.name)
    return i


def save_wide_gamut_uint16_array_as_srgb():
    print(out.shape)
    img2 = np.clip(np.rint(out * 65535), 0, 65535).astype(np.uint16).transpose(1, 2, 0)[:,:,[2,1,0]]
    cv2.imwrite(f"{sys.argv[3]}.png", img2)


def rgb_image_to_wide_gamut_uint16_array(filepath):
    with TemporaryDirectory(dir="./tmp") as dirname:
        dest_path = build_filepath(dirname, "tmp", "png")
        sp.run(magick + [filepath, "-intent", "relative", "-profile", icc_path, "PNG48:" + dest_path], check=True)




# RGB 画像をハーフトーン化し、
# ソース画像には ICC プロファイルが埋め込まれている必要がある
# truth_pair が真のとき、
def halftone_rgb_image_to_wide_gamut_uint16_array(filepath, cmyk_icc_path, pitch, angles, truth_pair=False, perceptual=False):
    with NamedTemporaryFile(mode="w", suffix=".tiff", dir="./tmp", delete=False) as cmyk_tmp:
        if perceptual:
            sp.run(magick + [filepath, "-intent", "perceptual", "-black-point-compensation", "-profile", cmyk_icc_path, cmyk_tmp.name], check=True)
        else:
            sp.run(magick + [filepath, "-intent", "relative", "-black-point-compensation", "-profile", cmyk_icc_path, cmyk_tmp.name], check=True)
        with TemporaryDirectory(dir="./tmp") as dirname:

            sp.run(halftonecv + [cmyk_tmp.name, "-d", dirname, "-m", "CMYK", "-o", "CMYK", "-p", f"{pitch:.14f}", "-a"] + [str(a) for a in angles], check=True)
            f = glob_shallowly(dirname, "tiff")
            assert len(f) == 1
            return cmyk_tiff_to_wide_gamut_uint16_array(f[0], "")


            if truth_pair:
                pass
