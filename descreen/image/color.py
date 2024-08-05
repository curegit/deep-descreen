import functools
import numpy as np
import descreen.assets


def srgb_profile_file():
    return descreen.assets.grab_file("profiles", "sRGB-v4.icc")


def wide_profile_file():
    return descreen.assets.grab_file("profiles", "WideGamutCompat-v4.icc")


@functools.cache
def srgb_profile() -> bytes:
    with srgb_profile_file() as path:
        with open(path, "rb") as fp:
            return fp.read()


@functools.cache
def wide_profile() -> bytes:
    with wide_profile_file() as path:
        with open(path, "rb") as fp:
            return fp.read()


# sRGB D65 -> D50 XYZ (D50 PCS)
srgb_primaries = np.array(
    [
        [0.436041, 0.385113, 0.143046],
        [0.222485, 0.716905, 0.060610],
        [0.013920, 0.097067, 0.713913],
    ]
)

srgb_primaries_inv = np.array(
    [
        [3.134187, -1.617209, -0.490694],
        [-0.978749, 1.916130, 0.033433],
        [0.071964, -0.228994, 1.405754],
    ]
)

# Adobe wide gamut D50 -> D50 XYZ
wide_gamut_primaries = np.array(
    [
        [0.71639665, 0.10102557, 0.14678978],
        [0.25869066, 0.72471815, 0.01659118],
        [0.0, 0.05121435, 0.77397393],
    ]
)

wide_gamut_primaries_inv = np.array(
    [
        [1.46251661, -0.18455245, -0.27342076],
        [-0.52284242, 1.4479168, 0.0681228],
        [0.03459682, -0.09580958, 1.28752545],
    ]
)
