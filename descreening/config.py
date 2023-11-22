import sys
from descreening.utilities import file_rel_path

# ImageMagick を呼び出すコマンドライン
magick = ["magick"]

# halftonecv を呼び出すコマンドライン
halftonecv = [sys.executable, "-m", "halftonecv"]

# sRGB プロファイルへのパス
srgb_icc = file_rel_path("profiles/sRGB.icc")

# 広域 RGB プロファイルへのパス
wide_rgb_icc = file_rel_path("profiles/WideGamutCompat-v4.icc")

#
pitches = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10]

#
pitch_range = (0.9, 16)

#
cmyk_angles = [
(15, 45, 90, 75),
(15, 75, 30, 45),
(15, 75, 90, 45),
(105, 75, 90, 15),
(165, 45, 90, 105),
]
