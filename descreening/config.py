import sys
from descreening.utilities import file_rel_path


# magick コマンドへのパス
magick = ["magick"]

#
halftonecv = [sys.executable, "-m", "halftonecv"]

srgb_icc = file_rel_path("../sRGB.icc")

wide_rgb_icc = file_rel_path("../WideGamutD65.icc")

#
pitches = [1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16]

#
pitch_range = min(pitches), max(pitches)

#
cmyk_angles = [
(15, 45, 90, 75),
(15, 75, 30, 45),
(15, 75, 90, 45),
(105, 75, 90, 15),
(165, 45, 90, 105),
]
