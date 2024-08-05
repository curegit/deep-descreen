import importlib.resources
import descreen

root = importlib.resources.files(descreen)

asset_root = root / "assets"


def grab_file(*comps):
    p = asset_root
    for c in comps:
        p = p / c
    return importlib.resources.as_file(p)
