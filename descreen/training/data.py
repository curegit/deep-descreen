import random
import torch
from pathlib import Path
from numpy import ndarray
from torch import Tensor
from torch.utils.data import Dataset
from ..image import load_image
from ..utilities import flatten
from ..utilities.array import unpad
from ..utilities.filesys import resolve_path, relaxed_glob_recursively
from ..utilities.subps import halftonecv, magickpng


class HalftonePairDataset(Dataset[ndarray]):

    extensions = ["png", "jpg", "jpeg", "jpe", "jp2", "bmp", "dib", "tif", "tiff", "webp", "avif"]

    min_pitch, max_pitch = 2.14, 5.24

    cmyk_angles: list[tuple[int, ...]] = [
        (15, 45, 90, 75),
        (15, 75, 30, 45),
        (15, 75, 90, 45),
        (105, 75, 90, 15),
        (165, 45, 90, 105),
        ]

    def __init__(self, dirpath: str | Path, cmyk_profile: str | Path, patch_size: int, reduced_padding: int, *, augment:bool=False) -> None:
        super().__init__()
        self.dirpath = resolve_path(dirpath, strict=True)
        self.cmyk_profile = resolve_path(cmyk_profile, strict=True)
        self.reduced_padding = reduced_padding
        self.patch_size = patch_size
        self.files = flatten(relaxed_glob_recursively(dirpath, ext) for ext in self.extensions)


    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[ndarray, ndarray]:
        path = self.files[idx]


        img = Image.open(path)
        margin = 8
        crop_size = self.patch_size + margin
        #print(random.random())
        left = random.randrange(img.width - crop_size)
        upper = random.randrange(img.height - crop_size)

        right = left + crop_size
        lower = upper + crop_size
        img = img.crop((left, upper, right, lower))
        assert img.height >= crop_size
        assert img.width >= crop_size
        import io

        buf = io.BytesIO()
        img.save(buf, format='PNG')

        img_in_bytes = buf.getvalue()

        # ピッチバリエーション
        pitch = random.uniform(self.min_pitch, self.max_pitch)
        # 角度バリエーション
        angles = random.choice(self.cmyk_angles)
        theta = random.random() * 90
        angles = tuple(a + theta for a in angles)

        halftone = halftonecv(img, ["-p", f"{pitch:.8f}", "-a"] + [str(a) for a in angles])

        #
        cmds = []
        wide_x = magickpng(halftone, cmds, png48=True)
        wide_y = magickpng(img, cmds, png48=True)
        #
        x = load_image(wide_x, orient=False, assert16=True)
        y = load_image(wide_y, orient=False, assert16=True)
        #
        x = unpad(x, margin)
        y = unpad(y, margin + self.reduced_padding)
        return x, y



    def as_tensor(self):
        return HalftonePairTensorDataset(self)



class HalftonePairTensorDataset(Dataset[Tensor]):
    def __init__(self, base_dataset: HalftonePairDataset) -> None:
        super().__init__()
        self.base = base_dataset

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        x, y = self.base[idx]
        xt = torch.from_numpy(x)
        yt = torch.from_numpy(y)
        return xt, yt