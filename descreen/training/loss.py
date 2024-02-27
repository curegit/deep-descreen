
from torch import Tensor

def total_variation(x: Tensor) -> Tensor:
    b, c, h, w = x.shape
    pixel_dif1 = x[..., 1:, :] - x[..., :-1, :]
    pixel_dif2 = x[..., :, 1:] - x[..., :, :-1]
    reduce_axes = (-3, -2, -1)
    return (pixel_dif1.abs().sum(dim=reduce_axes) + pixel_dif2.abs().sum(dim=reduce_axes)) / (c * h * w * b)
