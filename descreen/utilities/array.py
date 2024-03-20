from numpy import ndarray
from torch import Tensor

type Array = ndarray | Tensor


def unpad[T: Array](x: T, n: int) -> T:
    if n < 1:
        raise ValueError()
    if x.shape[-1] < n * 2 or x.shape[-2] < n * 2:
        raise ValueError()
    return x[..., n:-n, n:-n]


def fit_to_smaller[T: Array](x: T, y: T) -> tuple[T, T]:
    *_, h1, w1 = x.shape
    *_, h2, w2 = y.shape
    h = min(h1, h2)
    w = min(w1, w2)
    h1_s = (h1 - h) // 2
    h1_f = h1_s + h
    w1_s = (w1 - w) // 2
    w1_f = w1_s + w
    h2_s = (h2 - h) // 2
    h2_f = h2_s + h
    w2_s = (w2 - w) // 2
    w2_f = w2_s + w
    a = x[..., h1_s:h1_f, w1_s:w1_f]
    b = y[..., h2_s:h2_f, w2_s:w2_f]
    return a, b


def fit_to_smaller_add[T: Array](x: T, y: T) -> T:
    a, b = fit_to_smaller(x, y)
    return a + b
