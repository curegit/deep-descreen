from torch import Tensor
from torch.nn.functional import mse_loss
from descreen.utilities import prod


def descreen_loss(pred: Tensor, real: Tensor, *, tv: float = 0.0, gd: float = 0.0) -> Tensor:
    return mse_loss(pred, real) + tv * total_variation(pred) + gd * gradient_difference(pred, real)


def total_variation(x: Tensor, *, mean: bool = True) -> Tensor:
    *_, c, h, w = x.shape
    assert h >= 2 and w >= 2
    diff1 = (x[..., 1:, :] - x[..., :-1, :]).abs()
    diff2 = (x[..., :, 1:] - x[..., :, :-1]).abs()
    reduce = (-3, -2, -1)
    loss = (diff1.sum(dim=reduce) + diff2.sum(dim=reduce)) / prod((c, (h - 1), (w - 1)))
    if mean:
        return loss.mean()
    else:
        return loss.sum()


def gradient_difference(x: Tensor, y: Tensor, *, mean: bool = True) -> Tensor:
    *_, c, h, w = x.shape
    assert x.shape == y.shape
    assert h >= 2 and w >= 2
    diff1 = ((x[..., 1:, :] - x[..., :-1, :]).abs() - (y[..., 1:, :] - y[..., :-1, :]).abs()).abs() * (y[..., 1:, :] - y[..., :-1, :]).abs()
    diff2 = ((x[..., :, 1:] - x[..., :, :-1]).abs() - (y[..., :, 1:] - y[..., :, :-1]).abs()).abs() * (y[..., :, 1:] - y[..., :, :-1]).abs()
    reduce = (-3, -2, -1)
    loss = (diff1.sum(dim=reduce) + diff2.sum(dim=reduce)) / prod((c, (h - 1), (w - 1)))
    if mean:
        return loss.mean()
    else:
        return loss.sum()
