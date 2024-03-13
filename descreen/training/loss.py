from torch import Tensor


def total_variation(x: Tensor, mean=True) -> Tensor:
    *b, c, h, w = x.shape
    assert h >= 2 and w >= 2
    diff1 = (x[..., 1:, :] - x[..., :-1, :]).abs()
    diff2 = (x[..., :, 1:] - x[..., :, :-1]).abs()
    reduce = (-3, -2, -1)
    loss = (diff1.sum(dim=reduce) + diff2.sum(dim=reduce)) / (c * (h - 1) * (w - 1))
    if mean:
        return loss.mean()
    else:
        return loss.sum()
