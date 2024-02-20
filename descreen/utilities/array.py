
def unpad[T](x: T, n: int) -> T:
    return x[..., n:-n, n:-n]

def fit_to_smaller(x, y):
    *_, h1, w1 = x.shape
    *_, h2, w2 = y.shape

    h = min(h1, h2)
    w = min(w1, w2)

    h1_start = (h1 - h) // 2
    h1_end = h1_start + h

    w1_start = (w1 - w) // 2
    w1_end = w1_start + w

    h2_start = (h2 - h) // 2
    h2_end = h2_start + h

    w2_start = (w2 - w) // 2
    w2_end = w2_start + w

    x = x[:, :, h1_start:h1_end, w1_start:w1_end]
    y = y[:, :, h2_start:h2_end, w2_start:w2_end]

    return x, y
