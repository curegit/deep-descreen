import functools
import operator as op
from typing import Any
from collections.abc import Iterable, Callable


def identity[T](x: T) -> T:
    return x


def once[T](func: Callable[..., T]):
    fst = True
    result = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        nonlocal fst, result
        if fst:
            fst = False
            result = func(*args, **kwargs)
        return result

    return wrapper


def prod(xs: Iterable[Any], start: Any = 1) -> Any:
    return functools.reduce(op.mul, xs, start)


def flatmap[T, S](xs: Iterable[T], f: Callable[[T], Iterable[S]] = identity) -> list[T]:
    return sum((list(f(x)) for x in xs), [])


def range_chunks(length: int, n: int):
    for i in range(0, length, n):
        if i + n < length:
            yield i, i + n
        else:
            yield i, length
