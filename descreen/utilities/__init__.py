import operator as op
from functools import reduce


def prod(xs, start=1):
    return reduce(op.mul, xs, start)


from collections.abc import Iterable

from typing import TypeVar


T = TypeVar("T")


def flatten(xs: Iterable[Iterable[T]]) -> list[T]:
    return sum([list(x) for x in xs], [])


def range_chunks(length, n):
    for i in range(0, length, n):
        if i + n < length:
            yield i, i + n
        else:
            yield i, length
