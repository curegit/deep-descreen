from collections.abc import Iterable

def flatten[T](xs: Iterable[Iterable[T]]) -> list[T]:
    return sum([list(x) for x in xs], [])
