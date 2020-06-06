import itertools
from typing import Iterable, Tuple, TypeVar, MutableSequence, Union, Any, Generator

import numpy as np

T = TypeVar("T")


def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def spread_int(a: int, min_bound: int, max_bound: int) -> Iterable[int]:
    """a, min, max -> a, a + 1, a - 1, a + 2, a - 2, ..., min/(max - 1)"""
    yield a
    diff = 1
    done = False
    while not done:
        done = True
        if (x := a + diff) < max_bound:
            yield x
            done = False
        if (x := a - diff) >= min_bound:
            yield x
            done = False
        diff += 1


def swap_same_index(
    seq1: MutableSequence[T], seq2: MutableSequence[T], index: int,
) -> None:
    seq1[index], seq2[index] = seq2[index], seq1[index]


def swap_numpy_same_index(
    array1: np.ndarray, array2: np.ndarray, swap_index: Union[int, Tuple[int, ...]],
) -> None:
    array1[swap_index], array2[swap_index] = (
        array2[swap_index],
        array1[swap_index].copy(),
    )


def generator_value(gen: Generator[Any, None, T]) -> T:
    try:
        while True:
            next(gen)
    except StopIteration as si:
        return si.value
