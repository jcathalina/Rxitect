import itertools
from typing import Iterable


def flatten_iterable(iterable: Iterable) -> Iterable:
    return itertools.chain.from_iterable(iterable)