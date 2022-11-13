from typing import List


def mutless_pop(lst: List, idx: int) -> List:
    """
    Returns a copy of the list without the element at the given index.
    Parameters
    ----------
    lst:
        List to process
    idx:
        Index of element to remove from list
    Returns
    -------
    List:
        The passed list without the element at the given index
    """
    return lst[:idx] + lst[(idx+1):]
