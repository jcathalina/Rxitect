from typing import List

import numpy as np
import selfies as sf
from tqdm import tqdm


def _single_selfies_to_onehot(
    selfies: str, selfies_vocab: List[str], longest_selfies_len: int
) -> np.ndarray:
    """
    Helper function to one-hot encode a single SELFIES repr. molecule.
    """
    symbol_to_int = dict((c, i) for i, c in enumerate(selfies_vocab))

    # pad with [nop]
    selfies += "[nop]" * (longest_selfies_len - sf.len_selfies(selfies))

    # integer encode
    symbol_list = sf.split_selfies(selfies)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(selfies_vocab)
        letter[index] = 1
        onehot_encoded.append(letter)

    return np.array(onehot_encoded)


def selfies_to_onehot(
    selfies_list: List[str], selfies_vocab: List[str], longest_selfies_len: int
) -> np.ndarray:
    """
    Convert a list of SELFIES strings to their one-hot encoding.

    Args:
        selfies_list: TODO
        selfies_vocab: TODO
        longest_selfies_len: TODO

    Returns:
        np.ndarray: An array representing the onehot encoded vectors of a list of SELFIES.
    """
    onehot_encoding = [
        _single_selfies_to_onehot(
            selfies=s,
            selfies_vocab=selfies_vocab,
            longest_selfies_len=longest_selfies_len,
        )
        for s in tqdm(selfies_list, desc="Converting SELFIES to one-hot encoding.")
    ]
    onehot_encoding = np.array(onehot_encoding)

    return onehot_encoding
