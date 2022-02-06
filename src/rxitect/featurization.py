import logging
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import selfies as sf
from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class SelfiesEncodings:
    """A collection of SELFIES encodings generated from a list of SMILES."""

    selfies_list: List[str]
    selfies_vocab: List[str]
    longest_selfies_len: int


def get_selfie_encodings_for_dataset(file_path: str) -> SelfiesEncodings:
    """
    Generates encoding, alphabet and length of largest molecule in
    SELFIES, given a file containing SMILES molecules.
    Args:
        file_path: path to csv file with SMILES column. Column's name must be 'smiles'.
    Returns:
        SelfiesEncodings: A collection of SELFIES encodings generated from a list of SMILES.
    """
    df = pd.read_csv(file_path)

    try:
        smiles_list = np.array(df["smiles"])
    except KeyError as e:
        raise ValueError(
            f"{e}: CSV file needs to contain column with the name 'smiles'"
        )

    selfies_list = [
        sf.encoder(smiles)
        for smiles in tqdm(smiles_list, desc="Translating SMILES to SELFIES")
    ]

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add("[nop]")
    selfies_vocab = list(all_selfies_symbols)

    longest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    logging.info("Finished generating SEFLIES encodings...")

    encodings = SelfiesEncodings(
        selfies_list=selfies_list,
        selfies_vocab=selfies_vocab,
        longest_selfies_len=longest_selfies_len,
    )

    return encodings
