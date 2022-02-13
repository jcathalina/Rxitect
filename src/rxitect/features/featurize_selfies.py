import logging
import pickle
from dataclasses import astuple, dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import selfies as sf
from tqdm import tqdm

from globals import root_path
from rxitect.structs.vocabulary import SelfiesVocabulary

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(order=True, frozen=True)
class SelfiesEncodings:
    """A class representing SELFIES encodings generated from a list of SMILES."""

    selfies_list: List[str]
    selfies_vocab: List[str]
    longest_selfies_len: int

    def __iter__(self) -> Iterable:
        return iter(astuple(self))


def generate_selfies_encodings(file_path: str, sep: str = "\t") -> SelfiesEncodings:
    """
    Generates encoding, alphabet and length of largest molecule in
    SELFIES, given a file containing SMILES molecules.
    Args:
        file_path: path to csv file with SMILES column. Column's name must be 'smiles'.
        sep: which separator to use when reading the file with the smiles column.
    Returns:
        SelfiesEncodings: A collection of SELFIES encodings generated from a list of SMILES.
    """
    df = pd.read_csv(file_path, sep=sep)

    try:
        smiles_list = np.array(df["smiles"])
    except KeyError as e:
        raise ValueError(
            f"{e}: CSV file needs to contain column with the name 'smiles'"
        )

    selfies_list = [
        safe_encode_to_selfies(smi)
        for smi in tqdm(smiles_list, desc="Translating SMILES to SELFIES")
    ]
    selfies_list = [selfie for selfie in selfies_list if selfie]

    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add("[nop]")
    selfies_vocab = list(all_selfies_symbols)

    longest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    logger.info("Finished generating SEFLIES encodings...")

    encodings = SelfiesEncodings(
        selfies_list=selfies_list,
        selfies_vocab=selfies_vocab,
        longest_selfies_len=longest_selfies_len,
    )

    return encodings


def safe_encode_to_selfies(smi: str) -> Optional[str]:
    try:
        return sf.encoder(smiles=smi)
    except sf.EncoderError:
        return None


def featurization(selfies_enc: SelfiesEncodings, create_voc_file: bool = False) -> None:
    """Generate the csv files used for training our molecule generator
    
    Args:
        selfies_enc: An object containing encoded information of a SELFIES dataset,
                     including the SELFIES, the vocabulary and the length of the
                     longest selfie.
        create_voc_file: Flag that determines if a vocabulary file should be created.
                         False by default.
    """
    selfies_list, selfies_voc, max_selfie_len = selfies_enc
    tokenized_selfies_list = [" ".join(list(sf.split_selfies(selfie))) for selfie in selfies_list]
    selfies_df = pd.DataFrame(data={"selfies": selfies_list, "token": tokenized_selfies_list})
    selfies_df.to_csv(root_path / "data/processed/selfies_chembl_corpus.csv", index=False)
    
    if create_voc_file:
        with open(root_path / "data/processed/selfies_voc.txt", "w") as f:
            f.write('\n'.join(selfies_voc))


def main():
    chembl_selfies_enc = generate_selfies_encodings(
        root_path / "data/processed/smiles_chembl_corpus.txt"
    )
    # Only generate voc for chembl dataset, already contains all available tokens.
    featurization(chembl_selfies_enc, create_voc_file=True)

    ligand_selfies_enc = generate_selfies_encodings(
        root_path / "data/processed/smiles_ligand_corpus.txt"
    )
    featurization(ligand_selfies_enc)


if __name__ == "__main__":
    main()
