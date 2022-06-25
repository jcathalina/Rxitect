import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from rxitect.chem import mol_utils
from rxitect.structs.vocabulary import SmilesVocabulary

logger = logging.getLogger(__name__)


def generate_smiles_corpus(
    smiles: List[str],
    output_dir: Path,
    standardize: bool = True,
    create_voc_file: bool = False,
    return_dataframe: bool = False,
    is_fine_tune_batch: bool = False,
    min_token_len: int = 10,
    max_token_len: int = 100,
    n_jobs: int = 1,
) -> Optional[pd.DataFrame]:
    """
    This method constructs a dataset with molecules represented as SMILES. Each molecule will be decomposed
    into a series of tokens. In the end, all the tokens will be put into one set as vocabulary.

    Args:
        smiles (List[str]): The file path of input, either .sdf file or tab-delimited file
        output_dir (Path): The directory to write the created files to.
        standardize (bool): If the molecule is should be standardized. The charges will be
                            removed and only the largest fragment will be kept. True by default.
        create_voc_file (bool): If a vocabulary file should be created based on the processed smiles list. False by default.
        return_dataframe (bool): If the dataframe created during corpus generation should be returned in addition to being written to file.
                                 False by default.
        is_fine_tune_batch (bool): If the smiles passed should be used for creating the fine tuning corpus. False by default.
        min_token_len (int): The shortest token length allowed for SMILES included in the final corpus.
        max_token_len (int): The longest token length allowed for SMILES included in the final corpus.
        n_jobs (int): The number of workers that should be used for the methods that can be embarassingly parallelized using joblib.
                    Set this to -1 to use all workers. Defaults to 1 worker (sequential).

    Returns:
        pd.DataFrame (optional): The dataframe containing the final corpus. Only gets returned if `return_dataframe` was set to True.
    """
    voc = SmilesVocabulary()
    words = set()
    canon_smiles = []
    tokenized_smiles = []
    if standardize:
        logger.info("Standardizing SMILES...")
        smiles = set(
            Parallel(n_jobs=n_jobs)(
                delayed(mol_utils.clean_and_canonalize)(smi) for smi in smiles
            )
        )
    for smi in tqdm(smiles, desc="Tokenizing SMILES"):
        tokens = voc.tokenize(smi)
        # Only collect the organic molecules
        if {"C", "c"}.isdisjoint(tokens):
            logger.warning("Non-organic token detected: ", smi)
            continue
        # Remove the metal tokens
        if not {"[Na]", "[Zn]"}.isdisjoint(tokens):
            logger.warning("Metal token detected: ", smi)
            continue
        # control the minimum and maximum of sequence length.
        if min_token_len < len(tokens) <= max_token_len:
            words.update(tokens)
            canon_smiles.append(smi)
            tokenized_smiles.append(" ".join(tokens))

    if create_voc_file:
        # output the vocabulary file
        with open(output_dir / f"smiles_voc.txt", "w") as voc_file:
            voc_file.write("\n".join(sorted(words)))

    corpus_type = "fine_tune" if is_fine_tune_batch else "chembl"
    output_path = output_dir / f"smiles_{corpus_type}_corpus.tsv"
    corpus_df = _write_corpus(
        canon_smiles=canon_smiles,
        output_path=output_path,
        tokenized_smiles=tokenized_smiles,
        return_dataframe=return_dataframe,
    )

    if corpus_df:
        return corpus_df


def _write_corpus(
    canon_smiles: List[str],
    output_path: Path,
    tokenized_smiles: List[str],
    return_dataframe: bool = False,
) -> Optional[pd.DataFrame]:
    """(private) Helper func that writes the dataset file as a tab-delim. file, and optionally returns a dataframe containing this data.

    Args:
        canon_smiles (List[str]): A list of canonalized SMILES (up to the user to make sure this is the case.)
        output_path (Path): The full path to write the corpus file to.
        tokenized_smiles (List[str]): A list of tokenized SMILES. Is expected to be of the same size and in the same order as `canon_smiles`.
        return_dataframe (bool): If the dataframe should be returned from the function, if set to False, only writes to file. Default is False.

    Returns:
        pd.DataFrame (optional): The dataframe containing the final corpus. Only gets returned if `return_dataframe` was set to True.
    """
    corpus_df = pd.DataFrame()
    corpus_df["smiles"] = canon_smiles
    corpus_df["token"] = tokenized_smiles
    corpus_df.drop_duplicates(subset="smiles")
    corpus_df.to_csv(path_or_buf=output_path, sep="\t", index=False)

    if return_dataframe:
        return corpus_df
