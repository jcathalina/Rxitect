import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from rxitect.utils.smiles import SmilesTokenizer, clean_and_canonalize

logger = logging.getLogger(__name__)


def generate_smiles_corpus(
    smiles: List[str],
    output_dir: Path,
    standardize: bool = True,
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
    tokenizer = SmilesTokenizer()
    canon_smiles = []
    if standardize:
        logger.info("Standardizing SMILES...")
        smiles = set(
            Parallel(n_jobs=n_jobs)(
                delayed(clean_and_canonalize)(smi) for smi in smiles
            )
        )
    for smi in tqdm(smiles, desc="Tokenizing SMILES"):
        tokens = tokenizer.tokenize(smi)
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
            canon_smiles.append(smi)

    if is_fine_tune_batch:
        output_path = output_dir / "fine_tune.clean.smi"
    else:
        output_path = output_dir / "chembl.v30.clean.smi"

    corpus_df = pd.DataFrame(columns=["smiles"], data=canon_smiles)
    corpus_df.to_csv(output_path, index=False)

    if return_dataframe:
        return corpus_df
