import gzip
import logging
from pathlib import Path
from typing import List

import pandas as pd
from rdkit import Chem
from tqdm import tqdm

from globals import chembl_26_size, root_path
from rxitect import mol_utils
from rxitect.structs.vocabulary import SmilesVocabulary

logger = logging.getLogger(__name__)
MIN_TOKEN_LEN = 10
MAX_TOKEN_LEN = 100


def generate_smiles_corpus(
    raw_data_filepath: Path,
    destdir: Path,
    corpus_type: str,
    requires_clean: bool = True,
    is_isomeric: bool = False,
    create_voc_file: bool = False,
):
    """
    This method constructs a dataset with molecules represented as SMILES. Each molecule will be decomposed
    into a series of tokens. In the end, all the tokens will be put into one set as vocabulary.

    Args:
        raw_data_filepath (Path): The file path of input, either .sdf file or tab-delimited file
        destdir (Path): The file path to write the created files to.
        corpus_type (str): type of the file to be processed and written. Can be 'chembl' or 'ligand'.
        requires_clean (bool): If the molecule is required to be clean, the charge metal will be
                removed and only the largest fragment will be kept.
        is_isomeric (bool): If the molecules in the dataset keep conformational information. If not,
                the conformational tokens (e.g. @@, @, \, /) will be removed. False by default.
        create_voc_file:
    """
    if corpus_type == "chembl":
        df = get_mols_from_sdf(is_isomeric, raw_data_filepath)
    elif corpus_type == "ligand":
        df = pd.read_table(raw_data_filepath).Smiles.dropna()
    else:
        raise ValueError("Only valid corpus types are 'chembl' and 'ligand'.")

    voc = SmilesVocabulary(vocabulary_file_path=None)
    words = set()
    canons = []
    tokens = []
    if requires_clean:
        smiles = set()
        for smile in tqdm(df, desc="Cleaning molecules"):
            try:
                smile = mol_utils.clean_mol(smile, is_isomeric=is_isomeric)
                smiles.add(Chem.CanonSmiles(smile))
            except Exception as e:
                logger.warning("Parsing Error: ", e)
    else:
        smiles = df.values
    for smile in tqdm(smiles, desc="Tokenizing SMILES"):
        token = voc.tokenize(smile)
        # Only collect the organic molecules
        if {"C", "c"}.isdisjoint(token):
            logger.warning("Non-organic token detected: ", smile)
            continue
        # Remove the metal tokens
        if not {"[Na]", "[Zn]"}.isdisjoint(token):
            logger.warning("Metal token detected: ", smile)
            continue
        # control the minimum and maximum of sequence length.
        if MIN_TOKEN_LEN < len(token) <= MAX_TOKEN_LEN:
            words.update(token)
            canons.append(smile)
            tokens.append(" ".join(token))

    if create_voc_file:
        # output the vocabulary file
        with open(destdir / f"smiles_voc.txt", "w") as voc_file:
            voc_file.write("\n".join(sorted(words)))

    outfile = destdir / f"smiles_{corpus_type}_corpus.txt"
    write_corpus(canon_smiles=canons, outfile=outfile, tokens=tokens)


def write_corpus(canon_smiles, outfile, tokens):
    """Output the dataset file as tab-delimited file"""
    corpus_df = pd.DataFrame()
    corpus_df["smiles"] = canon_smiles
    corpus_df["token"] = tokens
    corpus_df.drop_duplicates(subset="smiles")
    corpus_df.to_csv(path_or_buf=outfile, sep="\t", index=False)


def get_mols_from_sdf(is_isomeric: bool, raw_data_filepath: Path) -> List[str]:
    """Handle sdf file with RDkit"""
    inf = gzip.open(raw_data_filepath)
    fsuppl = Chem.ForwardSDMolSupplier(inf)
    smiles = []
    for mol in tqdm(fsuppl, total=chembl_26_size, desc="Processing ChEMBL molecules"):
        try:
            smiles.append(Chem.MolToSmiles(mol, is_isomeric))
        except Exception as e:
            logger.warning(f"Was not able to convert {mol} to smiles: {e}")
    return smiles


def main():
    generate_smiles_corpus(
        raw_data_filepath=root_path / "data/raw/chembl_26.sdf.gz",
        destdir=root_path / "data/processed",
        corpus_type="chembl",
        create_voc_file=True,
    )

    generate_smiles_corpus(
        raw_data_filepath=root_path / "data/raw/ligand_raw.tsv",
        destdir=root_path / "data/processed",
        corpus_type="ligand",
    )


if __name__ == "__main__":
    main()
