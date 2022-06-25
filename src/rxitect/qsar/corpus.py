import logging
import os
from typing import List, Optional

import pandas as pd

from rxitect.utils.types import PathLike

logger = logging.getLogger(__name__)


def construct_qsar_dataset(
    raw_data_path: str,
    target: str,
    usecols: List[str],
    sep: str = ";",
    dummy_pchembl_value: float = 3.99,
    out_dir: Optional[PathLike] = None,
) -> None:
    """Method that constructs a dataset from ChEMBL data for a single target to train QSAR regression models on,
    using a temporal split to create a hold-out test dataset for evaluation.

    Args:
        raw_data_path: filepath of the raw data
        target: ChEMBL ID that is relevant for the dataset creation
        usecols: relevant columns for current dataset creation
        sep: seperator to use for the data file being read
        dummy_pchembl_value: pChEMBL value to use for negative examples
        out_dir (optional): filepath where the processed data should be saved to, default is None
    """
    # Load and standardize raw data
    df = pd.read_csv(filepath_or_buffer=raw_data_path, sep=sep, usecols=usecols)
    df = (
        df.pipe(lowercase_cols)
        .pipe(remove_na_smiles)
        .pipe(filter_target, target=target)
        .pipe(set_smiles_as_idx)
        .pipe(preprocess_pchembl_values, dummy_pchembl_value=dummy_pchembl_value)
        .pipe(reset_indices)
    )

    if out_dir:
        df.to_parquet(
            os.path.join(out_dir, f"{target}_dataset_xuhan.pq"),
            index=True,
        )


def lowercase_cols(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    return df


def remove_na_smiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["smiles"])
    return df


def filter_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df[df["target chembl id"] == target]
    return df


def set_smiles_as_idx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.set_index("smiles")
    return df


def preprocess_pchembl_values(
    df: pd.DataFrame, dummy_pchembl_value: float = 3.99
) -> pd.DataFrame:
    numery = df["pchembl value"].groupby("smiles").mean().dropna()
    comments = df[(df["comment"].str.contains("Not Active") == True)]
    inhibits = df[
        (df["standard type"] == "Inhibition")
        & df["standard relation"].isin(["<", "<="])
    ]
    relations = df[
        df["standard type"].isin(["EC50", "IC50", "Kd", "Ki"])
        & df["standard relation"].isin([">", ">="])
    ]
    binary = pd.concat([comments, inhibits, relations], axis=0)
    binary = binary[~binary.index.isin(numery.index)]
    binary["pchembl value"] = dummy_pchembl_value
    binary = binary["pchembl value"].groupby(binary.index).first()
    new_values = pd.concat([numery, binary])
    df.update(new_values)
    return df


def reset_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=False)
    return df
