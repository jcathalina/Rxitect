import os
import pandas as pd

from typing import Optional, Union


TARGET_CHEMBL_IDS = ["CHEMBL226", "CHEMBL240", "CHEMBL251"]
RELEVANT_COLS = [
        "target_chembl_id",
        "smiles",
        "pchembl_value",
        "comment",
        "standard_type",
        "standard_relation",
        "document_year",
    ]
PX_UNDEF_VAL = 3.99
PX_ACTIVE_THRESH = 6.5


def _load(filepath: Optional[Union[str, bytes, os.PathLike]] = None) -> pd.DataFrame:
    if filepath:
        return pd.read_csv(filepath, sep="\t")

    raise NotImplementedError()


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    df.dropna(subset=["smiles"], inplace=True)
    
    df = df[df["target_chembl_id"].isin(TARGET_CHEMBL_IDS)]
    df = df[RELEVANT_COLS].set_index("smiles")
    df.dropna(subset=["document_year"], inplace=True)
    df["document_year"] = df["document_year"].astype("Int16")  # Nullable integer type

    return df


def _transform(df: pd.DataFrame) -> pd.DataFrame:
    pchembl_val = df["pchembl_value"].groupby("smiles").mean().dropna()
    comments = df.query(expr="comment.str.contains('Not Active', na=False)")
    inhibition = df.query(expr="standard_type == 'Inhibition' & standard_relation.isin(['<', '<='])")
    relations = df.query(expr="standard_type.isin(['EC50', 'IC50', 'Kd', 'Ki']) & standard_relation.isin(['>', '>='])")

    bin_features = pd.concat([comments, inhibition, relations], axis=0)
    bin_features = bin_features[~bin_features.index.isin(pchembl_val.index)]
    bin_features["pchembl_value"] = PX_UNDEF_VAL
    bin_features["pchembl_value"] = bin_features["pchembl_value"].groupby(bin_features.index).first()

    df = pd.concat([pchembl_val, bin_features["pchembl_value"]], axis=0)

    return df


def ligand_dataset(filepath: Optional[Union[str, bytes, os.PathLike]] = None) -> pd.DataFrame:
    df = _load(filepath)
    df = _clean(df)
    df = _transform(df)

    return df


def main():
    raise NotImplementedError()


if __name__ == "__main__":
    main()
