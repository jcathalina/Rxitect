import os
import hydra
import omegaconf
import pandas as pd

from typing import List, Union



def _load(filepath: Union[str, bytes, os.PathLike]) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    return df 

def _clean(df: pd.DataFrame, target_chembl_ids: List[str], relevant_cols: List[str]) -> pd.DataFrame:
    df.columns = df.columns.str.lower()
    df.dropna(subset=["smiles"], inplace=True)
    
    df = df[df["target_chembl_id"].isin(target_chembl_ids)]
    df = df[relevant_cols].set_index("smiles")
    df.dropna(subset=["document_year"], inplace=True)
    df["document_year"] = df["document_year"].astype("Int16")  # Nullable integer type

    return df


def _transform(df: pd.DataFrame, px_undef_val: int = 3.99) -> pd.DataFrame:
    pchembl_val = df["pchembl_value"].groupby("smiles").mean().dropna()
    comments = df.query(expr="comment.str.contains('Not Active', na=False)")
    inhibition = df.query(expr="standard_type == 'Inhibition' & standard_relation.isin(['<', '<='])")
    relations = df.query(expr="standard_type.isin(['EC50', 'IC50', 'Kd', 'Ki']) & standard_relation.isin(['>', '>='])")

    bin_features = pd.concat([comments, inhibition, relations], axis=0)
    bin_features = bin_features[~bin_features.index.isin(pchembl_val.index)]
    bin_features["pchembl_value"] = px_undef_val
    bin_features["pchembl_value"] = bin_features["pchembl_value"].groupby(bin_features.index).first()

    df = pd.concat([pchembl_val, bin_features["pchembl_value"]], axis=0)

    return df


@hydra.main(config_path="../config", config_name="main_qsar")
def main(conf: omegaconf.DictConfig):
    to_clf = conf.process.to_classifier  # Flag to transform data for classifiers
    px_active_thresh = conf.process.px_active_thresh  # Threshold value for classifiers

    df = _load(filepath=conf.raw.path)
    df = _clean(df=df,
                target_chembl_ids=conf.process.target_chembl_ids,
                relevant_cols=conf.process.relevant_cols)
    df = _transform(df=df, px_undef_val=conf.process.px_undef_val)

    if to_clf:
        df = (df > px_active_thresh).astype("f4")

    df.to_csv(conf.processed.path)


if __name__ == "__main__":
    main()
