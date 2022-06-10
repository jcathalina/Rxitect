import logging
import os
from typing import List, Optional

import hydra
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from rxitect.data.utils import QSARDataset

logger = logging.getLogger(__name__)


def construct_qsar_dataset(
    raw_data_path: str,
    targets: List[str],
    cols: List[str],
    px_placeholder: float = 3.99,
    random_state: int = 42,
    tsplit_year: Optional[int] = None,
    negative_samples: bool = True,
    out_dir: Optional[str] = None,
) -> QSARDataset:
    """Method that constructs a dataset from ChEMBL data to train QSAR regression models on,
    using a temporal split to create a hold-out test dataset for evaluation.

    Args:
        raw_data_path: filepath of the raw data
        targets: ChEMBL IDs that are relevant for the dataset creation
        cols: relevant columns for current dataset creation
        px_placeholder: pChEMBL value to use for negative examples
        random_state: seed integer to ensure reproducibility when randomness is involved
        tsplit_year (optional): year at which the temporal split should happen to create the held-out test set
        negative_samples: boolean flag that determines if negative samples should be included, default is True
        out_dur (optional): filepath where the processed data should be saved to, default is None

    Returns:
        A QSARDataset object - a convenient abstraction.
    """
    # Load and standardize raw data
    df = pd.read_csv(raw_data_path, sep="\t")
    df.columns = df.columns.str.lower()
    df.dropna(subset=["smiles"], inplace=True)

    # Filter data to only contain relevant targets
    df = df[df["target_chembl_id"].isin(targets)]

    # Create temporal split for hold-out test set creation downstream
    smiles_by_year = df.groupby("smiles")["document_year"].min().dropna()
    smiles_by_year = smiles_by_year.astype("Int16")
    tsplit_test_idx = smiles_by_year[smiles_by_year > 2015].index

    # Re-index data to divide SMILES per target
    df = df[cols].set_index(["target_chembl_id", "smiles"])

    # Process positive examples from data, taking the mean of duplicates and removing missing entries
    pos_samples = (
        df["pchembl_value"]
        .groupby(["target_chembl_id", "smiles"])
        .mean()
        .dropna()
    )

    df_processed = pos_samples
    if negative_samples:
        # Process negative examples from data, setting their default pChEMBL values to some threshold (default 3.99)
        # Looks for where inhibition or no activity are implied
        comments = df[(df["comment"].str.contains("Not Active") == True)]
        inhibitions = df[
            (df["standard_type"] == "Inhibition")
            & df["standard_relation"].isin(["<", "<="])
        ]
        relations = df[
            df["standard_type"].isin(["EC50", "IC50", "Kd", "Ki"])
            & df["standard_relation"].isin([">", ">="])
        ]
        neg_samples = pd.concat([comments, inhibitions, relations])
        # Ensure only true negative samples remain in the negative sample set
        neg_samples = neg_samples[~neg_samples.index.isin(pos_samples.index)]
        neg_samples["pchembl_value"] = px_placeholder
        neg_samples = (
            neg_samples["pchembl_value"]
            .groupby(["target_chembl_id", "smiles"])
            .first()
        )  # Regroup indices
        df_processed = pd.concat([pos_samples, neg_samples])

    df_processed = df_processed.unstack("target_chembl_id")

    if tsplit_year:
        idx_test = list(set(df_processed.index).intersection(tsplit_test_idx))
        df_test = df_processed.loc[idx_test]
        df_train = df_processed.drop(df_test.index)
        file_suffix = f"splityear={tsplit_year}.csv"
    else:
        df_train, df_test = train_test_split(
            df_processed, test_size=0.2, random_state=random_state
        )
        file_suffix = f"seed={random_state}.csv"

    qsar_dataset = QSARDataset(
        df_train=df_train.reset_index(drop=False),
        df_test=df_test.reset_index(drop=False),
        targets=targets,
    )

    if out_dir:
        df_test.to_csv(
            os.path.join(out_dir, f"ligand_test_{file_suffix}"), index=True
        )
        df_train.to_csv(
            os.path.join(out_dir, f"ligand_train_{file_suffix}"), index=True
        )

    return qsar_dataset


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    abs_raw_path = abspath(cfg.qsar_dataset.raw.path)
    abs_proc_dir = abspath(cfg.qsar_dataset.processed.dir)
    targets = cfg.qsar_dataset.targets
    cols = cfg.qsar_dataset.cols
    px_placeholder = cfg.qsar_dataset.px_placeholder

    construct_qsar_dataset(
        raw_data_path=abs_raw_path,
        targets=targets,
        cols=cols,
        px_placeholder=px_placeholder,
        tsplit_year=2015,  # TODO: Add to config
        random_state=42,  # TODO: Add to config
        negative_samples=True,  # TODO: Add to config,
        out_dir=abs_proc_dir,
    )


if __name__ == "__main__":
    main()
