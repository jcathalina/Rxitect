import logging
import os
from typing import List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from rxitect.data.datasets import SingleTargetQSARDataset

logger = logging.getLogger(__name__)


def construct_qsar_dataset(
    raw_data_path: str,
    target: str,
    cols: List[str],
    px_placeholder: float = 3.99,
    random_state: int = 42,
    tsplit_year: Optional[int] = None,
    negative_samples: bool = True,
    out_dir: Optional[str] = None,
) -> SingleTargetQSARDataset:
    """Method that constructs a dataset from ChEMBL data for a single target to train QSAR regression models on,
    using a temporal split to create a hold-out test dataset for evaluation.

    Args:
        raw_data_path: filepath of the raw data
        target: ChEMBL ID that is relevant for the dataset creation
        cols: relevant columns for current dataset creation
        px_placeholder: pChEMBL value to use for negative examples
        random_state: seed integer to ensure reproducibility when randomness is involved
        tsplit_year (optional): year at which the temporal split should happen to create the held-out test set
        negative_samples: boolean flag that determines if negative samples should be included, default is True
        out_dur (optional): filepath where the processed data should be saved to, default is None

    Returns:
        A SingleTargetQSARDataset object - a convenient abstraction for a dataset containing data for a single ChEMBL target.
    """
    # Load and standardize raw data
    df = pd.read_csv(raw_data_path, sep="\t")
    df.columns = df.columns.str.lower()
    df.dropna(subset=["smiles"], inplace=True)

    # Filter data to only contain relevant targets
    df = df[df["target_chembl_id"] == target]
    
    # Re-index data to divide SMILES per target
    df = df[cols].set_index("smiles")

    # Create temporal split for hold-out test set creation downstream
    year = df["document_year"].groupby("smiles").min().dropna()
    tsplit_test_idx = year[year > 2015].index

    # Process positive examples from data, taking the mean of duplicates and removing missing entries
    pos_samples = (
        df["pchembl_value"]
        .groupby("smiles")
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
        neg_samples = pd.concat([comments, inhibitions, relations], axis=0)
        # Ensure only true negative samples remain in the negative sample set
        neg_samples = neg_samples[~neg_samples.index.isin(pos_samples.index)]
        neg_samples["pchembl_value"] = px_placeholder
        neg_samples = (
            neg_samples["pchembl_value"]
            .groupby(neg_samples.index)
            .first()
        )  # Regroup indices
        df_processed = pd.concat([pos_samples, neg_samples])

    df_processed = df_processed.sample(frac=1.0, random_state=random_state)
    
    if tsplit_year:
        idx_test = list(set(df_processed.index).intersection(tsplit_test_idx))
        df_test = df_processed.loc[idx_test].dropna()
        df_train = df_processed.drop(df_test.index)
        file_suffix = f"splityear={tsplit_year}.csv"
    else:
        df_train, df_test = train_test_split(
            df_processed, test_size=0.2, random_state=random_state
        )
        file_suffix = f"seed={random_state}.csv"

    qsar_dataset = SingleTargetQSARDataset(
        df_train=df_train.reset_index(drop=False),
        df_test=df_test.reset_index(drop=False),
        target=target,
    )

    if out_dir:
        df_test.to_csv(
            os.path.join(out_dir, f"ligand_{target}_test_{file_suffix}"), index=True
        )
        df_train.to_csv(
            os.path.join(out_dir, f"ligand_{target}_train_{file_suffix}"), index=True
        )

    return qsar_dataset
