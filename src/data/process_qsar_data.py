import hydra
import pandas as pd
from hydra.core.config_store import ConfigStore

from src.structs.qsar_data_config import QSARDataConfig

cs = ConfigStore.instance()
cs.store(name="qsar_data", node=QSARDataConfig)


def process_qsar_data(cfg: QSARDataConfig) -> pd.DataFrame:
    """Function that loads and processed ChEMBL data for specific ligands
    to be used for training QSAR models.

    Args:
        cfg: A dictionary used to configure the parameters and filepaths
                              used for the processing of the ChEMBL data into usable features
                              for an eventual QSAR model.

    Returns:
        A DataFrame that contains structured information with SMILES and their respective
        features that will be used in the training of a QSAR model.
    """
    # Load
    df = pd.read_csv(cfg.raw.path, sep="\t")
    df.columns = df.columns.str.lower()
    df.dropna(subset=["smiles"], inplace=True)

    # Clean
    df = df[df["target_chembl_id"].isin(cfg.params.targets)]
    df = df[cfg.params.cols].set_index("smiles")
    df.dropna(subset=["document_year"], inplace=True)
    df["document_year"] = df["document_year"].astype(
        "Int16"
    )  # Nullable integer type

    # Transform
    pchembl_val = df["pchembl_value"].groupby("smiles").mean().dropna()
    comments = df.query(expr="comment.str.contains('Not Active', na=False)")
    inhibition = df.query(
        expr="standard_type == 'Inhibition' & standard_relation.isin(['<', '<='])"
    )
    relations = df.query(
        expr="standard_type.isin(['EC50', 'IC50', 'Kd', 'Ki']) & standard_relation.isin(['>', '>='])"
    )

    bin_features = pd.concat([comments, inhibition, relations], axis=0)
    bin_features = bin_features[~bin_features.index.isin(pchembl_val.index)]
    bin_features["pchembl_value"] = cfg.params.px_placeholder
    bin_features["pchembl_value"] = (
        bin_features["pchembl_value"].groupby(bin_features.index).first()
    )

    df = pd.concat([pchembl_val, bin_features["pchembl_value"]], axis=0)

    return df


def write_training_data(
    df: pd.DataFrame,
    dir: str,
    train_data: str,
    random_seed: int,
) -> None:
    """Function that divides the training data based on chemical diversity, into the
    appropriate train/test/val sets and writes them to their respective files.

    Args:
        df: A DataFrame containing the cleaned ChEMBL data prepared for QSAR model training.
        train_data: Filepath to where the train data should be stored.
        random_seed: Number of random seed to ensure reproducibility of experiments.
        test_size: Fraction of data that should be held out for the test set.
        val_size: Fraction of data that should be held out for the validation set.
    """
    raise NotImplementedError


@hydra.main(config_path="../../config", config_name="qsar_data_config")
def main(cfg: QSARDataConfig):
    df = process_qsar_data(cfg=cfg)
    if cfg.params.classification:
        df = (df > cfg.params.px_thresh).astype("f4")

    df.to_csv(cfg.processed.path)

    write_training_data(
        df=df,
        dir=cfg.files.dir,
        train_data=cfg.files.train_data,
        random_seed=cfg.params.random_seed,
    )


if __name__ == "__main__":
    main()
