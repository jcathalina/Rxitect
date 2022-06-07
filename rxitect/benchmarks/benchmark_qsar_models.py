"""ABSOLUTELY NOT DONE YET."""

import errno
from datetime import datetime

import os
import logging
import hydra
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import pandas as pd
from sklearn.svm import SVR
from numpy.typing import ArrayLike
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from rdkit import Chem

from rxitect.chem.utils import calc_fp
from rxitect.data.utils import QSARModel
from rxitect.process_qsar_data import transform

logger = logging.getLogger(__name__)


def cross_validate_svr(dataset: pd.DataFrame, n_splits: int, random_state: int, out_dir: str, target: str):
    res = dataset.copy()
    X, y = calc_fp(dataset["smiles"]), dataset["pchembl_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    logger.info(">>> Scaling the train and test data")
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed = scaler.transform(X_test)

    logger.info(f">>> Splitting data into {n_splits} folds")
    cvs = np.zeros(y_train.shape)
    inds = np.zeros(y_test.shape)
    logger.info(f">>> Starting Grid Search")
    gs = GridSearchCV(SVR(), {'C': 2.0 ** np.array([-15, 15]), 'gamma': 2.0 ** np.array([-15, 15])}, n_jobs=-1)
    gs.fit(X_train_transformed, y_train)
    best_params = gs.best_params_
    logger.info(f"Best parameters found for SVR: {best_params}")

    folds = KFold(n_splits=n_splits).split(X)
    for train_index, test_index in tqdm(folds, desc="Looping through folds"):
        model = SVR(**best_params)
        model.fit(X[train_index], y[train_index], sample_weight=[1 if px_val >= 4 else 0.1 for px_val in y[train_index]])
        cvs[test_index] = model.predict(X[test_index])
        inds += model.predict(X_test_transformed)
    train_scores, test_scores = cvs, inds / 5
    
    logger.info(">>> Storing benchmark results")
    benchmark_df_train = pd.DataFrame(train_scores)
    benchmark_df_test = pd.DataFrame(test_scores)
    benchmark_df_train.to_csv(f"{out_dir}/{target}_TRAIN_{date_label()}.csv")
    benchmark_df_test.to_csv(f"{out_dir}/{target}_TEST_{date_label()}.csv")
    logger.info("*+.Done.+*")


def safe_mkdir(dir_path: str) -> None:
    """A helper function that allows you to cleanly create a directory if
    it does not exist yet.
    
    Args:
        dir_path: The path to the directory that would have to be created
    """
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def date_label() -> str:
    """Helper function that returns the formatted datetime as a convenient
    label for document name creation.
    
    Returns:
        The formatted datetime string
    """
    return datetime.now().strftime("%d_%m_%Y_%H%M%S")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Function to benchmark QSAR model(s)"""

    raw_path = abspath(cfg.qsar_dataset.raw.path)
    targets = cfg.qsar_dataset.targets
    cols = cfg.qsar_dataset.cols
    px_placeholder = cfg.qsar_dataset.px_placeholder
    random_state = cfg.random_state
    benchmark_dir = abspath(cfg.qsar_model.benchmark.dir)

    safe_mkdir(benchmark_dir)   

    for target in targets:
        df = transform(raw_path, [target], cols, px_placeholder)
        df = df.sample(frac=1, random_state=random_state)
        # mols = [
        #     Chem.MolFromSmiles(mol)
        #     for mol in tqdm(df.index, desc="Converting SMILES to Mol objects")
        # ]

        # X = calc_fp(mols=mols)
        # y = df.values

        if cfg.qsar_model.name == QSARModel.XGB:
            pass
        elif cfg.qsar_model.name == QSARModel.RF:
            pass
        elif cfg.qsar_model.name == QSARModel.SVR:
            cross_validate_svr(df, n_splits=3, random_state=random_state, out_dir=benchmark_dir, target=target)
        else:
            raise Exception(
                f"Chosen model '{cfg.qsar_model.name}' does not exist."
            )


if __name__ == "__main__":
    main()
