"""ABSOLUTELY NOT DONE YET."""

import errno
from datetime import datetime

import os
import logging
from typing import List, Union
import hydra
import joblib
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
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from rdkit import Chem

from rxitect.chem.utils import calc_fp
from rxitect.data.utils import QSARDataset, QSARModel
from rxitect.process_qsar_data import construct_qsar_dataset

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


def kfold_cv_benchmark(model: Union[RandomForestRegressor, xgb.XGBRegressor],
                        dataset: QSARDataset,
                        k: int,
                        out_dir: str,
                        target: str,
                        scoring: List[str] = ['r2', 'neg_root_mean_squared_error'],
                        n_jobs: int = -1) -> None:
    """
    """
    X_train = dataset.X_train(target)
    X_test = dataset.X_test(target)
    y_train = dataset.y_train(target)
    y_test = dataset.y_test(target)

    model.fit(X=X_train, y=y_train, sample_weight=[1 if px_val >= 4 else 0.1 for px_val in y_train])
    cv_result = cross_validate(model, X_train, y_train, cv=k, scoring=scoring,
                               return_train_score=True, return_estimator=True, n_jobs=n_jobs)

    joblib.dump(cv_result, filename=f"{out_dir}/cv_{k}_fold_results_{target}_{date_label()}.pkl")


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

    dataset = QSARDataset.load_from_file()

    for target in targets:
        if cfg.qsar_model.name == QSARModel.XGB:
            model = xgb.XGBRegressor(**cfg.qsar_model.params)
        elif cfg.qsar_model.name == QSARModel.RF:
            model = RandomForestRegressor(n_estimators=1_000, max_features="sqrt")
        else:
            raise Exception(
                f"Chosen model '{cfg.qsar_model.name}' does not exist."
            )
        if model:
            kfold_cv_benchmark(model, dataset=dataset, k=5, out_dir=benchmark_dir,
                                    target=target)


if __name__ == "__main__":
    main()
