"""ABSOLUTELY NOT DONE YET."""

from copy import deepcopy
from enum import Enum

import hydra
import joblib
import numpy as np
from tqdm import tqdm
import xgboost as xgb
from sklearn.svm import SVR
from numpy.typing import ArrayLike
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.base import BaseEstimator
from rdkit import Chem

import wandb
from rxitect.chem.utils import calc_fp
from rxitect.data.utils import LigandTrainingData
from rxitect.data.utils import QSARModel
from rxitect.process_qsar_data import transform


def cross_validate_svr(model: BaseEstimator, X: ArrayLike, y: ArrayLike):
    # X_train, X_test, y_train, y_test = train_test_split(X=X, y=y, test_size=0.2)
    # folds = KFold(5).split(X)

    # cvs = np.zeros(y.shape)
    # inds = np.zeros(y_ind.shape)
    # gs = GridSearchCV(deepcopy(alg), {'C': 2.0 ** np.array([-15, 15]), 'gamma': 2.0 ** np.array([-15, 15])}, n_jobs=10)
    # gs.fit(X, y)
    # params = gs.best_params_
    # print(params)
    # for i, (trained, valided) in enumerate(folds):
    #     model = deepcopy(alg)
    #     model.C = params['C']
    #     model.gamma = params['gamma']
    #     model.fit(X[trained], y[trained], sample_weight=[1 if v >= 4 else 0.1 for v in y[trained]])
    #     cvs[valided] = model.predict(X[valided])
    #     inds += model.predict(X_ind)
    # return cvs, inds / 5
    pass


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Function to benchmark QSAR model(s)"""

    wandb.init(project="benchmark-qsar-model")

    raw_path = abspath(cfg.qsar_dataset.raw.path)
    targets = cfg.qsar_dataset.targets
    cols = cfg.qsar_dataset.cols
    px_placeholder = cfg.qsar_dataset.px_placeholder
    random_state = cfg.random_state

    for target in targets:
        df = transform(raw_path, [target], cols, px_placeholder)
        df = df.sample(frac=1, random_state=random_state)
        mols = [
            Chem.MolFromSmiles(mol)
            for mol in tqdm(df["smiles"], desc="Converting SMILES to Mol objects")
        ]

        model = None
        if cfg.qsar_model.name == QSARModel.XGB:
            model = xgb.XGBRegressor(**cfg.qsar_model.params)
        elif cfg.qsar_model.name == QSARModel.RF:
            model = RandomForestRegressor(**cfg.qsar_model.params)
        elif cfg.qsar_model.name == QSARModel.SVM:
            model = SVR(**cfg.qsar_model.params)
        else:
            raise Exception(
                f"Chosen model '{cfg.qsar_model.name}' does not exist."
            )

        X = calc_fp(mols=mols)
        y = df["pchembl_value"]


        model.fit(
            X=X_train,
            y=y_train,
            sample_weight=[1.0 if px_val >= 4 else 0.1 for px_val in y_train],
        )
    # joblib.dump(model, output_path)

    wandb.sklearn.plot_regressor(
        model, X_train, X_test, y_train, y_test, model_name=cfg.qsar_model.name
    )
    score = model.score(
        X=X_test,
        y=y_test,
        sample_weight=[1.0 if px_val >= 4 else 0.1 for px_val in y_test],
    )
    print(f"score: {score}")
    wandb.finish()


if __name__ == "__main__":
    main()
