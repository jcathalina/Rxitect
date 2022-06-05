from enum import Enum

import hydra
import joblib
import xgboost as xgb
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import wandb
from rxitect.data.utils import LigandTrainingData


class QSARModel(str, Enum):
    XGB = "xgboost"
    RF = "random_forest"


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Function to train the model"""

    wandb.init(project="train-qsar-model")

    dataset_path = abspath(cfg.qsar_dataset.files.train_data)
    output_path = abspath(cfg.model_path)

    # TODO: Make log
    print(f"Train modeling using {dataset_path}")
    print(f"Model used: {cfg.qsar_model.name}")
    print(f"Model params: {cfg.qsar_model.params}")
    print(
        f"Save the output to {output_path}"
    )  # TODO: specify if reg/clf in filename?

    dataset = joblib.load(dataset_path)
    X, y = dataset.X, dataset.y.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.random_state
    )

    model = None
    if cfg.qsar_model.name == QSARModel.XGB:
        model = xgb.XGBRegressor(**cfg.qsar_model.params)
    elif cfg.qsar_model.name == QSARModel.RF:
        model = RandomForestRegressor(**cfg.qsar_model.params)
    else:
        raise Exception

    model.fit(
        X=X_train,
        y=y_train,
        sample_weight=[1.0 if px_val >= 4 else 0.1 for px_val in y_train],
    )
    joblib.dump(model, output_path)

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
