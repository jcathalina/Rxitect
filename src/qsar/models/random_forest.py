import joblib
import mlflow
import numpy as np

from qsar.data.train_data import qsar_train_data, qsar_train_data_dev
from qsar.models.base import QSARModel
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from globals import root_path

from typing import Dict, Any


MODEL_PATH = root_path / "models/qsar"


class RandomForest(QSARModel):
    def __init__(self, configuration: Dict[str, Any]):
        super().__init__()
        self.model = RandomForestRegressor(**configuration)

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(X=X_train, y=y_train,
                       sample_weight=[1 if val >= 4 else 0.1 for val in y_train])
        joblib.dump(value=self.model, filename=MODEL_PATH / f"qsar_rf_{get_date_label()}.pkg")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)


def get_date_label() -> str:
    date_label = datetime.now().strftime(fmt="%H_%M_%-d%b%-y")
    return date_label


if __name__ == "__main__":
    mlflow.set_tracking_uri("https://dagshub.com/naisuu/Rxitect.mlflow")
    mlflow.sklearn.autolog()

    train_data = qsar_train_data_dev(100)
    X_train, y_train, X_test, y_test = train_data

    conf = {
        "n_estimators": 1_000,
        "n_jobs": -1,
    }
    rf_qsar = RandomForest(conf)

    with mlflow.start_run() as run:
        rf_qsar.train(X_train, y_train)
        y_pred = rf_qsar.predict(X_test)
    print(rf_qsar.__dict__)
