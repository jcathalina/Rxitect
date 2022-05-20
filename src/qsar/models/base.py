from abc import ABC, abstractmethod

import numpy as np


class QSARModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> None:
        pass
