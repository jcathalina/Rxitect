from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from hydra.utils import to_absolute_path as abspath
from numpy.typing import ArrayLike

from rxitect.chem.utils import calc_fp
from rxitect.utils.types import ArrayDict


@dataclass
class SingleTargetQSARDataset:
    """Class representing the dataset used to train QSAR models for single ChEMBL targets"""

    df_train: pd.DataFrame
    df_test: pd.DataFrame
    target: str
    _X_train: np.ndarray = np.array([])
    _X_test: np.ndarray = np.array([])
    _y_train: np.ndarray = np.array([])
    _y_test: np.ndarray = np.array([])

    def get_train_test_data(self) -> Tuple[np.ndarray, ...]:
        """
        """
        return (
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
        )

    @property
    def X_train(self) -> np.ndarray:
        """Lazily evaluates the train data points for a given target ChEMBL ID

        Returns:
            An array containing the fingerprints of all train data points for the given target ChEMBL ID
        """
        if not self._X_train.size:
            data = self.df_train["smiles"]
            self._X_train = calc_fp(data, accept_smiles=True)
        return self._X_train

    @property
    def X_test(self) -> np.ndarray:
        """Lazily evaluates the test data points for a given target ChEMBL ID

        Returns:
            An array containing the fingerprints of all test data points for the given target ChEMBL ID
        """
        if not self._X_test.size:
            data = self.df_test["smiles"]
            self._X_test = calc_fp(data, accept_smiles=True)
        return self._X_test

    @property
    def y_train(self) -> np.ndarray:
        """Lazily evaluates the train labels for a given target ChEMBL ID

        Returns:
            An array containing the pChEMBL value of all train data points for the given target ChEMBL ID
        """
        if not self._y_train.size:
            data = self.df_train["pchembl_value"]
            self._y_train = data
        return self._y_train
    
    @property
    def y_test(self) -> np.ndarray:
        """Lazily evaluates the test labels for a given target ChEMBL ID

        Returns:
            An array containing the pChEMBL value of all test data points for the given target ChEMBL ID
        """
        if not self._y_test.size:
            data = self.df_test["pchembl_value"]
            self._y_test = data
        return self._y_test

    def get_classifier_labels(
        self, target_chembl_id: str
    ) -> Tuple[ArrayLike, ArrayLike]:
        """ """
        y_train_clf = np.where(
            self.y_train > 6.5, 1, 0
        )  # TODO: Make 6.5 thresh a const
        y_test_clf = np.where(self.y_test(target_chembl_id) > 6.5, 1, 0)

        return y_train_clf, y_test_clf

    @classmethod
    def load_from_file(cls, train_file: str, test_file: str, target: Optional[str] = None) -> SingleTargetQSARDataset:
        """ """
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        target = target if target else f"Loaded from files: TRAIN='{df_train}' --- TEST='{df_test}'"

        return SingleTargetQSARDataset(df_train, df_test, target)
