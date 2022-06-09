from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.model_selection import train_test_split

from rxitect.chem.utils import calc_fp
from rxitect.utils.types import ArrayDict


@dataclass
class QSARDataset:
    """Class representing the dataset used to train QSAR models"""
    df_test: pd.DataFrame
    df_train: pd.DataFrame
    targets: List[str]
    _X_train: ArrayDict = field(init=False)
    _X_test: ArrayDict = field(init=False)
    _y_train: ArrayDict = field(init=False)
    _y_test: ArrayDict = field(init=False)
    
    def __post_init__(self) -> None:
        """Initializes the train and test data based on a temporal split in the data to be used for QSAR model fitting."""
        self._X_train = {k: np.array([]) for k in self.targets}
        self._X_test = {k: np.array([]) for k in self.targets}
        self._y_train = {k: np.array([]) for k in self.targets}
        self._y_test = {k: np.array([]) for k in self.targets}

    def get_train_test_data(self, target_chembl_id: str) -> Tuple[np.ndarray, ...]:
        """
        """
        return (self.X_train(target_chembl_id),
                self.y_train(target_chembl_id),
                self.X_test(target_chembl_id),
                self.y_test(target_chembl_id))

    def X_train(self, target_chembl_id: str) -> np.ndarray:
        """Lazily evaluates the train data points for a given target ChEMBL ID
        
        Args:
            target_chembl_id:
        
        Returns:
            An array containing the fingerprints of all train data points for the given target ChEMBL ID
        """
        if not self._X_train[target_chembl_id].size:
            data = self.df_train[target_chembl_id].dropna().index
            self._X_train[target_chembl_id] = calc_fp(data, accept_smiles=True)
        return self._X_train[target_chembl_id]
    
    def X_test(self, target_chembl_id: str) -> np.ndarray:
        """Lazily evaluates the test data points for a given target ChEMBL ID
        
        Args:
            target_chembl_id:
        
        Returns:
            An array containing the fingerprints of all test data points for the given target ChEMBL ID
        """
        if not self._X_test[target_chembl_id].size:
            data = self.df_test[target_chembl_id].dropna().index
            self._X_test[target_chembl_id] = calc_fp(data, accept_smiles=True)
        return self._X_test[target_chembl_id]
    
    def y_train(self, target_chembl_id: str) -> np.ndarray:
        """Lazily evaluates the train labels for a given target ChEMBL ID
        
        Args:
            target_chembl_id:
        
        Returns:
            An array containing the pChEMBL value of all train data points for the given target ChEMBL ID
        """
        if not self._y_train[target_chembl_id].size:
            data = self.df_train[target_chembl_id].dropna().values
            self._y_train[target_chembl_id] = data
        return self._y_train[target_chembl_id]
    
    def y_test(self, target_chembl_id: str) -> np.ndarray:
        """Lazily evaluates the test labels for a given target ChEMBL ID
        
        Args:
            target_chembl_id:
        
        Returns:
            An array containing the pChEMBL value of all test data points for the given target ChEMBL ID
        """
        if not self._y_test[target_chembl_id].size:
            data = self.df_test[target_chembl_id].dropna().values
            self._y_test[target_chembl_id] = data
        return self._y_test[target_chembl_id]

    @classmethod
    def load_from_file(cls, train_file: str, test_file: str) -> QSARDataset:
        """
        """
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file)
        targets = [target for target in df_train.columns[1:].values]  # TODO: Assert that targets are equal for train and test

        return QSARDataset(df_train, df_test, targets)


class QSARModel(str, Enum):
    XGB = "xgboost"
    RF = "random_forest"
    SVR = "svr"
    

def train_test_val_split(
    X: ArrayLike,
    y: ArrayLike,
    train_size: float,
    test_size: float,
    val_size: float,
    random_state: int = 42,
) -> Tuple[ArrayLike, ...]:
    """Helper function that extends scikit-learn's train test split to also accomodate validation set creation.

    Args:
        X: Array-like containing the full dataset without labels
        y: Array-like containing all the dataset labels
        train_size: The fraction of the data that should be reserved for training
        test_size: The fraction of the data that should be reserved for testing
        val_size: The fraction of the data that should be reserved for validation
        random_state: The random seed number used to enforce reproducibility

    Returns:
        A tuple containing all the train/test/val data in the following order:
        X_train, X_test, X_val, y_train, y_test, y_val
    """
    assert train_size + test_size + val_size == 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
    )

    return X_train, X_test, X_val, y_train, y_test, y_val
