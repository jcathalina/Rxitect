from dataclasses import dataclass, astuple

import numpy as np

from globals import root_path, seed
from sklearn.model_selection import train_test_split
from rdkit import Chem

from qsar.data.transform_ligand_raw import ligand_dataset
from qsar.models.predictor import Predictor


@dataclass
class LigandDataset:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __iter__(self):
        return iter(astuple(self))


RAW_DATA_PATH = root_path / "data/raw"
TEST_SIZE = 0.2

data = ligand_dataset(filepath=RAW_DATA_PATH / "ligand_raw.tsv")
train_data, test_data = train_test_split(data, test_size=TEST_SIZE, random_state=seed)

X_train = Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in train_data.index])
y_train = train_data.values

X_test = Predictor.calc_fp([Chem.MolFromSmiles(mol) for mol in test_data.index])
y_test = test_data.values


def qsar_train_data():
    return LigandDataset(X_train, y_train, X_test, y_test)


def qsar_train_data_dev(dev_set_size: int = 100):
    return LigandDataset(X_train[:dev_set_size], y_train[:dev_set_size], X_test[:dev_set_size], y_test[:dev_set_size])
