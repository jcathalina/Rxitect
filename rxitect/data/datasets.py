import pandas as pd
import torch
import numpy as np

from typing import Callable, List, Tuple
from torch.utils.data import Dataset, DataLoader, random_split
from hydra.utils import to_absolute_path as abspath
from numpy.typing import ArrayLike
from rdkit import Chem

from rxitect.chem.utils import calc_single_fp, smiles_to_rdkit_mol

def smiles_to_fingerprint(smiles: str) -> ArrayLike:
    """
    Helper function that transforms SMILES strings into
    the enhanced 2067D-Fingerprint representation used for training in Rxitect.
    If only a single SMILES was passed, will return a single array containing
    its fingerprint.

    Args:
        smiles: A list of SMILES representations of molecules.
    """
    fingerprint = np.array([])
    try:
        rdkit_mol = Chem.MolFromSmiles(smiles[0])
        fingerprint = calc_single_fp(rdkit_mol)
    except TypeError:
        fingerprint = smiles_to_rdkit_mol(smiles_list=smiles)

    if not fingerprint.size:
        raise Exception
    return fingerprint

class PyTorchQSARDataset(Dataset):
    def __init__(self, ligand_file: str, target_chembl_id: str, transform: Callable = None) -> None:
        self.pchembl_values: pd.DataFrame = pd.read_csv(ligand_file, usecols=["smiles", target_chembl_id]).dropna().reset_index(drop=True)
        self.transform: Callable = transform

    def __len__(self) -> int:
        return len(self.pchembl_values)

    def __getitem__(self, index) -> Tuple[ArrayLike, float]:
        smiles = self.pchembl_values.iloc[index, 0]
        print(smiles)
        pchembl_value = self.pchembl_values.iloc[index, 1]
        if self.transform:
            smiles = self.transform(smiles)
        return smiles, pchembl_value


if __name__ == "__main__":
    test_data = PyTorchQSARDataset(ligand_file=abspath("data/processed/ligand_test_splityear=2015.csv"),
                                            target_chembl_id="CHEMBL226",
                                            transform=smiles_to_fingerprint)

    train_data = PyTorchQSARDataset(ligand_file=abspath("data/processed/ligand_train_splityear=2015.csv"),
                                            target_chembl_id="CHEMBL226",
                                            transform=smiles_to_fingerprint)
    
    X_train, y_train = train_data[:]
    print(X_train.shape)