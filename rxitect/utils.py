import torch

from rdkit import Chem
from rdkit.Chem import AllChem


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def filter_duplicate_tensors(x: torch.Tensor) -> torch.Tensor:
    return x.unique_consecutive(dim=0)
