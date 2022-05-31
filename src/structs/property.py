import numpy as np
from rdkit.Chem import (
    AllChem,
    Crippen,
    Descriptors,
    Lipinski,
    RDConfig,
)
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.QED import qed
from typing import TYPE_CHECKING

# special snippet to import SA Score, from https://github.com/rdkit/rdkit/issues/2279
import os
import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore

if TYPE_CHECKING:
    from src.utils.types import RDKitMol, List

# TODO: Make prop dict keys use enums
prop_dict = {
    "MW": Descriptors.MolWt,
    "logP": Crippen.MolLogP,
    "HBA": AllChem.CalcNumLipinskiHBA,
    "HBD": AllChem.CalcNumLipinskiHBD,
    "Rotable": AllChem.CalcNumRotatableBonds,
    "Amide": AllChem.CalcNumAmideBonds,
    "Bridge": AllChem.CalcNumBridgeheadAtoms,
    "Hetero": AllChem.CalcNumHeteroatoms,
    "Heavy": Lipinski.HeavyAtomCount,
    "Spiro": AllChem.CalcNumSpiroAtoms,
    "FCSP3": AllChem.CalcFractionCSP3,
    "Ring": Lipinski.RingCount,
    "Aliphatic": AllChem.CalcNumAliphaticRings,
    "Aromatic": AllChem.CalcNumAromaticRings,
    "Saturated": AllChem.CalcNumSaturatedRings,
    "HeteroR": AllChem.CalcNumHeterocycles,
    "TPSA": AllChem.CalcTPSA,
    "Valence": Descriptors.NumValenceElectrons,
    "MR": Crippen.MolMR,
    "QED": qed,
    "SA": sascorer.calculateScore,
    "Bertz": BertzCT,
}


def calculate(mols: List[RDKitMol], prop: str) -> np.ndarray:
    """Calculates the value of a molecular property for a batch of molecules.

    Args:
        mols: A list of RDKit Mol objects.
        prop: The dictionary key for the property to be calculated.

    Returns:
        an array of scores for each molecule given.
    """
    scores = np.zeros(len(mols))
    for i, mol in enumerate(mols):
        try:
            scores[i] = prop_dict.get(prop)(mol)
        except Exception as e:  # NOTE: Can be handled.
            continue
    return scores
