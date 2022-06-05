import os
import sys
from enum import Enum

import numpy as np
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, RDConfig
from rdkit.Chem.GraphDescriptors import BertzCT
from rdkit.Chem.QED import qed

from rxitect.utils.types import List, RDKitMol

# special snippet to import SA Score, from https://github.com/rdkit/rdkit/issues/2279
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # type: ignore


class Property(Enum):
    MolecularWeight = "MW"
    LogP = "logP"
    HBA = "HBA"
    HBD = "HBD"
    RotatableBonds = "Rotatable"
    AmideBonds = "Amide"
    BridgeheadAtoms = "Bridge"
    HeteroAtoms = "Hetero"
    HeavyAtoms = "Heavy"
    SpiroAtoms = "Spiro"
    FCSP3 = "FCSP3"
    RingCount = "Ring"
    AliphaticRings = "Aliphatic"
    AromaticRings = "Aromatic"
    SaturatedRings = "Saturated"
    Heterocycles = "HeteroR"
    TPSA = "TPSA"
    ValenceElectrons = "Valence"
    CrippenMolMR = "MR"
    QED = "QED"
    SyntheticAccessibility = "SA"
    BertzComplexity = "Bertz"


prop_dict = {
    "MW": Descriptors.MolWt,
    "logP": Crippen.MolLogP,
    "HBA": AllChem.CalcNumLipinskiHBA,
    "HBD": AllChem.CalcNumLipinskiHBD,
    "Rotatable": AllChem.CalcNumRotatableBonds,
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


def calc_prop(mols: List[RDKitMol], prop: str) -> np.ndarray:
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
