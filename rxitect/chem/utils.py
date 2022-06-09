import logging
from termios import N_SLIP
from typing import Union

import numpy as np
import rdkit.Chem
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from rxitect.structs.property import Property, calc_prop
from rxitect.utils.types import RDKitMol, List
from numpy.typing import ArrayLike

N_PHYSCHEM_PROPS = 19


def calc_single_fp(mol: Union[RDKitMol, str], radius: int = 3, bit_len: int=2048, accept_smiles: bool = True) -> ArrayLike:
    fp = calc_fp([mol], radius, bit_len, accept_smiles)[0]
    return fp


def calc_fp(mols: Union[List[RDKitMol], List[str]], radius: int = 3, bit_len: int = 2048, accept_smiles: bool = False) -> ArrayLike:
    if accept_smiles:
        mols = smiles_to_rdkit_mol(smiles_list=mols)

    ecfp = _calc_ecfp(mols, radius=radius, bit_len=bit_len)
    phch = _calc_physchem(mols)
    fps = np.concatenate([ecfp, phch], axis=1)
    return fps


def _calc_ecfp(mols: List[RDKitMol], radius: int = 3, bit_len: int = 2048) -> ArrayLike:
    fps = np.zeros((len(mols), bit_len))
    for i, mol in enumerate(
        tqdm(mols, desc="Calculating Extended-Connectivity Fingerprints")
    ):
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=bit_len
        )
        DataStructs.ConvertToNumpyArray(fp, fps[i, :])
    return fps


def _calc_physchem(mols: List[RDKitMol]) -> ArrayLike:
    prop_list = [
        Property.MolecularWeight,
        Property.LogP,
        Property.HBA,
        Property.HBD,
        Property.RotatableBonds,
        Property.AmideBonds,
        Property.BridgeheadAtoms,
        Property.HeteroAtoms,
        Property.HeavyAtoms,
        Property.SpiroAtoms,
        Property.FCSP3,
        Property.RingCount,
        Property.AliphaticRings,
        Property.AromaticRings,
        Property.SaturatedRings,
        Property.Heterocycles,
        Property.TPSA,
        Property.ValenceElectrons,
        Property.CrippenMolMR,
    ]
    assert len(prop_list) == N_PHYSCHEM_PROPS, f"Invalid number of properties: {len(prop_list)}, should be {N_PHYSCHEM_PROPS}"
    fps = np.zeros((len(mols), N_PHYSCHEM_PROPS))
    for i, prop in enumerate(
        tqdm(prop_list, desc="Calculating physico-chemical properties")
    ):
        fps[:, i] = calc_prop(mols=mols, prop=prop.value)
    return fps


def smiles_to_rdkit_mol(smiles_list: List[str]) -> List[RDKitMol]:
    """Helper function to convert a list of SMILES to RDKit Mol objects
    
    Args:
        smiles: List of SMILES representations of molecules
    
    Returns:
        A list of RDKit Mol objects created from the given SMILES
    """
    rdkit_mols = [Chem.MolFromSmiles(smi)
            for smi in tqdm(smiles_list, desc="Converting SMILES to Mol objects")]
    
    return rdkit_mols
