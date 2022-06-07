import logging

import numpy as np
import rdkit.Chem
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from rxitect.structs.property import Property, calc_prop
from rxitect.utils.types import RDKitMol, List


def calc_fp(mols: List[RDKitMol], radius: int = 3, bit_len: int = 2048):
    ecfp = _calc_ecfp(mols, radius=radius, bit_len=bit_len)
    phch = _calc_physchem(mols)
    fps = np.concatenate([ecfp, phch], axis=1)
    return fps


def _calc_ecfp(mols: List[RDKitMol], radius: int = 3, bit_len: int = 2048):
    fps = np.zeros((len(mols), bit_len))
    for i, mol in enumerate(
        tqdm(mols, desc="Calculating Extended-Connectivity Fingerprints")
    ):
        try:
            if isinstance(mol, str):
                # Make Mol object before calculating morgan FP or else it breaks.
                mol = rdkit.Chem.MolFromSmiles(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=radius, nBits=bit_len
            )
            DataStructs.ConvertToNumpyArray(fp, fps[i, :])
        except Exception as e:
            logging.error(
                f"Something went wrong while creating fingerprints: {e}"
            )
    return fps


def _calc_physchem(mols: List[RDKitMol]):
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
    fps = np.zeros((len(mols), 19))
    for i, prop in enumerate(
        tqdm(prop_list, desc="Calculating physico-chemical properties")
    ):
        fps[:, i] = calc_prop(mols=mols, prop=prop)
    return fps


def smiles_to_rdkit_mol(smiles: List[str]) -> List[RDKitMol]:
    """Helper function to convert a list of SMILES to RDKit Mol objects
    
    Args:
        smiles: List of SMILES representations of molecules
    
    Returns:
        A list of RDKit Mol objects created from the given SMILES
    """
    rdkit_mols = [Chem.MolFromSmiles(smi)
            for smi in tqdm(smiles, desc="Converting SMILES to Mol objects")]
    
    return rdkit_mols