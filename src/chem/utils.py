import logging

import numpy as np
import rdkit.Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

from src.structs.property import Property, calc_prop
from src.utils.types import RDKitMol


def calc_fp(mols: RDKitMol, radius: int = 3, bit_len: int = 2048):
    ecfp = _calc_ecfp(mols, radius=radius, bit_len=bit_len)
    phch = _calc_physchem(mols)
    fps = np.concatenate([ecfp, phch], axis=1)
    return fps


def _calc_ecfp(mols, radius: int = 3, bit_len: int = 2048):
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


def _calc_physchem(mols):
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
