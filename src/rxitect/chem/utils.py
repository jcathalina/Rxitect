from typing import Dict, Iterable

import numpy as np
import selfies as sf
from numpy.typing import ArrayLike
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from rxitect.structs.property import Property, batch_calc_prop, calc_prop
from rxitect.utils.types import List, RDKitMol

N_PHYSCHEM_PROPS = 19
PROP_LIST = [
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


def calc_fp(
    mol: RDKitMol,
    radius: int = 3,
    bit_len: int = 2048,
) -> np.ndarray:
    ecfp = calc_ecfp(mol, radius, bit_len)
    phch = calc_physchem(mol)
    fp = np.concatenate([ecfp, phch])
    return fp


def calc_ecfp(mol: RDKitMol, radius: int = 3, bit_len: int = 2048) -> np.ndarray:
    fp = np.zeros(shape=(bit_len))
    _fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bit_len)
    DataStructs.ConvertToNumpyArray(_fp, fp)
    return fp


def calc_physchem(mol: RDKitMol, prop_list: List[str] = PROP_LIST):
    assert (
        len(prop_list) == N_PHYSCHEM_PROPS
    ), f"Invalid number of properties: {len(prop_list)}, should be {N_PHYSCHEM_PROPS}"
    fp = np.zeros(shape=(N_PHYSCHEM_PROPS))
    for i, prop in enumerate(prop_list):
        fp[i] = calc_prop(mol=mol, prop=prop.value)
    return fp


def batch_calc_fp(
    mols: Iterable[RDKitMol],
    radius: int = 3,
    bit_len: int = 2048,
) -> np.ndarray:
    ecfp = batch_calc_ecfp(mols, radius, bit_len)
    phch = batch_calc_physchem(mols)
    fingerprints = np.concatenate([ecfp, phch], axis=1)
    return fingerprints


def batch_calc_ecfp(
    mols: Iterable[RDKitMol], radius: int = 3, bit_len: int = 2048
) -> np.ndarray:
    fingerprints = np.zeros((len(mols), bit_len))
    for i, mol in enumerate(mols):
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bit_len)
        DataStructs.ConvertToNumpyArray(fp, fingerprints[i, :])
    return fingerprints


def batch_calc_physchem(
    mols: Iterable[RDKitMol], prop_list: List[str] = PROP_LIST
) -> ArrayLike:
    assert (
        len(prop_list) == N_PHYSCHEM_PROPS
    ), f"Invalid number of properties: {len(prop_list)}, should be {N_PHYSCHEM_PROPS}"
    fingerprints = np.zeros((len(mols), N_PHYSCHEM_PROPS))
    for i, prop in enumerate(prop_list):
        fingerprints[:, i] = batch_calc_prop(mols=mols, prop=prop.value)
    return fingerprints


def batch_mol_from_smiles(smiles_list: List[str]) -> List[RDKitMol]:
    """Helper function to convert a list of SMILES to RDKit Mol objects

    Args:
        smiles: List of SMILES representations of molecules

    Returns:
        A list of RDKit Mol objects created from the given SMILES
    """
    rdkit_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    return rdkit_mols


def mol_from_selfies(
    selfies: str, sanitize: bool = True, replacements: Dict[str, str] = {}
) -> RDKitMol:
    smiles = sf.decoder(selfies)
    rdkit_mol = Chem.MolFromSmiles(smiles, sanitize, replacements)
    return rdkit_mol


def mol_to_selfies(
    mol: RDKitMol,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    rooted_at_atom: int = -1,
    canonical: bool = True,
    all_bonds_explicit: bool = False,
    all_Hs_explicit: bool = False,
    do_random: bool = False,
) -> str:
    """Helper function that wraps around Chem.MolToSmiles adapted for SELFIES"""
    smiles = Chem.MolToSmiles(
        mol,
        isomeric_smiles,
        kekule_smiles,
        rooted_at_atom,
        canonical,
        all_bonds_explicit,
        all_Hs_explicit,
        do_random,
    )
    selfies = sf.encoder(smiles)
    return selfies


def randomize_selfies(selfies: str, isomeric_smiles: bool = False) -> str:
    """Perform a randomization of a SELFIES string
    must be RDKit sanitizable"""

    mol = mol_from_selfies(selfies)
    nmol = randomize_mol(mol)
    selfies = mol_to_selfies(mol=nmol, canonical=False, isomeric_smiles=isomeric_smiles)
    return selfies


def randomize_mol(mol: RDKitMol) -> RDKitMol:
    """Performs a randomization of the atom order of an RDKit molecule"""
    ans = list(range(mol.GetNumAtoms()))
    np.random.shuffle(ans)
    return Chem.RenumberAtoms(mol, ans)
