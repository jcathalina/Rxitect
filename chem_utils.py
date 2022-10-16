# TODO: rename this to chem_utils, or utils.chem namespace? idk.
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit import RDConfig
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType
from numpy.typing import ArrayLike, NDArray

rdBase.DisableLog("rdApp.error")

import pandas as pd

import torch
from torch_geometric.data import Data
from torch_sparse import coalesce

atomic_numbers = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
}


def onehot(arr: ArrayLike, num_classes: int, dtype=np.int8) -> NDArray:
    """Converts an array to its one-hot representation

    Args:
        arr (ArrayLike): An array-like data structure, such as a list
        num_classes (int): The number of classes that should be accounted for in the one-hot array
        dtype: The NumPy datatype that the array should store. Defaults to 64-bit integer

    Returns:
        NDArray: The one-hot representation of the given array
    """
    arr = np.asarray(arr, dtype=dtype)
    assert len(arr.shape) == 1, "dims other than 1 not implemented"
    onehot_arr = np.zeros(arr.shape + (num_classes,), dtype=dtype)
    onehot_arr[np.arange(arr.shape[0]), arr] = 1
    return onehot_arr


def mol_from_fragments(
        jbonds: List[List[int]],
        frags: Optional[List[Chem.rdchem.Mol]] = None,
        frag_smiles: Optional[List[str]] = None,
        optimize: bool = False,
) -> Tuple[Chem.rdchem.Mol, List[int]]:
    """Joins 2 or more fragments into a single molecule

    Args:
        jbonds (ArrayLike): An array-like (e.g., a list) object containing junction bonds
        frags (Optional[List[Mol]]): A list of RDKit Mol objects to be combined. Should be given if frag_smiles is not present
        frag_smiles (Optional[List[Mol]]): A list of SMILES strings to be made into RDKit Mol objects and combined. Must be present if frags is not
        optimize (bool): If the molecule's 3D structure should be optimized. Defaults to False

    Returns:
        Tuple[Mol, List[int]]: A tuple containing the combined molecule as an RDKit Mol object, and a list containing the bonds
    """
    jbonds = np.asarray(jbonds)

    if frags is not None:
        pass
    elif frags is None and frag_smiles is not None:
        frags = [Chem.MolFromSmiles(smi) for smi in frag_smiles]
    else:
        raise ValueError("At least one of `frags` or `frag_smiles` should be given.")

    if len(frags) == 0:
        return None, None

    num_frags = len(frags)
    # combine fragments into a single molecule
    mol = frags[0]
    for i in np.arange(start=1, stop=num_frags):
        mol = Chem.CombineMols(mol, frags[i])
    # add junction bonds between fragments
    frag_start_idx = np.concatenate(
        [[0], np.cumsum([frag.GetNumAtoms() for frag in frags])], 0
    )[:-1]

    if jbonds.size == 0:
        mol_bonds = []
    else:
        mol_bonds = frag_start_idx[jbonds[:, 0:2]] + jbonds[:, 2:4]

    rw_mol = Chem.EditableMol(mol)

    [rw_mol.AddBond(int(bond[0]), int(bond[1]), Chem.BondType.SINGLE) for bond in mol_bonds]
    mol = rw_mol.GetMol()
    atoms = list(mol.GetAtoms())

    def _pop_H(atom):
        num_h = atom.GetNumExplicitHs()
        if num_h > 0:
            atom.SetNumExplicitHs(num_h - 1)

    [(_pop_H(atoms[bond[0]]), _pop_H(atoms[bond[1]])) for bond in mol_bonds]
    Chem.SanitizeMol(mol)

    # create and optimize 3D structure
    if optimize:
        assert not "h" in set(
            [atom.GetSymbol().lower() for atom in mol.GetAtoms()]
        ), "can't optimize molecule with h"
        Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        Chem.RemoveHs(mol)
    return mol, mol_bonds


_mpnn_feat_cache = [None]


def mpnn_feat(
        mol: Chem.rdchem.Mol,
        get_coord: bool = True,
        pandas_fmt: bool = False,
        onehot_atom: bool = False,
        donor_features: bool = False,
) -> Tuple[NDArray, Optional[NDArray], NDArray, NDArray]:
    """Calculates molecule features to be passed to the MPNN

    Args:
        mol (Mol): RDKit Mol representing a molecule
        get_coord (bool): If coordinates should be calculated and returned. Defaults to True
        pandas_fmt (bool): If the features should be returned in a pandas-friendly format. Defaults to False
        onehot_atom (bool): If the atom features should be returned as onehot vectors. Defaults to False
        donor_features (bool): If donor/acceptor features should be calculated and added to the atom features. Defaults to False
    """
    atom_types = {"H": 0, "C": 1, "N": 2, "O": 3, "F": 4}
    bond_types = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

    num_atoms = len(mol.GetAtoms())
    num_atom_types = len(atom_types)
    # featurize elements
    # columns are: ["type_idx" .. , "atomic_number", "acceptor", "donor",
    # "aromatic", "sp", "sp2", "sp3", "num_hs", [atomic_number_onehot] .. ])

    num_feats = num_atom_types + 1 + 8
    if onehot_atom:
        num_feats += len(atomic_numbers)
    atom_feat = np.zeros((num_atoms, num_feats))

    # featurize
    for i, atom in enumerate(mol.GetAtoms()):
        type_idx = atom_types.get(atom.GetSymbol(), 5)
        atom_feat[i, type_idx] = 1
        if onehot_atom:
            atom_feat[i, num_atom_types + 9 + atom.GetAtomicNum() - 1] = 1
        else:
            atom_feat[i, num_atom_types + 1] = (atom.GetAtomicNum() % 16) / 2.0
        atom_feat[i, num_atom_types + 4] = atom.GetIsAromatic()
        hybridization = atom.GetHybridization()
        atom_feat[i, num_atom_types + 5] = hybridization == HybridizationType.SP
        atom_feat[i, num_atom_types + 6] = hybridization == HybridizationType.SP2
        atom_feat[i, num_atom_types + 7] = hybridization == HybridizationType.SP3
        atom_feat[i, num_atom_types + 8] = atom.GetTotalNumHs(includeNeighbors=True)

    # get donors and acceptors
    if donor_features:
        if _mpnn_feat_cache[0] is None:
            fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
            factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
            _mpnn_feat_cache[0] = factory
        else:
            factory = _mpnn_feat_cache[0]

        feats = factory.GetFeaturesForMol(mol)
        for j in range(0, len(feats)):
            if feats[j].GetFamily() == "Donor":
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    atom_feat[k, num_atom_types + 3] = 1
            elif feats[j].GetFamily() == "Acceptor":
                node_list = feats[j].GetAtomIds()
                for k in node_list:
                    atom_feat[k, num_atom_types + 2] = 1
    # get coord
    if get_coord:
        coord = np.asarray([mol.GetConformer(0).GetAtomPosition(j) for j in range(num_atoms)])
    else:
        coord = None
    # get bonds and bond features
    bond = np.asarray(
        [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] for bond in mol.GetBonds()]
    )
    bond_feat = [bond_types[bond.GetBondType()] for bond in mol.GetBonds()]
    bond_feat = onehot(bond_feat, num_classes=len(bond_types))

    # convert atom_feat to pandas
    if pandas_fmt:
        atom_feat_pd = pd.DataFrame(
            index=range(num_atoms),
            columns=[
                "type_idx",
                "atomic_number",
                "acceptor",
                "donor",
                "aromatic",
                "sp",
                "sp2",
                "sp3",
                "num_hs",
            ],
        )
        atom_feat_pd["type_idx"] = atom_feat[:, : num_atom_types + 1]
        atom_feat_pd["atomic_number"] = atom_feat[:, num_atom_types + 1]
        atom_feat_pd["acceptor"] = atom_feat[:, num_atom_types + 2]
        atom_feat_pd["donor"] = atom_feat[:, num_atom_types + 3]
        atom_feat_pd["aromatic"] = atom_feat[:, num_atom_types + 4]
        atom_feat_pd["sp"] = atom_feat[:, num_atom_types + 5]
        atom_feat_pd["sp2"] = atom_feat[:, num_atom_types + 6]
        atom_feat_pd["sp2"] = atom_feat[:, num_atom_types + 7]
        atom_feat_pd["sp3"] = atom_feat[:, num_atom_types + 8]
        atom_feat = atom_feat_pd
    return atom_feat, coord, bond, bond_feat


def mol_to_pyg(
        atom_feat: NDArray,
        coord: Optional[NDArray],
        bond: NDArray,
        bond_feat: NDArray,
        props: Dict[str, Any] = {},
) -> Data:
    """Convert molecule data to PyTorch Geometric module format

    Args:
        atom_feat (NDArray): Array containing atom features
        coord (Optional[NDArray]): Array containing coordinates. Defaults to None
        bond (NDArray): Array containing bonds
        bond_feat (NDArray): Array containing bond features

    Returns:
        Data: The molecule data in PyTorch Geometric format
    """
    num_atoms = atom_feat.shape[0]
    # transform to torch_geometric bond format; send edges both ways; sort bonds
    atom_feat = torch.tensor(atom_feat, dtype=torch.float32)
    if bond.shape[0] > 0:
        edge_index = torch.tensor(
            np.concatenate([bond.T, np.flipud(bond.T)], axis=1), dtype=torch.int64
        )
        edge_attr = torch.tensor(
            np.concatenate([bond_feat, bond_feat], axis=0), dtype=torch.float32
        )
        edge_index, edge_attr = coalesce(edge_index, edge_attr, num_atoms, num_atoms)
    else:
        edge_index = torch.zeros((0, 2), dtype=torch.int64)
        edge_attr = torch.tensor(bond_feat, dtype=torch.float32)

    # make torch data
    if coord is not None:
        coord = torch.tensor(coord, dtype=torch.float32)
        data = Data(x=atom_feat, pos=coord, edge_index=edge_index, edge_attr=edge_attr, **props)
    else:
        data = Data(x=atom_feat, edge_index=edge_index, edge_attr=edge_attr, **props)
    return data
