from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Batch, Data

if TYPE_CHECKING:
    from rxitect.data.composable_molecule import ComposableMolecule
    from rxitect.mdp import MarkovDecisionProcess


def is_valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def filter_duplicate_tensors(x: torch.Tensor) -> torch.Tensor:
    return x.unique_consecutive(dim=0)


def mol_from_fragments(
    jbonds: ArrayLike,
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

    [
        rw_mol.AddBond(int(bond[0]), int(bond[1]), Chem.BondType.SINGLE)
        for bond in mol_bonds
    ]
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


def mol2graph(cmol: ComposableMolecule, mdp: MarkovDecisionProcess) -> Data:
    """
    TODO
    """
    long_tensor = lambda x: torch.tensor(x, dtype=torch.long, device=mdp.device)
    if len(cmol.block_idxs) == 0:
        data = Data(  # There's an extra block embedding for the empty molecule
            x=long_tensor([mdp.num_true_blocks]),
            edge_index=long_tensor([[], []]),
            edge_attr=long_tensor([]).reshape((0, 2)),
            stems=long_tensor([(0, 0)]),
            stem_types=long_tensor([mdp.num_stem_types]),
        )  # also extra stem type embedding
        return data
    edges = [(i[0], i[1]) for i in cmol.jbonds]
    # edge_attrs = [mdp.bond_type_offset[i[2]] +  i[3] for i in mol.jbonds]
    t = mdp.true_block_idx
    if 0:
        edge_attrs = [
            (
                (mdp.stem_type_offset[t[cmol.block_idxs[i[0]]]] + i[2])
                * mdp.num_stem_types
                + (mdp.stem_type_offset[t[cmol.block_idxs[i[1]]]] + i[3])
            )
            for i in cmol.jbonds
        ]
    else:
        edge_attrs = [
            (
                mdp.stem_type_offset[t[cmol.block_idxs[i[0]]]] + i[2],
                mdp.stem_type_offset[t[cmol.block_idxs[i[1]]]] + i[3],
            )
            for i in cmol.jbonds
        ]
    """
    Here stem_type_offset is a list of offsets to know which
    embedding to use for a particular stem. Each (blockidx, atom)
    pair has its own embedding.
    """
    stem_types = [
        mdp.stem_type_offset[t[cmol.block_idxs[i[0]]]] + i[1] for i in cmol.stems
    ]

    data = Data(
        x=long_tensor([t[i] for i in cmol.block_idxs]),
        edge_index=long_tensor(edges).T if len(edges) else long_tensor([[], []]),
        edge_attr=long_tensor(edge_attrs)
        if len(edges)
        else long_tensor([]).reshape((0, 2)),
        stems=long_tensor(cmol.stems) if len(cmol.stems) else long_tensor([(0, 0)]),
        stem_types=long_tensor(stem_types)
        if len(cmol.stems)
        else long_tensor([mdp.num_stem_types]),
    )
    data.to(mdp.device)
    return data


def mols2batch(mols: List[Data], mdp: MarkovDecisionProcess) -> Batch:
    """
    TODO
    """
    batch = Batch.from_data_list(mols, follow_batch=["stems"])
    batch.to(mdp.device)
    return batch
