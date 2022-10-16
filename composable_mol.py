from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from rdkit.Chem import Mol
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from rdkit import Chem

import chem_utils
from numpy.typing import NDArray, ArrayLike


class ComposableMolecule:
    def __init__(self):
        self.frag_idxs: List[int] = []  # indices of every block
        self.fragments: List[Mol] = []  # rdkit molecule objects for every
        self.slices: List[int] = [0]  # atom index at which every block starts
        self.num_fragments: int = 0
        self.jbonds: List[List[int]] = []  # [block1, block2, bond1, bond2]
        self.stems: List[List[int]] = []  # [block1, bond1]
        self._rdmol: Optional[Mol] = None

    def add_fragment(
        self,
        frag_idx: int,
        fragment: Mol,
        frag_r: List[int],
        stem_idx: Optional[int] = None,
        atom_idx: Optional[int] = None,
    ) -> None:
        """Adds a molecule block to the current molecule being composed.

        Parameters
        ----------
        frag_idx (int):
            The index of the molecule block that should be added
        fragment (Mol):
            The RDKit Mol representation of the block to be added
        frag_r (List[int]):
            The added block's list of indices of stems where other blocks can eventually be added
        stem_idx (Optional[int]):
            The index of the stem where the block should be added. Can be inferred from atom_idx
        atom_idx (Optional[int]):
            The index of the atom where the block should be added. Has to be present if stem_idx is not

        Raises
        ------
        ValueError
        """
        self.frag_idxs.append(frag_idx)
        self.fragments.append(fragment)
        self.slices.append(self.slices[-1] + fragment.GetNumAtoms())
        self.num_fragments += 1
        [self.stems.append([self.num_fragments - 1, r]) for r in frag_r[1:]]

        if len(self.fragments) == 1:
            self.stems.append([self.num_fragments - 1, frag_r[0]])
        else:
            if stem_idx is None:  # Try inferring stem_idx from atom_idx
                if atom_idx is None:
                    raise ValueError("One of 'stem_idx' or 'atom_idx' is required.")
                # try:
                stem_idx = np.where(self.stem_atom_idxs == atom_idx)[0][0]
                # except IndexError:
                #     print(f"Stem index could not be inferred using given atom index: {atom_idx}.")

            stem = self.stems[stem_idx]
            bond = [stem[0], self.num_fragments - 1, stem[1], frag_r[0]]
            self.stems.pop(stem_idx)
            self.jbonds.append(bond)

            # wipe current rdmol view so that it's updated on next call
            self._rdmol = None

    def delete_blocks(self, block_mask: ArrayLike) -> NDArray:
        """Deletes the given blocks from the current molecule

        Args:
            block_mask (ArrayLike): TODO

        Returns:
            NDArray: TODO
        """

        # update number of blocks
        self.num_fragments = np.sum(np.asarray(block_mask, dtype=np.int_))
        self.fragments = list(np.asarray(self.fragments)[block_mask])
        self.frag_idxs = list(np.asarray(self.frag_idxs)[block_mask])

        # update junction bonds
        reindex = np.cumsum(np.asarray(block_mask, np.int_)) - 1
        jbonds = []
        for bond in self.jbonds:
            if block_mask[bond[0]] and block_mask[bond[1]]:
                jbonds.append(np.array([reindex[bond[0]], reindex[bond[1]], bond[2], bond[3]]))
        self.jbonds = jbonds

        # update r-groups
        stems = []
        for stem in self.stems:
            if block_mask[stem[0]]:
                stems.append(np.array([reindex[stem[0]], stem[1]]))
        self.stems = stems

        # update slices
        natms = [block.GetNumAtoms() for block in self.fragments]
        self.slices = [0] + list(np.cumsum(natms))

        # destroy properties
        self._rdmol = None
        return reindex

    def remove_jbond(
        self, jbond_idx: Optional[int] = None, atom_idx: Optional[int] = None
    ) -> int:
        """Removes a junction bond from the current molecule

        Args:
            jbond_idx (Optional[int]): The index of the junction bond. Has to be present if atom_idx is not
            atom_idx (Optional[int]): The index of the atom to remove the junction bond at. Has to be present if jbond_idx is not.

        Returns:
            The index of the atom where the junction bond was removed
        """
        if jbond_idx is None:
            assert atom_idx is not None, "need jbond or atom idx"
            jbond_idx = np.where(self.jbond_atom_idxs == atom_idx)[0][0]
        else:
            assert atom_idx is None, "can't use stem and atom indices at the same time"

        # find index of the junction bond to remove
        jbond = self.jbonds.pop(jbond_idx)

        # find the largest connected component; delete rest
        jbonds = np.asarray(self.jbonds, dtype=np.int_)
        # handle the case when single last jbond was deleted
        jbonds = jbonds.reshape([len(self.jbonds), 4])
        graph = csr_matrix(
            (np.ones(self.num_fragments - 2), (jbonds[:, 0], jbonds[:, 1])),
            shape=(self.num_fragments, self.num_fragments),
        )
        _, components = connected_components(csgraph=graph, directed=False, return_labels=True)
        block_mask = components == np.argmax(np.bincount(components))
        reindex = self.delete_blocks(block_mask)

        if block_mask[jbond[0]]:
            stem = np.asarray([reindex[jbond[0]], jbond[2]])
        else:
            stem = np.asarray([reindex[jbond[1]], jbond[3]])
        self.stems.append(stem)
        atom_idx = self.slices[stem[0]] + stem[1]
        return atom_idx

    def copy(self) -> ComposableMolecule:
        """Creates a shallow copy of the current composable molecule object

        Returns
        -------
        ComposableMolecule:
            A shallow copy of this object
        """
        copy = ComposableMolecule()
        copy.frag_idxs = list(self.frag_idxs)
        copy.fragments = list(self.fragments)
        copy.slices = list(self.slices)
        copy.num_fragments = self.num_fragments
        copy.jbonds = list(self.jbonds)
        copy.stems = list(self.stems)
        return copy

    def to_dict(self) -> Dict[str, Any]:
        """Returns the attributes with their respective data as a dictionary

        Returns:
            Dict[str, Any]: The state data of this object as a dictionary
        """
        return {
            "frag_idxs": self.frag_idxs,
            "slices": self.slices,
            "num_fragments": self.num_fragments,
            "jbonds": self.jbonds,
            "stems": self.stems,
        }

    @property
    def stem_atom_idxs(self) -> NDArray:
        """The stem atom indices of the current molecule"""
        stems = np.asarray(self.stems)
        if stems.shape[0] == 0:
            stem_atom_idxs = np.array([])
        else:
            stem_atom_idxs = np.asarray(self.slices)[stems[:, 0]] + stems[:, 1]
        return stem_atom_idxs

    @property
    def jbond_atom_idxs(self) -> NDArray:
        """The junction bond atom indices of the current molecule"""
        jbonds = np.asarray(self.jbonds)
        if jbonds.shape[0] == 0:
            jbond_atom_idxs = np.array([])
        else:
            jbond_atom_idxs = np.stack(
                [
                    np.concatenate([np.asarray(self.slices)[jbonds[:, 0]] + jbonds[:, 2]]),
                    np.concatenate([np.asarray(self.slices)[jbonds[:, 1]] + jbonds[:, 3]]),
                ],
                1,
            )
        return jbond_atom_idxs

    @property
    def rdmol(self) -> Mol:
        """The current molceule in its RDKit Mol object representation. Lazily evaluated"""
        if self._rdmol is None:
            self._rdmol, _ = chem_utils.mol_from_fragments(
                jbonds=self.jbonds, frags=self.fragments
            )
        return self._rdmol

    @property
    def smiles(self) -> str:
        """The current composable molecule state in its SMILES represenation

        Returns
        -------
        str:
            SMILES representation of the composable molecule
        """
        return Chem.MolToSmiles(self.rdmol) if self.rdmol else ""
