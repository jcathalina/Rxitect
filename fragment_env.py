from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from os import PathLike
from typing import List, Tuple, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Mol
from torch import nn

import chem_utils
from composable_mol import ComposableMolecule
from torch_geometric.data import Data, Batch
from pyprojroot import here


class FragmentEnv:
    def __init__(
        self,
        fragments_filepath: PathLike = here() / "bengio2021_fragments_105.json",
        device: str = "cpu",
    ) -> None:
        """
        A class that abstracts the actions that can be taken to construct a molecule
        from a finite set of molecule building blocks (akin to LEGO) as a Markov Decision Process (MDP).

        Parameters
        ----------
        fragments_filepath:
            The filepath to a JSON file containing the molecule building blocks (fragments).
        device:
            The device that the data has to be processed on
        """
        fragments = pd.read_json(fragments_filepath)
        self.frag_rs: List[List[int]] = fragments.get("block_r").to_list()
        self.frag_smiles: List[str] = fragments.get("block_smi").to_list()
        self.frag_rdmols: List[Mol] = [Chem.MolFromSmiles(smi) for smi in self.frag_smiles]
        self.true_frag_smiles: List[str] = sorted(set(self.frag_smiles))
        self.num_true_frags: int = len(self.true_frag_smiles)
        self.frag_num_atoms = np.asarray(
            [rdmol.GetNumAtoms() for rdmol in self.frag_rdmols], dtype=np.int_
        )
        self.stem_type_offset = np.asarray(
            [0]
            + np.cumsum(
                [
                    max(self.frag_rs[self.frag_smiles.index(i)]) + 1
                    for i in self.true_frag_smiles
                ]
            ).tolist(),
            dtype=np.int_,
        )
        self.num_stem_types: int = self.stem_type_offset[-1]
        self.true_frag_idxs: List[int] = [self.true_frag_smiles.index(smi) for smi in self.frag_smiles]
        self.device = device

        # helper lambda
        self._long_tensor = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)

        # properties
        self._num_frags: Optional[int] = None
        self._cmol: Optional[ComposableMolecule] = None

        # inferred attributes
        self._build_translation_table()

    def cmol_to_data(self, cmol: ComposableMolecule) -> Data:
        """
        Transforms the composable molecule to it's pytorch geometric data representation for downstream use

        Parameters
        ----------
        cmol:
            A composable molecule

        Returns
        -------
        Data
            The torch geometric data representation of the composable molecule
        """
        if cmol.num_fragments == 0:
            return self._build_data(cmol=cmol)

        ti = self.true_frag_idxs
        edges = [(jbond[0], jbond[1]) for jbond in cmol.jbonds]

        edge_attrs = []
        for jbond in cmol.jbonds:
            edge_attr = (
                self.stem_type_offset[ti[cmol.frag_idxs[jbond[0]]]] + jbond[2],
                self.stem_type_offset[ti[cmol.frag_idxs[jbond[1]]]] + jbond[3],
            )
            edge_attrs.append(edge_attr)

        stem_types = [self.stem_type_offset[ti[cmol.frag_idxs[s[0]]]] for s in cmol.stems]
        data = self._build_data(
            cmol=cmol, edges=edges, edge_attrs=edge_attrs, stem_types=stem_types
        )
        data.to(device=self.device)
        return data

    def collate_fn(self, mol_data_list: List[Data]) -> Batch:
        batch = Batch.from_data_list(mol_data_list, follow_batch=["stems"])
        batch.to(self.device)
        return batch

    def cmols_to_batch(self, cmols: List[ComposableMolecule]) -> Batch:
        mol_data_list = [self.cmol_to_data(cmol) for cmol in cmols]
        batch = self.collate_fn(mol_data_list)
        return batch

    def reset(self) -> None:
        """Resets the internal Molecule being constructed within the MDP"""
        self._cmol = ComposableMolecule()

    def add_fragment_inplace(
        self,
        frag_idx: int,
        stem_idx: Optional[int] = None,
        atom_idx: Optional[int] = None,
    ) -> None:
        """In-place action to add a molecule block to the current molecule being constructed within the MDP.

        Parameters
        ----------
            frag_idx:
                The index of the block being added
            stem_idx:
                The index of the stem where the block should be added
            atom_idx:
                The index of the atom at which the block should be added
        """
        if frag_idx < 0 or frag_idx > self.num_frags:
            raise ValueError("Attemped to add an unknown block")

        self.cmol.add_fragment(
            frag_idx=frag_idx,
            fragment=self.frag_rdmols[frag_idx],
            frag_r=self.frag_rs[frag_idx],
            stem_idx=stem_idx,
            atom_idx=atom_idx,
        )

    def add_fragment(
        self,
        cmol: ComposableMolecule,
        frag_idx: int,
        stem_idx: Optional[int] = None,
        atom_idx: Optional[int] = None,
    ) -> ComposableMolecule:
        """Adds block to a copy of the passed composable molecule and returns this copy

        Parameters
        ----------
        cmol:
            A composable molecule
        frag_idx:
            The index of the fragment being added
        stem_idx:
            The index of the stem where the fragment should be added. Should be present if atom_idx is not
        atom_idx:
            The index of the atom at which the fragment should be added. Should be present if stem_idx is not

        Returns
        -------
        ComposableMolecule:
            A copy of the passed molecule with the indicated block added to it
        """
        if cmol.num_fragments == 0:
            stem_idx = None

        cmol_copy = cmol.copy()
        cmol_copy.add_fragment(
            frag_idx=frag_idx,
            fragment=self.frag_rdmols[frag_idx],
            frag_r=self.frag_rs[frag_idx],
            stem_idx=stem_idx,
            atom_idx=atom_idx,
        )
        return cmol_copy

    def parents(
        self, cmol: Optional[ComposableMolecule] = None
    ) -> List[Tuple[ComposableMolecule, Tuple[int, int]]]:
        """
        Retrieves all the possible parents of the given composable molecule, along with the action
        that would result in the parent state transitioning into the current state.
        Uses current molecule being built in environment by default if nothing is passed.

        Parameters
        ----------
        cmol:
            The molecule to get all possible parent molecules for.
            If left unspecified, uses the molecule currently being constructed in the MDP.

        Returns
        -------
        List[Tuple[ComposableMolecule, Tuple[int, int]]]
            A list of (ComposableMolecule, (frag_idx, stem_idx)) pairs such that
            for each pair (Par(m), (f, s)), adding fragment with index f to Par(m) at stem with index s
            will lead to the current/given composable molecule.
        """
        if cmol is None:
            cmol = self.cmol

        if len(cmol.frag_idxs) == 1:
            # If there's just a single block, then the only parent is
            # the empty block with the action that recreates that block
            return [(ComposableMolecule(), (cmol.frag_idxs[0], 0))]

        # Compute how many fragments each fragment is connected to
        blocks_degree = defaultdict(int)
        for a, b, _, _ in cmol.jbonds:
            blocks_degree[a] += 1
            blocks_degree[b] += 1
        # Keep only blocks of degree 1 (those are the ones that could have just been added)
        blocks_degree_1: List[int] = [i for i, d in blocks_degree.items() if d == 1]
        # Form new molecules without these blocks
        parent_mols = []

        for rm_block_idx in blocks_degree_1:
            cmol_copy = cmol.copy()
            # find which bond we're removing
            removed_bonds = [
                (jbond_idx, bond)
                for jbond_idx, bond in enumerate(cmol_copy.jbonds)
                if rm_block_idx in bond[:2]
            ]
            if len(removed_bonds) != 1:
                raise ValueError(f"Unexpected number of removable bonds found: {len(removed_bonds)}")

            rm_jbond_idx, rm_bond = removed_bonds[0]
            # Pop the bond
            cmol_copy.jbonds.pop(rm_jbond_idx)
            # Remove the block
            mask = np.ones(len(cmol_copy.frag_idxs), dtype=bool)
            mask[rm_block_idx] = 0
            reindex = cmol_copy.delete_blocks(mask)

            """
            reindex maps old blockidx to new blockidx, since the
            fragment that the removed fragment was attached to may have had its
            index shifted by 1.
            """

            # Compute which stem the bond was using
            stem = (
                [reindex[rm_bond[0]], rm_bond[2]]
                if rm_block_idx == rm_bond[1]
                else [reindex[rm_bond[1]], rm_bond[3]]
            )
            # and add it back
            cmol_copy.stems = [list(i) for i in cmol_copy.stems] + [stem]
            """
            and we have a parent. The stem idx to recreate mol is
            the last stem, since we appended `stem` in the back of
            the stem list.
            We also have to translate the block id to match the bond
            we broke, see build_translation_table().
            """
            removed_stem_atom = rm_bond[3] if rm_block_idx == rm_bond[1] else rm_bond[2]
            blockid = cmol.frag_idxs[rm_block_idx]
            if removed_stem_atom not in self.translation_table_[blockid]:
                raise ValueError(
                    "Could not translate removed stem to duplicate or symmetric block."
                )
            parent_mols.append(
                (
                    cmol_copy,
                    (
                        self.translation_table_[blockid][removed_stem_atom],
                        len(cmol_copy.stems) - 1,
                    ),
                )
            )
        if not len(parent_mols):
            raise ValueError("Could not find any parents")
        return parent_mols

    def random_walk(self, max_fragments: int) -> None:
        """Action to perform a random walk, i.e. construct a molecule by randomly adding fragments

        Args:
            max_fragments: The maximum number of fragments allowed to be used to create the molecule
        """
        done = False
        while not done:
            if self.cmol.num_fragments == 0:
                frag_idx = np.random.choice(np.arange(self.num_frags))
                stem_idx = None
                self.add_fragment_inplace(frag_idx=frag_idx, stem_idx=stem_idx)
                if self.cmol.num_fragments >= max_fragments:
                    if self.cmol.slices[-1] > 1:
                        done = True
                    else:
                        self.reset()
            elif len(self.cmol.stems) > 0:
                frag_idx = np.random.choice(np.arange(self.num_frags))
                stem_idx = np.random.choice(len(self.cmol.stems))
                self.add_fragment_inplace(frag_idx=frag_idx, stem_idx=stem_idx)
                if self.cmol.num_fragments >= max_fragments:
                    done = True
            else:
                self.reset()

    def cmol_to_nx(self, cmol: ComposableMolecule, use_true_fragments: bool = False) -> nx.DiGraph:
        """Gets the composable molecule in its networkx (directed) graph representation

        Parameters
        ----------
        cmol:
            A composable molecule
        use_true_fragments:
            Flag that determines if the true fragments should be used for constructing the graph. Defaults to False

        Returns
        -------
        nx.DiGraph
            The networkx directed graph representation of the given composable molecule
        """

        dG = nx.DiGraph()
        frag_idxs = (
            [self.true_frag_idxs[i] for i in cmol.frag_idxs] if use_true_fragments else cmol.frag_idxs
        )

        dG.add_nodes_from([(ix, {"fragment": frag_idxs[ix]}) for ix in range(len(frag_idxs))])

        if len(cmol.jbonds) > 0:
            edges = []
            for jbond in cmol.jbonds:
                edges.append((jbond[0], jbond[1], {"bond": [jbond[2], jbond[3]]}))
                edges.append((jbond[1], jbond[0], {"bond": [jbond[3], jbond[2]]}))
            dG.add_edges_from(edges)
        return dG

    @property
    def num_frags(self) -> int:
        """Max number of possible building blocks in current MDP instance"""
        if self._num_frags is None:
            self._num_frags = len(self.frag_smiles)
        return self._num_frags

    @property
    def cmol(self) -> ComposableMolecule:
        if self._cmol is None:
            self.reset()
        return self._cmol

    def _build_data(
        self,
        cmol: ComposableMolecule,
        edges: List[Tuple[int, int]] = None,
        edge_attrs: List[Tuple[int, int]] = None,
        stem_types: List = None,
    ) -> Data:
        """
        Helper func that returns the torch geometric data object representation of the composable molecule.
        If none of the params (except for the presumably empty composable molecule) are passed,
        this returns an empty torch geometric data object with the correct backbone expected by the agent.

        Parameters
        ----------
        cmol:
            x
        edges:
            x
        edge_attrs:
            x
        stem_types:
            x

        Returns
        -------
        Data
            The torch geometric data representation of the composable molecule
        """
        if edges is None:
            edges = []
        if edge_attrs is None:
            edge_attrs = []
        if stem_types is None:
            stem_types = []

        x = (self._long_tensor([self.true_frag_idxs[frag_idx] for frag_idx in cmol.frag_idxs])
             if cmol.num_fragments > 0
             else self._long_tensor([self.num_true_frags]))

        data = Data(
            x=x,
            edge_index=self._long_tensor(edges).T
            if len(edges)
            else self._long_tensor([[], []]),
            edge_attrs=self._long_tensor(edge_attrs)
            if len(edges)
            else self._long_tensor([]).reshape((0, 2)),
            stems=self._long_tensor(cmol.stems)
            if len(cmol.stems)
            else self._long_tensor([(0, 0)]),
            stem_types=self._long_tensor(stem_types)
            if len(cmol.stems)
            else self._long_tensor([self.num_stem_types]),
        )
        return data

    def _build_translation_table(self) -> None:
        """Builds a symmetry mapping for blocks. Necessary to compute parent transitions
        Creates and sets the inferred attribute 'translation_table_' for this class."""
        self.translation_table_ = {}
        for frag_idx in range(len(self.frag_rdmols)):
            """
            Blocks have multiple ways of being attached. By default,
            a new block is attached to the target stem by attaching
            its k-th atom, where k = block_rs[new_block_idx][0].
            When computing a reverse action (from a parent), we may
            wish to attach the new block to a different atom. In
            the blocks library, there are duplicates of the same
            block but with block_rs[block][0] set to a different
            atom. Thus, for the reverse action we have to find out
            which duplicate this corresponds to.

            Here, we compute, for each block's block_idx, what the index
            of the duplicate block is, if a block needs to be attached to
            atom x of said block.
            Thus: atom_map[x] == block_idx, such that block_rs[block_idx][0] == x
            """
            atom_map = {}
            for j in range(len(self.frag_rdmols)):
                if self.frag_smiles[frag_idx] == self.frag_smiles[j]:
                    atom_map[self.frag_rs[j][0]] = j
            self.translation_table_[frag_idx] = atom_map

        """ 
        We're still missing some "duplicates", as some might be
        symmetric versions of each other. For example, block CC with
        block_rs == [0,1] has no duplicate, because the duplicate
        with block_rs [1,0] would be a symmetric version (both C
        atoms are the "same").

        To test this, let's create nonsense molecules by attaching
        duplicate blocks to a Gold atom, and testing whether they
        are the same.
        """
        gold_dummy = Chem.MolFromSmiles("[Au]")
        """
        If we find that two molecules are the same when attaching
        them with two different atoms, then that means the atom
        numbers are symmetries. We can add those to the table.
        """
        for frag_idx in range(len(self.frag_rdmols)):
            for j in self.frag_rs[frag_idx]:
                if j not in self.translation_table_[frag_idx]:
                    symmetric_duplicate = None
                    for atom, block_duplicate in self.translation_table_[frag_idx].items():
                        molA, _ = chem_utils.mol_from_fragments(
                            jbonds=[[0, 1, 0, j]],
                            frags=[gold_dummy, self.frag_rdmols[frag_idx]],
                        )
                        molB, _ = chem_utils.mol_from_fragments(
                            jbonds=[[0, 1, 0, atom]],
                            frags=[gold_dummy, self.frag_rdmols[frag_idx]],
                        )
                        if Chem.MolToSmiles(molA) == Chem.MolToSmiles(
                            molB
                        ) or molA.HasSubstructMatch(molB):
                            symmetric_duplicate = block_duplicate
                            break
                    if symmetric_duplicate is None:
                        raise ValueError(
                            f"Fragment {frag_idx}: '{self.frag_smiles[frag_idx]}', has no duplicate for atom {j} in "
                            f"position 0, and no symmetrical correspondence, "
                        )
                    self.translation_table_[frag_idx][j] = symmetric_duplicate


# @dataclass(slots=True)
# class Trajectory:
#     pass
#
#
# def sample_model(sampling_model: nn.Module, max_frags: int, ctx: FragmentEnv):
#     cmol = ComposableMolecule()
#     trajs = []
#     s = ctx.cmols_to_batch([cmol])
#     for t in range(max_frags):
#         with torch.no_grad():
#             next_stem, next_cmol = sampling_model(s)


if __name__ == "__main__":
    env = FragmentEnv()
    print(env.__dict__)
