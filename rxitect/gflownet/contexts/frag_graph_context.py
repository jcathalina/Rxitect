from typing import List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import torch
from networkx import Graph
from rdkit.Chem import Mol, AtomValenceException
from torch_geometric.data import Data, Batch

from rxitect.gflownet.contexts.interfaces.graph_context import IGraphContext
from rxitect.gflownet.utils.graph import GraphActionType, GraphAction
from pyprojroot import here


class FragBasedGraphContext(IGraphContext):
    """A specification of what is being generated for a Fragment-based graph building environment

    This context specifies how to create molecules fragment by fragment as encoded by a junction tree.
    Fragments are obtained from the original GFlowNet paper, Bengio et al., 2021.
    """

    def __init__(self, max_frags: int = 9, num_cond_dim: int = 0) -> None:
        """Construct a fragment environment
        Parameters
        ----------
        max_frags: int
            The maximum number of fragments the agent is allowed to insert.
        num_cond_dim: int
            The dimensionality of the observations' conditional information vector (if >0)
        """
        self.max_frags = max_frags
        df = pd.read_json(here() / "data/bengio2021_fragments_105.json")
        self.frags_smi = df["frag_smiles"].to_list()
        self.frags_stems = df["frag_stems"].to_list()
        self.frags_mol = [Chem.MolFromSmiles(i) for i in self.frags_smi]
        self.frags_num_atom = [m.GetNumAtoms() for m in self.frags_mol]
        self.num_stem_acts = most_stems = max(map(len, self.frags_stems))
        self.action_map = [(frag_idx, stem_idx)
                           for frag_idx in range(len(self.frags_stems))
                           for stem_idx in range(len(self.frags_stems[frag_idx]))]
        self.num_actions = len(self.action_map)
        # These values are used by Models to know how many inputs/logits to produce
        self.num_new_node_values = len(self.frags_smi)
        self.num_node_attr_logits = 0
        self.num_node_dim = len(self.frags_smi) + 1
        self.num_edge_attr_logits = most_stems * 2
        self.num_edge_dim = most_stems * 2
        self.num_cond_dim = num_cond_dim

        # Order in which models have to output logits
        self.action_type_order = [GraphActionType.Stop, GraphActionType.AddNode, GraphActionType.SetEdgeAttr]
        self.device = torch.device('cpu')

    def aidx_to_GraphAction(self, g: Data, action_idx: Tuple[int, int, int]) -> GraphAction:
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction

        Parameters
        ----------
        g: gd.Data
            The graph object on which this action would be applied.
        action_idx: Tuple[int, int, int]
             A triple describing the type of action, and the corresponding row and column index for
             the corresponding Categorical matrix.

        Returns
        -------
        action: GraphAction
            A graph action whose type is one of Stop, AddNode, or SetEdgeAttr.
        """
        act_type, act_row, act_col = [int(i) for i in action_idx]
        t = self.action_type_order[act_type]
        if t is GraphActionType.Stop:
            return GraphAction(t)
        elif t is GraphActionType.AddNode:
            return GraphAction(t, source=act_row, value=act_col)
        elif t is GraphActionType.SetEdgeAttr:
            a, b = g.edge_index[:, act_row * 2]  # Edges are duplicated to get undirected GNN, deduplicated for logits
            if act_col < self.num_stem_acts:
                attr = f'{int(a)}_attach'
                val = act_col
            else:
                attr = f'{int(b)}_attach'
                val = act_col - self.num_stem_acts
            return GraphAction(t, source=a.item(), target=b.item(), attr=attr, value=val)

    def GraphAction_to_aidx(self, g: Data, action: GraphAction) -> Tuple[int, int, int]:
        """Translate a GraphAction to an index tuple

        Parameters
        ----------
        g: gd.Data
            The graph object on which this action would be applied.
        action: GraphAction
            A graph action whose type is one of Stop, AddNode, or SetEdgeAttr.

        Returns
        -------
        action_idx: Tuple[int, int, int]
             A triple describing the type of action, and the corresponding row and column index for
             the corresponding Categorical matrix.
        """
        if action.action is GraphActionType.Stop:
            row = col = 0
        elif action.action is GraphActionType.AddNode:
            row = action.source
            col = action.value
        elif action.action is GraphActionType.SetEdgeAttr:
            # Here the edges are duplicated, both (i,j) and (j,i) are in edge_index
            # so no need for a double check.
            row = (g.edge_index.T == torch.tensor([(action.source, action.target)])).prod(1).argmax()
            # Because edges are duplicated but logits aren't, divide by two
            row = row.div(2, rounding_mode='floor')  # type: ignore
            if action.attr == f'{int(action.source)}_attach':
                col = action.value
            else:
                col = action.value + self.num_stem_acts
        else:
            raise ValueError(f"Action of type {action.action} is not supported")
        type_idx = self.action_type_order.index(action.action)
        return type_idx, int(row), int(col)

    def graph_to_Data(self, g: Graph) -> Data:
        """Convert a networkx Graph to a torch geometric Data instance
        Parameters
        ----------
        g: Graph
            A Graph object representing a fragment junction tree

        Returns
        -------
        data:  Data
            The corresponding torch_geometric object.
        """
        x = torch.zeros((max(1, len(g.nodes)), self.num_node_dim))
        x[0, -1] = len(g.nodes) == 0
        for i, n in enumerate(g.nodes):
            x[i, g.nodes[n]['v']] = 1
        edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
        set_edge_attr_mask = torch.zeros((len(g.edges), self.num_edge_attr_logits))
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for n, offset in zip(e, [0, self.num_stem_acts]):
                idx = ad.get(f'{int(n)}_attach', 0) + offset
                edge_attr[i * 2, idx] = 1
                edge_attr[i * 2 + 1, idx] = 1
                if f'{int(n)}_attach' not in ad:
                    set_edge_attr_mask[i, offset:offset + len(self.frags_stems[g.nodes[n]['v']])] = 1
        edge_index = torch.tensor([e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long).reshape(
            (-1, 2)).T
        if x.shape[0] == self.max_frags:
            add_node_mask = torch.zeros((x.shape[0], 1))
        else:
            # TODO: This is where we should be checking if nodes are still valid to be used as sources right?
            add_node_mask = torch.ones((x.shape[0], 1))
            # ATTEMPT 1: MASK SOURCE NODES MAYBE
            if len(g.nodes) > 1:
                for i, n in enumerate(g.nodes):
                    node = g.nodes[n]
                    curr_frag_idx = node['v']
                    if g.degree(i) == len(self.frags_stems[curr_frag_idx]):
                        add_node_mask[i] = 0
                    if g.degree(i) > len(self.frags_stems[curr_frag_idx]):  # TODO: Grab stems from value of node
                        print("What the fuck man.")

        return Data(x, edge_index, edge_attr, add_node_mask=add_node_mask, set_edge_attr_mask=set_edge_attr_mask)

    def collate(self, graphs: List[Data]) -> Batch:
        """Batch Data instances

        Parameters
        ----------
        graphs:
            A list of gd.Data objects (e.g. given by graph_to_Data).

        Returns
        -------
        batch: Batch
            A torch_geometric Batch object
        """
        return Batch.from_data_list(graphs, follow_batch=['edge_index'])

    def mol_to_graph(self, mol: Mol) -> Graph:
        """Convert an RDMol to a Graph"""
        raise NotImplementedError()

    def graph_to_mol(self, g: Graph) -> Mol:
        """Convert a Graph to an RDKit molecule

        Parameters
        ----------
        g: Graph
            A Graph instance representing a fragment junction tree.

        Returns
        -------
        m: Mol
            The corresponding RDKit molecule
        """
        offsets = np.cumsum([0] + [self.frags_num_atom[g.nodes[i]['v']] for i in g])
        mol = None
        for i in g.nodes:
            if mol is None:
                mol = self.frags_mol[g.nodes[i]['v']]
            else:
                mol = Chem.CombineMols(mol, self.frags_mol[g.nodes[i]['v']])

        mol = Chem.EditableMol(mol)
        bond_atoms = []
        for a, b in g.edges:
            # try:
            afrag = g.nodes[a]['v']
            bfrag = g.nodes[b]['v']
            u, v = (int(self.frags_stems[afrag][g.edges[(a, b)].get(f'{a}_attach', 0)] + offsets[a]),
                    int(self.frags_stems[bfrag][g.edges[(a, b)].get(f'{b}_attach', 0)] + offsets[b]))
            # if self.frags_stems[afrag]:
            #     u = int(self.frags_stems[afrag][g.edges[(a, b)].get(f'{a}_attach', 0)] + offsets[a])
            # else:
            #     u = int(offsets[a])
            #
            # if self.frags_stems[bfrag]:
            #     v = int(self.frags_stems[bfrag][g.edges[(a, b)].get(f'{b}_attach', 0)] + offsets[b])
            # else:
            #     v = int(offsets[b])

            bond_atoms += [u, v]
            # except IndexError:
            #     print(f"Index that caused the error: a={a} OR b={b} -- edges: {g.edges}, #edges: {len(g.edges)}")
            #     print(f"a-frag: {afrag}, u: {self.frags_stems[afrag]}")
            #     print(f"b-frag: {bfrag}, v: {self.frags_stems[bfrag]}")
            #     exit(0)
            mol.AddBond(u, v, Chem.BondType.SINGLE)
        mol = mol.GetMol()

        # print(f"Current mol looks like this: {Chem.MolToSmiles(mol)}")

        def _pop_H(atom):
            atom = mol.GetAtomWithIdx(atom)
            nh = atom.GetNumExplicitHs()
            if nh > 0:
                atom.SetNumExplicitHs(nh - 1)

        for bond_atom in bond_atoms:
            _pop_H(bond_atom)

        smiles = Chem.MolToSmiles(mol)
        Chem.SanitizeMol(mol)
        return mol

    def is_sane(self, g: Graph) -> bool:
        """Verifies whether the given Graph is valid according to RDKit"""
        try:
            mol = self.graph_to_mol(g)
        # FIXME: I mean, this catch-all exception works but training is very unstable with all of these invalid
        #   options. This catches (1) Empty graphs, (2) Unkekulizable mols (3) Valence greater than permitted
        #   (4) List index out of range for when a fragment with no stems is picked
        except Exception as e:
            # print(f"The following exception occurred during sanity check: {e}")
            return False
        if mol is None:
            return False
        return True
