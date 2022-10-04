from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import numpy as np
import rdkit.Chem as Chem
import torch
from pyprojroot import here
from rdkit.Chem import Atom
from torch_geometric.data import Batch, Data

from rxitect.envs.contexts import Action, ActionType, Graph, GraphEnvContext

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

    from rxitect.envs.contexts import ActionIndex


class FragmentEnvContext(GraphEnvContext):
    """A specification of what is being generated for a GraphBuildingEnv
    This context specifies how to create molecules fragment by fragment as encoded by a junction tree.
    Fragments are obtained from the original GFlowNet paper, Bengio et al., 2021.
    """

    def __init__(
        self,
        max_frags: int = 8,
        num_cond_dim: int = 0,
        frags_filepath: Union[str, Path] = here()
        / "data/processed/bengio_2021_fragments.txt",
        device: str = "cpu",
    ):
        """Construct a fragment environment
        Parameters
        ----------
        max_frags: int
            The maximum number of fragments the agent is allowed to insert.
        num_cond_dim: int
            The dimensionality of the observations' conditional information vector (if >0)
        frags_filepath: str
            The file containing the fragments available to the agent to construct molecules with. Defaults to Bengio
            et al.'s (2021) original GFlowNet paper's fragments
        device: str
            The device to process the data on, can be either 'cpu' or 'cuda'. Defaults to 'cpu'
        """
        self.max_frags = max_frags
        self.frags_smi = open(frags_filepath, "r").read().splitlines()
        self.frags_mol = [Chem.MolFromSmiles(i) for i in self.frags_smi]
        self.frags_stems = [
            [
                atom_idx
                for atom_idx in range(m.GetNumAtoms())
                if m.GetAtomWithIdx(atom_idx).GetTotalNumHs() > 0
            ]
            for m in self.frags_mol
        ]
        self.frags_num_atoms = [m.GetNumAtoms() for m in self.frags_mol]
        self.num_stem_acts = most_stems = max(map(len, self.frags_stems))
        self.action_map = [
            (frag_idx, stem_idx)
            for frag_idx in range(len(self.frags_stems))
            for stem_idx in range(len(self.frags_stems[frag_idx]))
        ]
        self.num_actions = len(self.action_map)

        # These values are used by Models to know how many inputs/logits to produce
        self.num_new_node_values = len(self.frags_smi)
        self.num_node_attr_logits = 0
        self.num_node_dim = len(self.frags_smi) + 1
        self.num_edge_attr_logits = most_stems * 2
        self.num_edge_dim = most_stems * 2
        self.num_cond_dim = num_cond_dim
        self.num_stop_logits = 1

        # Order in which models have to output logits
        self.action_type_order = [
            ActionType.STOP,
            ActionType.ADD_NODE,
            ActionType.SET_EDGE_ATTR,
        ]
        self.device = torch.device(device)

    def idx_to_action(self, g: Data, action_idx: ActionIndex):
        """Translate an action index (e.g. from a GraphActionCategorical) to a GraphAction
        Parameters
        ----------
        g: Data
            The graph object on which this action would be applied.
        action_idx: ActionIndex
             A triple describing the type of action, and the corresponding row and column index for
             the corresponding Categorical matrix.
        Returns
        -------
        action: Action
            A graph action whose type is one of STOP, ADD_NODE, or SET_EDGE_ATTR.
        """
        act_type, act_row, act_col = [int(i) for i in action_idx]
        t = self.action_type_order[act_type]
        if t is ActionType.STOP:
            return Action(t)
        elif t is ActionType.ADD_NODE:
            return Action(t, source=act_row, value=act_col)
        elif t is ActionType.SET_EDGE_ATTR:
            a, b = g.edge_index[
                :, act_row * 2
            ]  # Edges are duplicated to get undirected GNN, deduplicated for logits
            if act_col < self.num_stem_acts:
                attr = f"{int(a)}_attach"
                val = act_col
            else:
                attr = f"{int(b)}_attach"
                val = act_col - self.num_stem_acts
            return Action(t, source=a.item(), target=b.item(), attr=attr, value=val)

    def action_to_idx(self, g: Data, action: Action) -> ActionIndex:
        """Translate a GraphAction to an index tuple
        Parameters
        ----------
        g: Data
            The graph object on which this action would be applied.
        action: Action
            A graph action whose type is one of Stop, AddNode, or SetEdgeAttr.
        Returns
        -------
        action_idx: ActionIndex
             A triple describing the type of action, and the corresponding row and column index for
             the corresponding Categorical matrix.
        """
        if action.act_type is ActionType.STOP:
            row = col = 0
        elif action.act_type is ActionType.ADD_NODE:
            row = action.source
            col = action.value
        elif action.act_type is ActionType.SET_EDGE_ATTR:
            # Here the edges are duplicated, both (i,j) and (j,i) are in edge_index
            # so no need for a double check.
            row = (
                (g.edge_index.T == torch.tensor([(action.source, action.target)]))
                .prod(1)
                .argmax()
            )
            # Because edges are duplicated but logits aren't, divide by two
            row = row.div(2, rounding_mode="floor")  # type: ignore
            if action.attr == f"{int(action.source)}_attach":
                col = action.value
            else:
                col = action.value + self.num_stem_acts
        else:
            raise ValueError(f"Action type '{action.act_type}' is unsupported.")
        type_idx = self.action_type_order.index(action.act_type)
        return type_idx, int(row), int(col)

    def graph_to_data(self, g: Graph) -> Data:
        """Convert a networkx Graph to a torch geometric Data instance
        Parameters
        ----------
        g: Graph
            A Graph object representing a fragment junction tree
        Returns
        -------
        data: Data
            The corresponding torch_geometric object.
        """
        x = torch.zeros((max(1, len(g.nodes)), self.num_node_dim))
        x[0, -1] = len(g.nodes) == 0
        for i, n in enumerate(g.nodes):
            x[i, g.nodes[n]["v"]] = 1
        edge_attr = torch.zeros((len(g.edges) * 2, self.num_edge_dim))
        set_edge_attr_mask = torch.zeros((len(g.edges), self.num_edge_attr_logits))
        for i, e in enumerate(g.edges):
            ad = g.edges[e]
            for n, offset in zip(e, [0, self.num_stem_acts]):
                idx = ad.get(f"{int(n)}_attach", 0) + offset
                edge_attr[i * 2, idx] = 1
                edge_attr[i * 2 + 1, idx] = 1
                if f"{int(n)}_attach" not in ad:
                    set_edge_attr_mask[
                        i, offset : offset + len(self.frags_stems[g.nodes[n]["v"]])
                    ] = 1
        edge_index = (
            torch.tensor(
                [e for i, j in g.edges for e in [(i, j), (j, i)]], dtype=torch.long
            )
            .reshape((-1, 2))
            .T
        )
        if x.shape[0] == self.max_frags:
            add_node_mask = torch.zeros((x.shape[0], 1))
        else:
            add_node_mask = torch.ones((x.shape[0], 1))

        return Data(
            x,
            edge_index,
            edge_attr,
            add_node_mask=add_node_mask,
            set_edge_attr_mask=set_edge_attr_mask,
        )

    def collate_fn(self, graphs: List[Data]) -> Batch:
        """Batch Data instances
        Parameters
        ----------
        graphs: List[gd.Data]
            A list of gd.Data objects (e.g. given by graph_to_Data).
        Returns
        -------
        batch: gd.Batch
            A torch_geometric Batch object
        """
        return Batch.from_data_list(graphs, follow_batch=["edge_index"])

    def mol_to_graph(self, mol) -> Graph:
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
        offsets = np.cumsum([0] + [self.frags_num_atoms[g.nodes[i]["v"]] for i in g])
        mol = None
        for i in g.nodes:
            if mol is None:
                mol = self.frags_mol[g.nodes[i]["v"]]
            else:
                mol = Chem.CombineMols(mol, self.frags_mol[g.nodes[i]["v"]])

        mol = Chem.EditableMol(mol)
        bond_atoms = []
        for a, b in g.edges:
            frag_a = g.nodes[a]["v"]
            frag_b = g.nodes[b]["v"]
            u, v = (
                int(
                    self.frags_stems[frag_a][g.edges[(a, b)].get(f"{a}_attach", 0)]
                    + offsets[a]
                ),
                int(
                    self.frags_stems[frag_b][g.edges[(a, b)].get(f"{b}_attach", 0)]
                    + offsets[b]
                ),
            )
            bond_atoms += [u, v]
            mol.AddBond(u, v, Chem.BondType.SINGLE)
        mol = mol.GetMol(None)

        def _pop_hydrogen_atom(atom: Atom) -> None:
            atom = mol.GetAtomWithIdx(atom)
            nh = atom.GetNumExplicitHs()
            if nh > 0:
                atom.SetNumExplicitHs(nh - 1)

        list(map(_pop_hydrogen_atom, bond_atoms))
        return mol

    def is_valid_graph(self, g: Graph) -> bool:
        """Verifies whether the given Graph is valid according to RDKit"""
        mol = self.graph_to_mol(g)
        assert Chem.MolFromSmiles(Chem.MolToSmiles(mol)) is not None
        if mol is None:
            return False
        return True
