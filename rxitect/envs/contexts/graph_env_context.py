"""Adapted from recursionpharma's gflownet implementation @ https://github.com/recursionpharma/gflownet.
Contains code designed to give context for actions an agent can take in a setting where
the actions are a combination of choosing a molecular fragment and where to attach it, effectively
resulting in the creation of a (final) molecular graph.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
from torch.types import Device
from torch_geometric.data import Batch, Data
from torch_scatter import scatter, scatter_max

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


class Graph(nx.Graph):
    """A wrapper around networkx's Graph class to facilitate debugging."""

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f"<{list(self.nodes)}, {list(self.edges)}, {list(self.nodes[i]['v'] for i in self.nodes)}>"


class ActionType(Enum):
    """Class that contains all actions in the context of fragment-based molecular graph building"""

    STOP = auto()
    ADD_NODE = auto()
    ADD_EDGE = auto()
    SET_NODE_ATTR = auto()
    SET_EDGE_ATTR = auto()
    REMOVE_NODE = auto()
    REMOVE_EDGE = auto()
    REMOVE_NODE_ATTR = auto()
    REMOVE_EDGE_ATTR = auto()


@dataclass
class Action:
    """A class representing a single graph-building action

    Parameters
    ----------
    act_type: ActionType
        The action type
    source: :obj:`int`, optional
        The source node this action is applied on
    target: :obj:`int`, optional
        The target node (i.e. if specified this is an edge action)
    value: :obj:`Any`, optional
        The value (e.g. new node type) applied
    attr: :obj:`str`, optional
        The set attribute of a node/edge
    """

    act_type: ActionType
    source: Optional[int] = None
    target: Optional[int] = None
    value: Optional[Any] = None
    attr: Optional[str] = None

    def __repr__(self) -> str:
        attrs = ", ".join(
            str(i)
            for i in [self.source, self.target, self.attr, self.value]
            if i is not None
        )
        return f"<{self.act_type}, {attrs}>"


class GraphEnvContext(ABC):
    device: Device

    @abstractmethod
    def idx_to_action(self, g: Data, idx: ActionIndex) -> Action:
        """Translate an action index (e.g. from an ActionCategorical) to an Action
        Parameters
        ----------
        g: Data
            The graph to which the action is being applied
        idx: ActionIndex
            The tensor indices for the corresponding action
        Returns
        -------
        action: Action
            A graph action that could be applied to the original graph corresponding to g.
        """
        pass

    @abstractmethod
    def action_to_idx(self, g: Data, action: Action) -> ActionIndex:
        """Translate a Action to an action index (e.g. from an ActionCategorical)
        Parameters
        ----------
        g: Data
            The graph to which the action is being applied
        action: Action
            A graph action that could be applied to the original graph corresponding to g.
        Returns
        -------
        action_idx: ActionIndex
            The tensor indices for the corresponding action
        """
        pass

    @abstractmethod
    def graph_to_data(self, g: Graph) -> Data:
        """Convert a networkx Graph to a torch geometric Data instance
        Parameters
        ----------
        g: Graph
            A graph instance.
        Returns
        -------
        torch_g: Data
            The corresponding torch_geometric graph.
        """
        pass

    @classmethod
    def collate_fn(cls, graphs: List[Data]) -> Batch:
        """Convert a list of torch geometric Data instances to a Batch
        instance.  This exists so that environment contexts can set
        custom batching attributes, e.g. by using `follow_batch`.
        Parameters
        ----------
        graphs: List[Data]
            Graph instances
        Returns
        -------
        batch: Batch
            The corresponding batch.
        """
        return Batch.from_data_list(graphs)

    @abstractmethod
    def is_valid_graph(self, g: Graph) -> bool:
        """Verifies whether a graph is valid according to the context. This can
        catch, e.g. impossible molecules.

        Parameters
        ----------
        g: Graph
            A graph.
        Returns
        -------
        is_sane: bool:
            True if the environment considers g to be valid.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def mol_to_graph(self, mol: Mol) -> Graph:
        """Transforms an RDKit representation of a molecule into
        its corresponding generic Graph representation
        Parameters
        ----------
        mol: Mol
            An RDKit molecule
        Returns
        -------
        g: Graph
            The corresponding Graph representation of that molecule.
        """
        pass


class ActionCategorical:
    def __init__(
        self,
        graphs: Batch,
        logits: List[torch.Tensor],
        keys: List[str],
        types: List[ActionType],
        deduplicate_edge_index: bool = True,
    ):
        """A multi-type Categorical compatible with generating structured actions.
        What is meant by type here is that there are multiple types of
        mutually exclusive actions, e.g. AddNode and AddEdge are
        mutually exclusive, but since their logits will be produced by
        different variable-sized tensors (corresponding to different
        elements of the graph, e.g. nodes or edges) it is inconvient
        to stack them all into one single Categorical. This class
        provides this convenient interaction between torch_geometric
        Batch objects and lists of logit tensors.
        Parameters
        ----------
        graphs: Batch
            A Batch of graphs to which the logits correspond
        logits: List[torch.Tensor]
            A list of tensors of shape `(n, m)` representing logits
            over a variable number of graph elements (e.g. nodes) for
            which there are `m` possible actions. `n` should thus be
            equal to the sum of the number of such elements for each
            graph in the Batch object. The length of the `logits` list
            should thus be equal to the number of element types (in
            other words there should be one tensor per type).
        keys: List[Union[str, None]]
            The keys corresponding to the Graph elements for each
            tensor in the logits list. Used to extract the `_batch`
            and slice attributes. For example, if the first logit
            tensor is a per-node action logit, and the second is a
            per-edge, `keys` could be `['x', 'edge_index']`. If
            keys[i] is None, the corresponding logits are assumed to
            be graph-level (i.e. if there are `k` graphs in the Batch
            object, this logit tensor would have shape `(k, m)`)
        types: List[ActionType]
           The action type each logit corresponds to.
        deduplicate_edge_index: bool, default=True
           If true, this means that the 'edge_index' keys have been reduced
           by e_i[::2] (presumably because the graphs are undirected)
        """
        # TODO: handle legal action masks? (e.g. can't add a node attr to a node that already has an attr)
        self.num_graphs = graphs.num_graphs
        # The logits
        self.logits = logits
        self.types = types
        self.keys = keys
        self.dev = dev = graphs.x.device

        # I'm extracting batches and slices in a slightly hackish way,
        # but I'm not aware of a proper API to torch_geometric that
        # achieves this "neatly" without accessing private attributes

        # This is the minibatch index of each entry in the logits
        # i.e., if graph i in the Batch has N[i] nodes,
        #    g.batch == [0,0,0, ...,  1,1,1,1,1, ... ]
        #                 N[0] times    N[1] times
        # This generalizes to edges and non-edges.
        # Append '_batch' to keys except for 'x', since TG has a special case (done by default for 'x')
        self.batch = [
            getattr(graphs, f"{k}_batch" if k != "x" else "batch") if k is not None
            # None signals a global logit rather than a per-instance logit
            else torch.arange(graphs.num_graphs, device=dev)
            for k in keys
        ]
        # This is the cumulative sum (prefixed by 0) of N[i]s
        self.slice = [
            graphs._slice_dict[k]
            if k is not None
            else torch.arange(graphs.num_graphs, device=dev)
            for k in keys
        ]
        self.log_probs = None

        if deduplicate_edge_index and "edge_index" in keys:
            idx = keys.index("edge_index")
            self.batch[idx] = self.batch[idx][::2]
            self.slice[idx] = self.slice[idx].div(2, rounding_mode="floor")

    def detach(self):
        new = copy(self)
        new.logits = [i.detach() for i in new.logits]
        if new.log_probs is not None:
            new.log_probs = [i.detach() for i in new.log_probs]
        return new

    def to(self, device):
        self.dev = device
        self.logits = [i.to(device) for i in self.logits]
        self.batch = [i.to(device) for i in self.batch]
        self.slice = [i.to(device) for i in self.slice]
        if self.log_probs is not None:
            self.log_probs = [i.to(device) for i in self.log_probs]
        return self

    def log_softmax(self):
        """Compute log-probabilities given logits"""
        if self.log_probs is not None:
            return self.log_probs
        # Use the `subtract by max` trick to avoid precision errors:
        # compute max
        maxl = (
            torch.cat(
                [
                    scatter(i, b, dim=0, dim_size=self.num_graphs, reduce="max")
                    for i, b in zip(self.logits, self.batch)
                ],
                dim=1,
            )
            .max(1)
            .values.detach()
        )
        # subtract by max then take exp
        # x[b, None] indexes by the batch to map back to each node/edge and adds a broadcast dim
        exp_logits = [
            (i - maxl[b, None]).exp() + 1e-40 for i, b in zip(self.logits, self.batch)
        ]
        # sum corrected exponentiated logits, to get log(Z - max) = log(sum(exp(logits)) - max)
        log_z = sum(
            [
                scatter(i, b, dim=0, dim_size=self.num_graphs, reduce="sum").sum(1)
                for i, b in zip(exp_logits, self.batch)
            ]
        ).log()
        # log probabilities is log(exp(logit) / Z)
        self.log_probs = [
            i.log() - log_z[b, None] for i, b in zip(exp_logits, self.batch)
        ]
        return self.log_probs

    def sample(self) -> List[ActionIndex]:
        # Use the Gumbel trick to sample categoricals
        # i.e. if X ~ argmax(logits - log(-log(uniform(logits.shape))))
        # then  p(X = i) = exp(logits[i]) / Z
        # Here we have to do the argmax first over the variable number
        # of rows of each element type for each graph in the
        # minibatch, then over the different types (since they are
        # mutually exclusive).

        # Uniform noise
        u = [torch.rand(i.shape, device=self.dev) for i in self.logits]
        # Gumbel noise
        gumbel = [logit - (-noise.log()).log() for logit, noise in zip(self.logits, u)]
        # scatter_max and .max create a (values, indices) pair
        # These logits are 2d (num_obj_of_type, num_actions_of_type),
        # first reduce-max over the batch, which preserves the
        # columns, so we get (minibatch_size, num_actions_of_type).
        # First we prefill `out` with very negative values in case
        # there are no corresponding logits (this can happen if e.g. a
        # graph has no edges), we don't want to accidentally take the
        # max of that type.
        mnb_max = [
            torch.zeros(self.num_graphs, i.shape[1], device=self.dev) - 1e6
            for i in self.logits
        ]
        mnb_max = [
            scatter_max(i, b, dim=0, out=out)
            for i, b, out in zip(gumbel, self.batch, mnb_max)
        ]
        # Then over cols, this gets us which col holds the max value,
        # so we get (minibatch_size,)
        col_max = [values.max(1) for values, idx in mnb_max]
        # Now we look up which row in those argmax cols was the max:
        row_pos = [
            idx_mnb[torch.arange(len(idx_col)), idx_col]
            for (_, idx_mnb), (_, idx_col) in zip(mnb_max, col_max)
        ]
        # The maxes themselves
        maxs = [values for values, idx in col_max]
        # Now we need to check which type of logit has the actual max
        type_max_val, type_max_idx = torch.stack(maxs).max(0)
        if torch.isfinite(type_max_val).logical_not_().any():
            raise ValueError(
                "Non finite max value in sample", (type_max_val, self.logits)
            )

        # Now we can return the indices of where the actions occurred
        # in the form List[(type, row, column)]
        actions = []
        for i in range(type_max_idx.shape[0]):
            t = type_max_idx[i]
            # Subtract from the slice of that type and index, since the computed
            # row position is batch-wise rather graph-wise
            actions.append(
                (int(t), int(row_pos[t][i] - self.slice[t][i]), int(col_max[t][1][i]))
            )
        # It's now up to the Context class to create GraphBuildingAction instances
        # if it wants to convert these indices to env-compatible actions
        return actions

    def log_probability(self, actions: List[ActionIndex]) -> torch.Tensor:
        """The log-probability of a list of action tuples
        Parameters
        ----------
        actions: ActionIndex
            A list of action indices (action index triples)
        """
        log_probs = self.log_softmax()
        return torch.stack(
            [
                log_probs[t][row + self.slice[t][i], col]
                for i, (t, row, col) in enumerate(actions)
            ]
        )


def generate_forward_trajectory(
    g: Graph, max_nodes: int = None
) -> List[Tuple[Graph, Action]]:
    """Sample (uniformly) a trajectory that generates `g`"""
    # TODO: should this be a method of GraphBuildingEnv? handle set_node_attr flags and so on?
    gn = Graph()
    # Choose an arbitrary starting point, add to the stack
    stack: List[Tuple[int, ...]] = [(np.random.randint(0, len(g.nodes)),)]
    traj = []
    # This map keeps track of node labels in gn, since we have to start from 0
    relabeling_map: Dict[int, int] = {}
    while len(stack):
        # We pop from the stack until all nodes and edges have been
        # generated and their attributes have been set. Un-inserted
        # nodes/edges will be added to the stack as the graph is
        # expanded from the starting point. Nodes/edges that have
        # attributes will be reinserted into the stack until those
        # attributes are "set".
        i = stack.pop(np.random.randint(len(stack)))

        gt = gn.copy()  # This is a shallow copy
        if len(i) > 1:  # i is an edge
            e = relabeling_map.get(i[0], None), relabeling_map.get(i[1], None)
            if e in gn.edges:
                # i exists in the new graph, that means some of its attributes need to be added
                attrs = [j for j in g.edges[i] if j not in gn.edges[e]]
                if len(attrs) == 0:
                    continue  # If nodes are in cycles edges leading to them get stack multiple times, disregard
                attr = attrs[np.random.randint(len(attrs))]
                gn.edges[e][attr] = g.edges[i][attr]
                act = Action(
                    ActionType.SET_EDGE_ATTR,
                    source=e[0],
                    target=e[1],
                    attr=attr,
                    value=g.edges[i][attr],
                )
            else:
                # i doesn't exist, add the edge
                if e[1] not in gn.nodes:
                    # The endpoint of the edge is not in the graph, this is a AddNode action
                    assert e[1] is None  # normally we shouldn't have relabeled i[1] yet
                    relabeling_map[i[1]] = len(relabeling_map)
                    e = e[0], relabeling_map[i[1]]
                    gn.add_node(e[1], v=g.nodes[i[1]]["v"])
                    gn.add_edge(*e)
                    for j in g[i[1]]:  # stack unadded edges/neighbours
                        jp = relabeling_map.get(j, None)
                        if jp not in gn or (e[1], jp) not in gn.edges:
                            stack.append((i[1], j))
                    act = Action(
                        ActionType.ADD_NODE, source=e[0], value=g.nodes[i[1]]["v"]
                    )
                    if len(gn.nodes[e[1]]) < len(g.nodes[i[1]]):
                        stack.append(
                            (i[1],)
                        )  # we still have attributes to add to node i[1]
                else:
                    # The endpoint is in the graph, this is an AddEdge action
                    assert e[0] in gn.nodes
                    gn.add_edge(*e)
                    act = Action(ActionType.ADD_EDGE, source=e[0], target=e[1])

            if len(gn.edges[e]) < len(g.edges[i]):
                stack.append(i)  # we still have attributes to add to edge i
        else:  # i is a node, (u,)
            u = i[0]
            n = relabeling_map.get(u, None)
            if n not in gn.nodes:
                # u doesn't exist yet, this should only happen for the first node
                assert len(gn.nodes) == 0
                act = Action(ActionType.ADD_NODE, source=0, value=g.nodes[u]["v"])
                n = relabeling_map[u] = len(relabeling_map)
                gn.add_node(0, v=g.nodes[u]["v"])
                for j in g[u]:  # For every neighbour of node u
                    if relabeling_map.get(j, None) not in gn:
                        stack.append((u, j))  # push the (u,j) edge onto the stack
            else:
                # u exists, meaning we have attributes left to add
                attrs = [j for j in g.nodes[u] if j not in gn.nodes[n]]
                attr = attrs[np.random.randint(len(attrs))]
                gn.nodes[n][attr] = g.nodes[u][attr]
                act = Action(
                    ActionType.SET_NODE_ATTR,
                    source=n,
                    attr=attr,
                    value=g.nodes[u][attr],
                )
            if len(gn.nodes[n]) < len(g.nodes[u]):
                stack.append((u,))  # we still have attributes to add to node u
        traj.append((gt, act))
    traj.append((gn, Action(ActionType.STOP)))
    return traj


# Utility type-aliases
StateActionPair = Tuple[Graph, Action]
ActionIndex = Tuple[int, int, int]
