import copy
import enum
from typing import List, Tuple, Optional, Dict

import networkx as nx
import numpy as np
import torch
# from networkx import Graph
from torch import nn
from torch_geometric.data import Batch
from torch_scatter import scatter, scatter_max


class Graph(nx.Graph):
    # Subclassing nx.Graph for debugging purposes
    def __str__(self):
        return repr(self)

    def __repr__(self):
        return f'<{list(self.nodes)}, {list(self.edges)}, {list(self.nodes[i]["v"] for i in self.nodes)}, {list(self.nodes[i]["frags"] for i in self.nodes)}, {list(self.nodes[i]["stems"] for i in self.nodes)}>'


class GraphActionType(enum.Enum):
    """All types of actions that can be taken on the graphs defined within this project"""
    Stop = enum.auto()
    AddNode = enum.auto()
    AddEdge = enum.auto()
    SetNodeAttr = enum.auto()
    SetEdgeAttr = enum.auto()


class GraphAction:
    def __init__(self, action: GraphActionType, source: int = None, target: int = None, value: int = None, attr: str = None) -> None:
        """A single graph-building action
        Parameters
        ----------
        action: GraphActionType
            the action type
        source: int
            the source node this action is applied on
        target: int, optional
            the target node (i.e. if specified this is an edge action)
        attr: str, optional
            the set attribute of a node/edge
        value: Any, optional
            the value (e.g. new node type) applied
        """
        self.action = action
        self.source = source
        self.target = target
        self.attr = attr
        self.value = value

    def __repr__(self) -> str:
        attrs = ', '.join(str(i) for i in [self.source, self.target, self.attr, self.value] if i is not None)
        return f"<{self.action}, {attrs}>"


class GraphActionCategorical:
    def __init__(self, graphs: Batch, logits: List[torch.Tensor], keys: List[str], types: List[GraphActionType],
                 deduplicate_edge_index: bool = True, masks: Optional[List[torch.Tensor]] = None) -> None:
        """A multi-type Categorical compatible with generating structured actions.
        What is meant by type here is that there are multiple types of
        mutually exclusive actions, e.g. AddNode and AddEdge are
        mutually exclusive, but since their logits will be produced by
        different variable-sized tensors (corresponding to different
        elements of the graph, e.g. nodes or edges) it is inconvenient
        to stack them all into one single Categorical. This class
        provides this convenient interaction between torch_geometric
        Batch objects and lists of logit tensors.
        Parameters
        ----------
        graphs: Batch
            A Batch of graphs to which the logits correspond
        logits: List[Tensor]
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
        types: List[GraphActionType]
           The action type each logit corresponds to.
        deduplicate_edge_index: bool, default=True
           If true, this means that the 'edge_index' keys have been reduced
           by e_i[::2] (presumably because the graphs are undirected)
        masks: List[Tensor], default=None
           If not None, a list of broadcastable tensors that multiplicative
           mask out logits of invalid actions
        """
        self.num_graphs = graphs.num_graphs
        # The logits
        self.logits = logits
        self.types = types
        self.keys = keys
        self.dev = dev = graphs.x.device
        # TODO: mask is only used by graph_sampler, but maybe we should be more careful with it
        # (e.g. in a softmax and such)
        # Can be set to indicate which logits are masked out (shape must match logits or have
        # broadcast dimensions already set)
        self.masks = masks

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
            getattr(graphs, f'{k}_batch' if k != 'x' else 'batch') if k is not None
            # None signals a global logit rather than a per-instance logit
            else torch.arange(graphs.num_graphs, device=dev) for k in keys
        ]
        # This is the cumulative sum (prefixed by 0) of N[i]s
        self.slice = [
            graphs._slice_dict[k] if k is not None else torch.arange(graphs.num_graphs, device=dev) for k in keys
        ]
        self.logprobs = None

        if deduplicate_edge_index and 'edge_index' in keys:
            idx = keys.index('edge_index')
            self.batch[idx] = self.batch[idx][::2]
            self.slice[idx] = self.slice[idx].div(2, rounding_mode='floor')

    def detach(self):
        new = copy.copy(self)
        new.logits = [i.detach() for i in new.logits]
        if new.logprobs is not None:
            new.logprobs = [i.detach() for i in new.logprobs]
        return new

    def to(self, device):
        self.dev = device
        self.logits = [i.to(device) for i in self.logits]
        self.batch = [i.to(device) for i in self.batch]
        self.slice = [i.to(device) for i in self.slice]
        if self.logprobs is not None:
            self.logprobs = [i.to(device) for i in self.logprobs]
        return self

    def logsoftmax(self):
        """Compute log-probabilities given logits"""
        if self.logprobs is not None:
            return self.logprobs
        # Use the `subtract by max` trick to avoid precision errors:
        # compute max
        maxl = torch.cat(
            [scatter(i, b, dim=0, dim_size=self.num_graphs, reduce='max') for i, b in zip(self.logits, self.batch)],
            dim=1).max(1).values.detach()
        # substract by max then take exp
        # x[b, None] indexes by the batch to map back to each node/edge and adds a broadcast dim
        exp_logits = [(i - maxl[b, None]).exp() + 1e-40 for i, b in zip(self.logits, self.batch)]
        # sum corrected exponentiated logits, to get log(Z - max) = log(sum(exp(logits)) - max)
        logZ = sum([
            scatter(i, b, dim=0, dim_size=self.num_graphs, reduce='sum').sum(1) for i, b in zip(exp_logits, self.batch)
        ]).log()
        # log probabilities is log(exp(logit) / Z)
        self.logprobs = [i.log() - logZ[b, None] for i, b in zip(exp_logits, self.batch)]
        return self.logprobs

    def logsumexp(self, x: Optional[List[torch.Tensor]] = None) -> float:
        """Reduces `x` (the logits by default) to one scalar per graph"""
        if x is None:
            x = self.logits
        # Use the `subtract by max` trick to avoid precision errors:
        # compute max
        maxl = torch.cat([scatter(i, b, dim=0, dim_size=self.num_graphs, reduce='max') for i, b in zip(x, self.batch)],
                         dim=1).max(1).values.detach()
        # substract by max then take exp
        # x[b, None] indexes by the batch to map back to each node/edge and adds a broadcast dim
        exp_vals = [(i - maxl[b, None]).exp() + 1e-40 for i, b in zip(x, self.batch)]
        # sum corrected exponentiated logits, to get log(Z - max) = log(sum(exp(logits)) - max)
        reduction = sum([
            scatter(i, b, dim=0, dim_size=self.num_graphs, reduce='sum').sum(1) for i, b in zip(exp_vals, self.batch)
        ]).log()
        # Add back max
        return reduction + maxl

    def sample(self) -> List[Tuple[int, int, int]]:
        """Samples this categorical
        Returns
        -------
        actions: List[Tuple[int, int, int]]
            A list of indices representing [action type, element index, action index]. See constructor.
        """
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
        # Take the argmax
        return self.argmax(x=gumbel)

    def argmax(self, x: List[torch.Tensor], batch: List[torch.Tensor] = None,
               dim_size: int = None) -> List[Tuple[int, int, int]]:
        """Takes the argmax, i.e. if x are the logits, returns the most likely action.
        Parameters
        ----------
        x: List[Tensor]
            Tensors in the same format as the logits (see constructor).
        batch: List[Tensor]
            Tensors in the same format as the batch indices of torch_geometric, default `self.batch`.
        dim_size: int
            The reduction dimension, default `self.num_graphs`.
        Returns
        -------
        actions: List[Tuple[int, int, int]]
            A list of indices representing [action type, element index, action index]. See constructor.
        """
        # scatter_max and .max create a (values, indices) pair
        # These logits are 2d (num_obj_of_type, num_actions_of_type),
        # first reduce-max over the batch, which preserves the
        # columns, so we get (minibatch_size, num_actions_of_type).
        # First we prefill `out` with very negative values in case
        # there are no corresponding logits (this can happen if e.g. a
        # graph has no edges), we don't want to accidentally take the
        # max of that type.
        if batch is None:
            batch = self.batch
        if dim_size is None:
            dim_size = self.num_graphs
        mnb_max = [torch.zeros(dim_size, i.shape[1], device=self.dev) - 1e6 for i in x]
        mnb_max = [scatter_max(i, b, dim=0, out=out) for i, b, out in zip(x, batch, mnb_max)]
        # Then over cols, this gets us which col holds the max value,
        # so we get (minibatch_size,)
        col_max = [values.max(1) for values, idx in mnb_max]
        # Now we look up which row in those argmax cols was the max:
        row_pos = [idx_mnb[torch.arange(len(idx_col)), idx_col] for (_, idx_mnb), (_, idx_col) in zip(mnb_max, col_max)]
        # The maxes themselves
        maxs = [values for values, idx in col_max]
        # Now we need to check which type of logit has the actual max
        type_max_val, type_max_idx = torch.stack(maxs).max(0)
        if torch.isfinite(type_max_val).logical_not_().any():
            raise ValueError('Non finite max value in sample', (type_max_val, x))

        # Now we can return the indices of where the actions occurred
        # in the form List[(type, row, column)]
        assert dim_size == type_max_idx.shape[0]
        argmaxes = []
        for i in range(type_max_idx.shape[0]):
            t = type_max_idx[i]
            # Subtract from the slice of that type and index, since the computed
            # row position is batch-wise rather graph-wise
            argmaxes.append((int(t), int(row_pos[t][i] - self.slice[t][i]), int(col_max[t][1][i])))
        # It's now up to the Context class to create GraphBuildingAction instances
        # if it wants to convert these indices to env-compatible actions
        return argmaxes

    def log_prob(self, actions: List[Tuple[int, int, int]], logprobs: torch.Tensor = None) -> torch.Tensor:
        """The log-probability of a list of action tuples, effectively indexes `logprobs` using internal
        slice indices.
        Parameters
        ----------
        actions: List[Tuple[int, int, int]]
            A list of n action tuples denoting indices
        logprobs: List[Tensor]
            The log-probablities to be indexed (self.logsoftmax() by default) in order (i.e. this
            assumes there are n graphs represented by this object).

        Returns
        -------
        log_prob: Tensor
            The log probability of each action.
        """
        if logprobs is None:
            logprobs = self.logsoftmax()
            # FIXME: breaks when actions contain nonsense, e.g. (0, 1, 0), stop
            #   action is supposed to be (0,0,0) no matter what.
            #   This happens when too many illegal set edge attrs happen and we
            #   have a model that outputs NaNs for logits :(
        return torch.stack([logprobs[t][row + self.slice[t][i], col] for i, (t, row, col) in enumerate(actions)])

    def entropy(self, logprobs=None):
        """The entropy for each graph categorical in the batch
        Parameters
        ----------
        logprobs: List[Tensor]
            The log-probablities of the policy (self.logsoftmax() by default)
        Returns
        -------
        entropies: Tensor
            The entropy for each graph categorical in the batch
        """
        if logprobs is None:
            logprobs = self.logsoftmax()
        entropy = -sum([
            scatter(i * i.exp(), b, dim=0, dim_size=self.num_graphs, reduce='sum').sum(1)
            for i, b in zip(logprobs, self.batch)
        ])
        return entropy


def generate_forward_trajectory(g: Graph) -> List[Tuple[Graph, GraphAction]]:
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
        # generated and their attributes have been set. Uninserted
        # nodes/edges will be added to the stack as the graph is
        # expanded from the starting point. Nodes/edges that have
        # attributes will be reinserted into the stack until those
        # attributes are "set".
        i = stack.pop(np.random.randint(len(stack)))

        gt = gn.copy()  # This is a shallow copy
        if len(i) > 1:  # i is an edge
            e = relabeling_map.get(i[0], None), relabeling_map.get(i[1], None)
            if e in gn.edges:
                # 'i' exists in the new graph, that means some of its attributes need to be added
                attrs = [j for j in g.edges[i] if j not in gn.edges[e]]
                if len(attrs) == 0:
                    continue  # If nodes are in cycles edges leading to them get stack multiple times, disregard
                attr = attrs[np.random.randint(len(attrs))]
                gn.edges[e][attr] = g.edges[i][attr]
                act = GraphAction(GraphActionType.SetEdgeAttr, source=e[0], target=e[1], attr=attr,
                                  value=g.edges[i][attr])
            else:
                # i doesn't exist, add the edge
                if e[1] not in gn.nodes:
                    # The endpoint of the edge is not in the graph, this is a AddNode action
                    assert e[1] is None  # normally we shouldn't have relabeled i[1] yet
                    relabeling_map[i[1]] = len(relabeling_map)
                    e = e[0], relabeling_map[i[1]]
                    gn.add_node(e[1], v=g.nodes[i[1]]['v'])
                    gn.add_edge(*e)
                    for j in g[i[1]]:  # stack unadded edges/neighbours
                        jp = relabeling_map.get(j, None)
                        if jp not in gn or (e[1], jp) not in gn.edges:
                            stack.append((i[1], j))
                    act = GraphAction(GraphActionType.AddNode, source=e[0], value=g.nodes[i[1]]['v'])
                    if len(gn.nodes[e[1]]) < len(g.nodes[i[1]]):
                        stack.append((i[1],))  # we still have attributes to add to node i[1]
                else:
                    # The endpoint is in the graph, this is an AddEdge action
                    assert e[0] in gn.nodes
                    gn.add_edge(*e)
                    act = GraphAction(GraphActionType.AddEdge, source=e[0], target=e[1])

            if len(gn.edges[e]) < len(g.edges[i]):
                stack.append(i)  # we still have attributes to add to edge i
        else:  # i is a node, (u,)
            u = i[0]
            n = relabeling_map.get(u, None)
            if n not in gn.nodes:
                # u doesn't exist yet, this should only happen for the first node
                assert len(gn.nodes) == 0
                act = GraphAction(GraphActionType.AddNode, source=0, value=g.nodes[u]['v'])
                n = relabeling_map[u] = len(relabeling_map)
                gn.add_node(0, v=g.nodes[u]['v'])
                for j in g[u]:  # For every neighbour of node u
                    if relabeling_map.get(j, None) not in gn:
                        stack.append((u, j))  # push the (u,j) edge onto the stack
            else:
                # u exists, meaning we have attributes left to add
                attrs = [j for j in g.nodes[u] if j not in gn.nodes[n]]
                attr = attrs[np.random.randint(len(attrs))]
                gn.nodes[n][attr] = g.nodes[u][attr]
                act = GraphAction(GraphActionType.SetNodeAttr, source=n, attr=attr, value=g.nodes[u][attr])
            if len(gn.nodes[n]) < len(g.nodes[u]):
                stack.append((u,))  # we still have attributes to add to node u
        traj.append((gt, act))
    traj.append((gn, GraphAction(GraphActionType.Stop)))
    return traj


def count_backward_transitions(g: Graph) -> int:
    """Counts the number of parents of g without checking for isomorphisms"""
    c = 0
    deg = [g.degree[i] for i in range(len(g.nodes))]
    for a, b in g.edges:
        if deg[a] > 1 and deg[b] > 1 and len(g.edges[(a, b)]) == 0:
            # Can only remove edges connected to non-leaves and without
            # attributes (the agent has to remove the attrs, then remove
            # the edge). Removal cannot disconnect the graph.
            new_g = graph_without_edge(g, (a, b))
            if nx.algorithms.is_connected(new_g):
                c += 1
        c += len(g.edges[(a, b)])  # One action per edge attr
    for i in g.nodes:
        if deg[i] == 1 and len(g.nodes[i]) == 1 and len(g.edges[list(g.edges(i))[0]]) == 0:
            c += 1
        c += len(g.nodes[i]) - 1  # One action per node attr, except 'v'
        if len(g.nodes) == 1 and len(g.nodes[i]) == 1:
            # special case if last node in graph
            c += 1
    return c


def graph_without_edge(g, e):
    g_copy = g.copy()
    g_copy.remove_edge(*e)
    return g_copy


def graph_without_node(g, n):
    g_copy = g.copy()
    g_copy.remove_node(n)
    return g_copy


def graph_without_node_attr(g, n, a):
    g_copy = g.copy()
    del g_copy.nodes[n][a]
    return g_copy


def graph_without_edge_attr(g, e, a):
    g_copy = g.copy()
    del g_copy.edges[e][a]
    return g_copy
