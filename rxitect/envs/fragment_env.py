from typing import List

import networkx as nx

from rxitect.envs.contexts import Action, ActionType, Graph, StateActionPair


class FragmentEnv:
    """
    A Graph building environment which induces a DAG state space, compatible with GFlowNet.
    Supports forward and backward actions, with a `parents` function that list parents of
    forward actions.
    Edges and nodes can have attributes added to them in a key:value style.
    Edges and nodes are created with _implicit_ default attribute
    values (e.g. chirality, single/double bondness) so that:
        - an agent gets to do an extra action to set that attribute, but only
          if it is still default-valued (DAG property preserved)
        - we can generate a legal action for any attribute that isn't a default one.
    """

    def __init__(
        self,
        allow_add_edge: bool = True,
        allow_node_attr: bool = True,
        allow_edge_attr: bool = True,
    ):
        """A graph building environment instance
        Parameters
        ----------
        allow_add_edge: bool
            if True, allows this action and computes AddEdge parents (i.e. if False, this
            env only allows for tree generation)
        allow_node_attr: bool
            if True, allows this action and computes SetNodeAttr parents
        allow_edge_attr: bool
            if True, allows this action and computes SetEdgeAttr parents
        """
        self.allow_add_edge = allow_add_edge
        self.allow_node_attr = allow_node_attr
        self.allow_edge_attr = allow_edge_attr

    @staticmethod
    def new():
        return Graph()

    def step(self, g: Graph, action: Action) -> Graph:
        """Step forward the given graph state with an action
        Parameters
        ----------
        g: Graph
            the graph to be modified
        action: GraphAction
            the action taken on the graph, indices must match
        Returns
        -------
        gp: Graph
            the new graph
        """
        gp = g.copy()
        if action.act_type is ActionType.ADD_EDGE:
            a, b = action.source, action.target
            assert self.allow_add_edge
            assert a in g and b in g
            if a > b:
                a, b = b, a
            assert a != b
            assert not g.has_edge(a, b)
            # Ideally the FA underlying this must only be able to send
            # create_edge actions which respect this a<b property (or
            # its inverse!) , otherwise symmetry will be broken
            # because of the way the parents method is written
            gp.add_edge(a, b)

        elif action.act_type is ActionType.ADD_NODE:
            if len(g) == 0:
                assert action.source == 0  # TODO: this may not be useful
                gp.add_node(0, v=action.value)
            else:
                assert action.source in g.nodes
                e = [action.source, max(g.nodes) + 1]
                assert not g.has_edge(*e)
                gp.add_node(e[1], v=action.value)
                gp.add_edge(*e)

        elif action.act_type is ActionType.SET_NODE_ATTR:
            assert self.allow_node_attr
            assert action.source in gp.nodes
            assert action.attr not in gp.nodes[action.source]
            gp.nodes[action.source][action.attr] = action.value

        elif action.act_type is ActionType.SET_EDGE_ATTR:
            assert self.allow_edge_attr
            assert g.has_edge(action.source, action.target)
            assert action.attr not in gp.edges[(action.source, action.target)]
            gp.edges[(action.source, action.target)][action.attr] = action.value
        else:
            # TODO: backward actions if we want to support MCMC-GFN style algorithms
            raise ValueError(f"Unknown action type {action.act_type}", action.act_type)

        return gp

    def parents(self, g: Graph) -> List[StateActionPair]:
        """List possible parents of graph `g`
        Parameters
        ----------
        g: Graph
            graph
        Returns
        -------
        parents: List[Pair(GraphAction, Graph)]
            The list of parent-action pairs that lead to `g`.
        """
        raise NotImplementedError()

    @staticmethod
    def count_backward_transitions(g: Graph):
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
            if (
                deg[i] == 1
                and len(g.nodes[i]) == 1
                and len(g.edges[list(g.edges(i))[0]]) == 0
            ):
                c += 1
            c += len(g.nodes[i]) - 1  # One action per node attr, except 'v'
            if len(g.nodes) == 1 and len(g.nodes[i]) == 1:
                # special case if last node in graph
                c += 1
        return c


# TODO: Move these to a graph utils file
def graph_without_edge(g, e):
    gp = g.copy()
    gp.remove_edge(*e)
    return gp
