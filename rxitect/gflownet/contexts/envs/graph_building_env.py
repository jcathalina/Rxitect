# from networkx import Graph
import copy

from rxitect.gflownet.contexts.frag_graph_context import FragBasedGraphContext
from rxitect.gflownet.utils.graph import GraphAction, GraphActionType, Graph


class GraphBuildingEnv:
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
    def __init__(self, ctx: FragBasedGraphContext, allow_add_edge: bool = True, allow_node_attr: bool = True, allow_edge_attr: bool = True) -> None:
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
        self.ctx = ctx
        self.allow_add_edge = allow_add_edge
        self.allow_node_attr = allow_node_attr
        self.allow_edge_attr = allow_edge_attr

    @staticmethod
    def new() -> Graph:
        return Graph()

    def step(self, g: Graph, action: GraphAction) -> Graph:
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
        if action.action is GraphActionType.AddEdge:
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

        elif action.action is GraphActionType.AddNode:
            init_stems = copy.deepcopy(self.ctx.frags_stems[action.value])
            if len(g) == 0:
                assert action.source == 0  # TODO: this may not be useful
                gp.add_node(0, v=action.value,
                            frags=self.ctx.frags_smi[action.value],
                            stems=init_stems)
            else:
                assert action.source in g.nodes
                e = [action.source, max(g.nodes) + 1]

                assert not g.has_edge(*e)
                gp.add_node(e[1], v=action.value,
                            frags=self.ctx.frags_smi[action.value],
                            stems=init_stems)
                gp.add_edge(*e)

        elif action.action is GraphActionType.SetNodeAttr:
            assert self.allow_node_attr
            assert action.source in gp.nodes
            assert action.attr not in gp.nodes[action.source]
            gp.nodes[action.source][action.attr] = action.value

        elif action.action is GraphActionType.SetEdgeAttr:
            assert self.allow_edge_attr
            assert g.has_edge(action.source, action.target)
            assert action.attr not in gp.edges[(action.source, action.target)]
            gp.edges[(action.source, action.target)][action.attr] = action.value
        else:
            # TODO: backward actions if we want to support MCMC-GFN style algorithms
            raise ValueError(f'Unknown action type {action.action}')

        return gp
