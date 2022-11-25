from typing import List, Tuple

import networkx as nx
import pandas as pd
import torch
from networkx import Graph
from pyprojroot import here
from rdkit import Chem
from rdkit.Chem import Mol
from torch_geometric.data import Data, Batch

from rxitect.gflownet.contexts.frag_graph_context import FragBasedGraphContext
from rxitect.gflownet.contexts.interfaces.graph_context import IGraphContext
from rxitect.gflownet.utils.graph import GraphAction, GraphActionType


def env_step(g: nx.Graph, action: GraphAction) -> nx.Graph:
    gp = g.copy()
    if action.action is GraphActionType.AddEdge:
        a, b = action.source, action.target
        if a > b:
            a, b = b, a
        gp.add_edge(a, b)

    elif action.action is GraphActionType.AddNode:
        if len(g) == 0:
            gp.add_node(0, v=action.value)
        else:
            e = [action.source, max(g.nodes) + 1]

            gp.add_node(e[1], v=action.value)
            gp.add_edge(*e)

    elif action.action is GraphActionType.SetEdgeAttr:
        gp.edges[(action.source, action.target)][action.attr] = action.value
    else:
        raise ValueError(f'Unknown action type {action.action}')

    return gp


def main():
    ctx = FragBasedGraphContext(max_frags=5)
    G = nx.Graph()
    G.add_node(0, v=0, stems=ctx.frags_stems[0])
    e = (0, 1)
    G.add_node(e[1], v=92, stems=ctx.frags_stems[92])
    G.add_edge(*e)
    e = (1, 2)  # This should not work because fragment 92 only had 1 stem
    G.add_node(e[1], v=90, stems=ctx.frags_stems[90])
    G.add_edge(*e)

    import matplotlib.pyplot as plt
    labels = nx.get_node_attributes(G, 'stems')
    # labels = {k: f"{ctx.frags_smi[v]}\n{ctx.frags_stems[v]}" for k, v in labels.items()}
    nx.draw(G, labels=labels, node_size=1000)
    plt.show()

    # aaa = nx.get_edge_attributes(G)
    aaa = list(list(G.edges(data=True))[0][-1].keys())

    M = ctx.graph_to_mol(g=G)
    Chem.MolToSmiles(M, sanitize=False)
    assert not ctx.is_sane(G)


if __name__ == "__main__":
    main()
