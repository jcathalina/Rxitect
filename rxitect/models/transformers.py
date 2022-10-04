from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch
import torch_geometric.nn as gnn
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.utils import add_self_loops


class GraphTransformer(nn.Module):
    """An agnostic GraphTransformer class, and the main model used by other model classes
    This graph model takes in node features, edge features, and graph features (referred to as
    conditional information, since they condition the output). The graph features are projected to
    virtual nodes (one per graph), which are fully connected.
    The per node outputs are the concatenation of the final (post graph-convolution) node embeddings
    and of the final virtual node embedding of the graph each node corresponds to.
    The per graph outputs are the concatenation of a global mean pooling operation, of the final
    virtual node embeddings, and of the conditional information embedding.
    """

    def __init__(
        self,
        x_dim: int,
        e_dim: int,
        g_dim: int,
        num_emb: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
    ):
        """
        Parameters
        ----------
        x_dim: int
            The number of node features
        e_dim: int
            The number of edge features
        g_dim: int
            The number of graph-level features
        num_emb: int
            The number of hidden dimensions, i.e. embedding size. Default 64.
        num_layers: int
            The number of Transformer layers.
        num_heads: int
            The number of Transformer heads per layer.
        """
        super().__init__()
        self.num_layers = num_layers

        self.x2h = create_mlp(x_dim, num_emb, num_emb, 2)
        self.e2h = create_mlp(e_dim, num_emb, num_emb, 2)
        self.c2h = create_mlp(g_dim, num_emb, num_emb, 2)
        self.graph2emb = nn.ModuleList(
            sum(
                [
                    [
                        gnn.GENConv(
                            num_emb, num_emb, num_layers=1, aggr="add", norm=None
                        ),
                        gnn.TransformerConv(
                            num_emb * 2, num_emb, edge_dim=num_emb, heads=num_heads
                        ),
                        nn.Linear(num_heads * num_emb, num_emb),
                        gnn.LayerNorm(num_emb, affine=False),
                        create_mlp(num_emb, num_emb * 4, num_emb, 1),
                        gnn.LayerNorm(num_emb, affine=False),
                    ]
                    for i in range(self.num_layers)
                ],
                [],
            )
        )

    def forward(self, g: Batch, cond: torch.Tensor):
        """Forward pass
        Parameters
        ----------
        g: Batch
            A standard torch_geometric Batch object. Expects `edge_attr` to be set.
        cond: torch.Tensor
            The per-graph conditioning information. Shape: (g.num_graphs, self.g_dim).
        Returns
        -------
        node_embeddings: torch.Tensor
            Per node embeddings. Shape: (g.num_nodes, self.num_emb * 2).
        graph_embeddings: torch.Tensor
            Per graph embeddings. Shape: (g.num_graphs, self.num_emb * 3).
        """
        o = self.x2h(g.x)
        e = self.e2h(g.edge_attr)
        c = self.c2h(cond)
        num_total_nodes = g.x.shape[0]
        # Augment the edges with a new edge to the conditioning
        # information node. This new node is connected to every node
        # within its graph.
        u, v = torch.arange(num_total_nodes, device=o.device), g.batch + num_total_nodes
        aug_edge_index = torch.cat(
            [g.edge_index, torch.stack([u, v]), torch.stack([v, u])], 1
        )
        e_p = torch.zeros((num_total_nodes * 2, e.shape[1]), device=g.x.device)
        e_p[:, 0] = 1  # Manually create a bias term
        aug_e = torch.cat([e, e_p], 0)
        aug_edge_index, aug_e = add_self_loops(aug_edge_index, aug_e, "mean")
        aug_batch = torch.cat([g.batch, torch.arange(c.shape[0], device=o.device)], 0)

        # Append the conditioning information node embedding to o
        o = torch.cat([o, c], 0)
        for i in range(self.num_layers):
            # Run the graph transformer forward
            gen, trans, linear, norm1, ff, norm2 = self.graph2emb[i * 6 : (i + 1) * 6]
            agg = gen(o, aug_edge_index, aug_e)
            o = norm1(
                o + linear(trans(torch.cat([o, agg], 1), aug_edge_index, aug_e)),
                aug_batch,
            )
            o = norm2(o + ff(o), aug_batch)

        glob = torch.cat(
            [gnn.global_mean_pool(o[: -c.shape[0]], g.batch), o[-c.shape[0] :], c], 1
        )
        o_final = torch.cat([o[: -c.shape[0]], c[g.batch]], 1)
        return o_final, glob


def create_mlp(
    n_in: int,
    n_hid: int,
    n_out: int,
    n_layer: int,
    activation_fn: Callable = nn.LeakyReLU,
):
    """Helper function that creates a fully-connected network with no activation after the last layer.
    If `n_layer` is 0 then this corresponds to `nn.Linear(n_in, n_out)`.

    Parameters
    ----------
    n_in: int
        x
    n_hid: int
        x
    n_out: int
        x
    n_layer: int
        x
    activation_fn: Callable
        x
    """
    n = [n_in] + [n_hid] * n_layer + [n_out]
    return nn.Sequential(
        *sum(
            [[nn.Linear(n[i], n[i + 1]), activation_fn()] for i in range(n_layer + 1)],
            [],
        )[:-1]
    )
