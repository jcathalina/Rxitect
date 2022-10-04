from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch_geometric.data import Batch

from rxitect.envs.contexts import ActionCategorical
from rxitect.models.transformers import GraphTransformer, create_mlp

if TYPE_CHECKING:
    from rxitect.envs import FragmentEnvContext


class FragmentBasedGFN(nn.Module):
    """GraphTransformer class for a GFlowNet which outputs a GraphActionCategorical. Meant for
    fragment-wise generation.
    Outputs logits for the following actions
    - STOP
    - ADD_NODE
    - SET_EDGE_ATTR
    """

    def __init__(
        self,
        ctx: FragmentEnvContext,
        num_emb: int = 64,
        num_layers: int = 3,
        num_heads: int = 2,
    ):
        """
        Parameters
        ----------
        ctx: FragmentEnvContext
            x
        num_emb: int
            x
        num_layers: int
            x
        num_heads: int
            x
        """
        super().__init__()
        self.transformer = GraphTransformer(
            x_dim=ctx.num_node_dim,
            e_dim=ctx.num_edge_dim,
            g_dim=ctx.num_cond_dim,
            num_emb=num_emb,
            num_layers=num_layers,
            num_heads=num_heads,
        )
        num_final = num_emb * 2
        num_mlp_layers = 0
        self.emb2add_node = create_mlp(
            num_final, num_emb, ctx.num_new_node_values, num_mlp_layers
        )
        # Edge attr logits are "sided", so we will compute both sides independently
        self.emb2set_edge_attr = create_mlp(
            num_emb + num_final, num_emb, ctx.num_edge_attr_logits // 2, num_mlp_layers
        )
        self.emb2stop = create_mlp(num_emb * 3, num_emb, 1, num_mlp_layers)
        self.emb2reward = create_mlp(num_emb * 3, num_emb, 1, num_mlp_layers)
        self.edge2emb = create_mlp(num_final, num_emb, num_emb, num_mlp_layers)
        self.log_z = create_mlp(ctx.num_cond_dim, num_emb * 2, 1, 2)
        self.action_type_order = ctx.action_type_order

    def forward(self, g: Batch, cond: torch.Tensor):
        """See `GraphTransformer` for argument values"""
        node_embeddings, graph_embeddings = self.transformer(g, cond)
        # On `::2`, edges are duplicated to make graphs undirected, only take the even ones
        e_row, e_col = g.edge_index[:, ::2]
        edge_emb = self.edge2emb(node_embeddings[e_row] + node_embeddings[e_col])
        src_anchor_logits = self.emb2set_edge_attr(
            torch.cat([edge_emb, node_embeddings[e_row]], 1)
        )
        dst_anchor_logits = self.emb2set_edge_attr(
            torch.cat([edge_emb, node_embeddings[e_col]], 1)
        )

        def _mask(x, m):
            # mask logit vector x with binary mask m, -1000 is a tiny log-value
            return x * m + -1000 * (1 - m)

        cat = ActionCategorical(
            g,
            logits=[
                self.emb2stop(graph_embeddings),
                _mask(self.emb2add_node(node_embeddings), g.add_node_mask),
                _mask(
                    torch.cat([src_anchor_logits, dst_anchor_logits], 1),
                    g.set_edge_attr_mask,
                ),
            ],
            keys=[None, "x", "edge_index"],
            types=self.action_type_order,
        )
        return cat, self.emb2reward(graph_embeddings)


if __name__ == "__main__":
    env = FragmentEnvContext()
    gfn = FragmentBasedGFN(ctx=env)
    # gfn.
