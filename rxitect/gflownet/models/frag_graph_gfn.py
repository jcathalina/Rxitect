from typing import Tuple

import torch
from torch_geometric.data import Batch

from rxitect.gflownet.contexts.frag_graph_context import FragBasedGraphContext
from rxitect.gflownet.models.components.common import mlp
from rxitect.gflownet.models.components.graph_transformer import GraphTransformer
from torch import nn

from rxitect.gflownet.utils.graph import GraphActionCategorical


class FragBasedGraphGFN(nn.Module):
    """GraphTransformer class for a GFlowNet which outputs a GraphActionCategorical. Meant for
    fragment-wise generation.
    Outputs logits for the following actions
    - Stop
    - AddNode
    - SetEdgeAttr
    """
    def __init__(self, env_ctx: FragBasedGraphContext, num_emb: int = 64, num_layers: int = 3, num_heads: int = 2, estimate_init_state_flow: bool = False) -> None:
        """
        Parameters
        ----------
        env_ctx:
            The context to use for graph-building
        num_emb:
            Number of embeddings
        num_layers:
            Number of layers in the network
        num_heads:
            Number of output prediction heads used for the transformer
        estimate_init_state_flow:
            If the model should include an inner model that estimates log(Z), where Z = F(s_0), the initial state flow
        """
        super().__init__()
        self.transf = GraphTransformer(x_dim=env_ctx.num_node_dim, e_dim=env_ctx.num_edge_dim,
                                       g_dim=env_ctx.num_cond_dim, num_emb=num_emb, num_layers=num_layers,
                                       num_heads=num_heads)
        num_final = num_emb * 2
        num_mlp_layers = 0
        self.emb2add_node = mlp(num_final, num_emb, env_ctx.num_new_node_values, num_mlp_layers)
        # Edge attr logits are "sided", so we will compute both sides independently
        self.emb2set_edge_attr = mlp(num_emb + num_final, num_emb, env_ctx.num_edge_attr_logits // 2, num_mlp_layers)
        self.emb2stop = mlp(num_emb * 3, num_emb, 1, num_mlp_layers)
        self.emb2reward = mlp(num_emb * 3, num_emb, 1, num_mlp_layers)
        self.edge2emb = mlp(num_final, num_emb, num_emb, num_mlp_layers)
        self.action_type_order = env_ctx.action_type_order
        if estimate_init_state_flow:
            self.logZ = mlp(env_ctx.num_cond_dim, num_emb * 2, 1, 2)

    def forward(self, g: Batch, cond: torch.Tensor) -> Tuple[GraphActionCategorical, torch.Tensor]:
        """See `GraphTransformer` for argument values

        Parameters
        ----------
        g:
            Batched pytorch geometric data (graphs)
        cond:
            Tensor containing embedded conditioning information

        Returns
        -------
        Tuple[GraphActionCategorical, Tensor]:
            A tuple containing a forward categorical to sample an action from, and a tensor containg reward preds
        """
        node_embeddings, graph_embeddings = self.transf(g, cond)
        # On `::2`, edges are duplicated to make graphs undirected, only take the even ones
        e_row, e_col = g.edge_index[:, ::2]
        edge_emb = self.edge2emb(node_embeddings[e_row] + node_embeddings[e_col])
        src_anchor_logits = self.emb2set_edge_attr(torch.cat([edge_emb, node_embeddings[e_row]], 1))
        dst_anchor_logits = self.emb2set_edge_attr(torch.cat([edge_emb, node_embeddings[e_col]], 1))

        def _mask(x, m):
            # mask logit vector x with binary mask m, -1000 is a tiny log-value
            return x * m + -1000 * (1 - m)

        cat = GraphActionCategorical(
            g,
            logits=[
                self.emb2stop(graph_embeddings),
                _mask(self.emb2add_node(node_embeddings), g.add_node_mask),
                _mask(torch.cat([src_anchor_logits, dst_anchor_logits], 1), g.set_edge_attr_mask),
            ],
            # FIXME: NaN bug happening at logit calc for some reason, not sure what's causing it. check if masks are sane?
            #   UPDATE: Figured it out, was the set_edge_attr mask... However, if we don't adjust for stem it does not
            #   always make legal attachment points when set_edge_attr action happens.
            keys=[None, 'x', 'edge_index'],
            types=self.action_type_order,
            masks=[torch.ones(1), g.add_node_mask.cpu(),
                   g.set_edge_attr_mask.cpu()],
        )
        return cat, self.emb2reward(graph_embeddings)
