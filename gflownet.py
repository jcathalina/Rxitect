from dataclasses import dataclass

import torch
from torch import nn
import torch_geometric.nn as gnn
from torch_geometric.data import Batch
from torch.nn import functional as F

from fragment_env import FragmentEnv


@dataclass(slots=True)
class FragmentGFlowNetOutput:
    mol_preds: torch.Tensor
    stem_preds: torch.Tensor


@dataclass(slots=True)
class FragmentGFlowNetPolicy:
    mol_policy: torch.Tensor
    stem_policy: torch.Tensor


class FragmentGFlowNet(nn.Module):
    def __init__(self, env: FragmentEnv, num_emb: int, out_per_stem: int, out_per_mol: int, num_conv_steps: int) -> None:
        super().__init__()
        self.frag_emb_layer = nn.Embedding(env.num_true_frags+1, num_emb)
        self.stem_emb_layer = nn.Embedding(env.num_stem_types+1, num_emb)
        self.bond_emb_layer = nn.Embedding(env.num_stem_types, num_emb)
        self.conv_layer = gnn.NNConv(num_emb, num_emb, nn.Sequential(), aggr="mean")
        self.rnn_layer = nn.GRU(num_emb, num_emb)

        self.frag2emb = nn.Sequential(nn.Linear(num_emb, num_emb), nn.LeakyReLU(), nn.Linear(num_emb, num_emb))
        self.stem2pred = nn.Sequential(nn.Linear(num_emb * 2, num_emb), nn.LeakyReLU(), nn.Linear(num_emb, num_emb), nn.LeakyReLU(), nn.Linear(num_emb, out_per_stem))
        self.global2pred = nn.Sequential(nn.Linear(num_emb, num_emb), nn.LeakyReLU(), nn.Linear(num_emb, out_per_mol))

        self.num_conv_steps = num_conv_steps
        self.num_emb = num_emb

    def forward(self, graph_batch: Batch) -> FragmentGFlowNetOutput:
        graph_batch.x = self.frag_emb_layer(graph_batch.x)
        graph_batch.stem_types = self.stem_emb_layer(graph_batch.stem_types)
        graph_batch.edge_attrs = self.bond_emb_layer(graph_batch.edge_attrs)

        graph_batch.edge_attrs = (
            graph_batch.edge_attrs[:, 0][:, :, None] * graph_batch.edge_attrs[:, 1][:, None, :]
        ).reshape((graph_batch.edge_index.shape[1], self.num_emb**2))

        out = graph_batch.x
        out: torch.Tensor = self.frag2emb(out)

        h = out.unsqueeze(dim=0)
        for _ in range(self.num_conv_steps):
            m = F.leaky_relu(self.conv_layer(out, graph_batch.edge_index, graph_batch.edge_attrs))
            out, h = self.rnn_layer(m.unsqueeze(dim=0).contiguous(), h.contiguous())
            out = out.squeeze(dim=0)

        # Index of the origin block of each stem in the batch (each stem is a pair [frag_idx, stem (atom) type],
        # we need to adjust for the batch packing). Batched data attrs that are being followed (here, "stem") gets an
        # additional field with "_batch" suffix, hence the "stems_batch" field.
        x_slices = (graph_batch._slice_dict['x']).clone().detach()[graph_batch.stems_batch]

        stem_block_batch_idx = (
                x_slices
                + graph_batch.stems[:, 0])

        stem_out_cat = torch.cat([out[stem_block_batch_idx], graph_batch.stem_types], 1)

        stem_preds = self.stem2pred(stem_out_cat)
        mol_preds = self.global2pred(gnn.global_mean_pool(out, graph_batch.batch))

        return FragmentGFlowNetOutput(mol_preds=mol_preds, stem_preds=stem_preds)

    @staticmethod
    def _get_policy(graph_batch: Batch, preds: FragmentGFlowNetOutput) -> FragmentGFlowNetPolicy:
        mol_exp = torch.exp(preds.mol_preds[:, 0])
        stem_exp = torch.exp(preds.stem_preds)

        Z = gnn.global_add_pool(x=stem_exp, batch=graph_batch.stems_batch).sum(1) + mol_exp + 1e-8
        return FragmentGFlowNetPolicy(mol_policy=(mol_exp / Z),
                                      stem_policy=(stem_exp / Z[graph_batch.stems_batch, None]),)

    @staticmethod
    def action_to_index(action: torch.Tensor, graph_batch: Batch, preds: FragmentGFlowNetOutput) -> torch.Tensor:
        stem_slices = (graph_batch._slice_dict["stems"][:-1]).clone().detach()
        a_: torch.Tensor = preds.stem_preds[stem_slices + action[:, 1]][torch.arange(action.shape[0]), action[:, 0]]
        b_ = a_ * (action[:, 0] >= 0) + preds.mol_preds * (action[:, 0] == -1)
        return b_

    def action_nll(self, action: torch.Tensor, graph_batch: Batch, preds: FragmentGFlowNetOutput):
        log_buffer = 1e-20  # small number to avoid log(0)
        policy = self._get_policy(graph_batch=graph_batch, preds=preds)

        mol_lsm = torch.log(policy.mol_policy + log_buffer)
        stem_lsm = torch.log(policy.stem_policy + log_buffer)
        log_preds = FragmentGFlowNetOutput(mol_preds=mol_lsm, stem_preds=stem_lsm)

        return -self.action_to_index(action=action, graph_batch=graph_batch, preds=log_preds)

    # def step(self, minibatch: MolMiniBatch, batch_idx):
    #     s, a, r, d, n, mols, idc, lens, *o = minibatch