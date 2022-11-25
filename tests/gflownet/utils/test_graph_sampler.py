import unittest

import numpy as np
import torch

from rxitect.gflownet.contexts.envs.graph_building_env import GraphBuildingEnv
from rxitect.gflownet.contexts.frag_graph_context import FragBasedGraphContext
from rxitect.gflownet.models.frag_graph_gfn import FragBasedGraphGFN
from rxitect.gflownet.utils.graph import GraphActionType, GraphAction
from rxitect.gflownet.utils.graph_sampler import GraphSampler


class TestGraphSampling(unittest.TestCase):

    def test_that_valid_molecules_are_built_by_sampling(self):
        env = GraphBuildingEnv()
        ctx = FragBasedGraphContext(max_frags=9, num_cond_dim=32)
        sampler = GraphSampler(ctx=ctx, env=env, max_nodes=9, max_steps=128, rng=np.random.RandomState(123))
        model = FragBasedGraphGFN(ctx=ctx, estimate_init_state_flow=True)
        s = sampler.sample_from_model(model=model, n=1, cond_info=torch.zeros(1), device=torch.device("cpu"))
        print(s)


if __name__ == "__main__":
    unittest.main()
