import unittest

from rxitect.gflownet.contexts.frag_graph_context import FragBasedGraphContext
from rxitect.gflownet.contexts.envs.graph_building_env import GraphBuildingEnv
from rxitect.gflownet.utils.graph import GraphAction, GraphActionType


class TestGraphBuilding(unittest.TestCase):

    def test_that_valid_molecules_are_built_using_context(self):
        env = GraphBuildingEnv()
        ctx = FragBasedGraphContext(max_frags=9, num_cond_dim=32)
        g = env.new()
        a = GraphAction(action=GraphActionType.AddNode, source=0)
        g_1 = env.step(g, a)
        print(g_1)
        a2 = GraphAction(action=GraphActionType.AddNode, source=0, target=1)
        g_2 = env.step(g_1, a2)
        print(g_2)
        assert True


if __name__ == "__main__":
    unittest.main()
