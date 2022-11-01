import unittest
from typing import Tuple, Callable, List, Dict, Union

import numpy as np
import torch
from rdkit.Chem import Mol
from torch import nn
from torch.distributions import Dirichlet
from torch.utils.data import Dataset

from rxitect.gflownet.algorithms.trajectory_balance import TrajectoryBalance
from rxitect.gflownet.contexts.envs.graph_building_env import GraphBuildingEnv
from rxitect.gflownet.contexts.frag_graph_context import FragBasedGraphContext
from rxitect.gflownet.models.frag_graph_gfn import FragBasedGraphGFN
from rxitect.gflownet.tasks.a import thermometer
from rxitect.gflownet.tasks.interfaces.graph_task import IGraphTask
from rxitect.gflownet.utils.multiproc import MPModelPlaceholder
from rxitect.gflownet.utils.types import FlatRewards


def sample_cond_info_uniformly():
    rng = np.random.RandomState(123)
    temperature_dist_params = (1, 192)
    n = 4
    beta = rng.uniform(*temperature_dist_params, n).astype(np.float32)
    upper_bound = temperature_dist_params[1]
    m = Dirichlet(torch.FloatTensor([1.] * 4))
    preferences = m.sample([n])
    beta_enc = thermometer(torch.tensor(beta), 32, 0, upper_bound)  # TODO: hyperparameters
    encoding = torch.cat([beta_enc, preferences], 1)
    return {'beta': torch.tensor(beta), 'encoding': encoding, 'preferences': preferences}


def encode_conditional_information(info, temperature_dist_params):
    encoding = torch.cat([torch.ones((len(info), 32)), info], 1)
    return {
        'beta': torch.ones(len(info)) * temperature_dist_params[-1],
        'encoding': encoding.float(),
        'preferences': info.float()
    }


class TestTrajectoryBalanceAlgorithm(unittest.TestCase):
    env = GraphBuildingEnv()
    ctx = FragBasedGraphContext()
    model = FragBasedGraphGFN(env_ctx=ctx, estimate_init_state_flow=True)
    hps = {
        'illegal_action_logreward': -75,
        'bootstrap_own_reward': False,
        'tb_epsilon': None,
        'reward_loss_multiplier': 1,
        'random_action_prob': 0.01,
    }
    tb = TrajectoryBalance(env=env, ctx=ctx, rng=np.random.RandomState(123), hps=hps, max_nodes=9)
    train_data = tb.create_training_data_from_own_samples(model=model, n=32, cond_info=torch.zeros(size=(32,)))
    print(train_data)

