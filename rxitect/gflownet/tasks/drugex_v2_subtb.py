import ast
import copy
from typing import Dict, Any, List, Tuple, Union, Callable

import numpy as np
import torch
from pyprojroot import here
from rdkit import RDLogger
from rdkit.Chem import Mol
from torch import nn
from torch.distributions import Dirichlet
from torch.utils.data import Dataset

from rxitect.gflownet.algorithms.sub_trajectory_balance import SubTrajectoryBalance
from rxitect.gflownet.base_trainer import BaseTrainer
from rxitect.gflownet.contexts.envs.graph_building_env import GraphBuildingEnv
from rxitect.gflownet.contexts.frag_graph_context import FragBasedGraphContext
from rxitect.gflownet.hooks.multiobjective_hooks import MultiObjectiveStatsHook, TopKHook
from rxitect.gflownet.models.frag_graph_gfn import FragBasedGraphGFN
from rxitect.gflownet.tasks.a import RepeatedPreferenceDataset, thermometer
from rxitect.gflownet.tasks.interfaces.graph_task import IGraphTask
from rxitect.gflownet.utils import metrics
from rxitect.gflownet.utils.multiproc import MPModelPlaceholder
from rxitect.gflownet.utils.types import FlatRewards
from rxitect.scorers.a2ascore import Predictor

NUM_PREFS = 3


class DrugExV2Task(IGraphTask):
    def __init__(self, dataset: Dataset, temperature_distribution: str, temperature_parameters: Tuple[float],
                 wrap_model: Callable[[nn.Module], nn.Module] = None) -> None:
        self._wrap_model = wrap_model
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.seeded_preference = None
        self.experimental_dirichlet = False
        self.rng = None

    def flat_reward_transform(self, y: Union[float, torch.Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def _load_task_models(self) -> Dict[str, MPModelPlaceholder]:
        a2a = Predictor(path=here() / "models/RF_REG_CHEMBL251.pkg")
        a1 = Predictor(path=here() / "models/RF_REG_CHEMBL226.pkg")
        herg = Predictor(path=here() / "models/RF_REG_CHEMBL240.pkg")
        a2a.model.n_jobs = 1
        a1.model.n_jobs = 1
        herg.model.n_jobs = 1
        return {'a2a': a2a,
                'a1': a1,
                'herg': herg}

    def sample_conditional_information(self, n: int) -> Dict[str, torch.Tensor]:
        beta = None
        if self.temperature_sample_dist == 'uniform':
            beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'beta':
            beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = 1
        beta_enc = thermometer(torch.tensor(beta), 32, 0, upper_bound)  # TODO: hyperparameters
        m = Dirichlet(torch.FloatTensor([1.] * NUM_PREFS))
        preferences = m.sample([n])
        encoding = torch.cat([beta_enc, preferences], 1)
        return {'beta': torch.tensor(beta), 'encoding': encoding, 'preferences': preferences}

    def encode_conditional_information(self, info: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoding = torch.cat([torch.ones((len(info), 32)), info], 1)
        return {
            'beta': torch.ones(len(info)) * self.temperature_dist_params[-1],
            'encoding': encoding.float(),
            'preferences': info.float()
        }

    def compute_flat_rewards(self, mols: List[Mol]) -> Tuple[FlatRewards, torch.Tensor]:
        enhanced_fps = Predictor.calc_fp(mols=mols)
        is_valid = torch.tensor([i is not None for i in enhanced_fps]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, NUM_PREFS))), is_valid

        a2ascore_preds = self.models['a2a'](enhanced_fps)
        a1score_preds = self.models['a1'](enhanced_fps)
        hergscore_preds = self.models['herg'](enhanced_fps)

        a2ascore_preds = torch.from_numpy(a2ascore_preds.astype(np.float32))
        a1score_preds = torch.from_numpy(a1score_preds.astype(np.float32))
        hergscore_preds = torch.from_numpy(hergscore_preds.astype(np.float32))

        a2as = a2ascore_preds / 9.0  # 95pth percentile for A2A pChEMBL values is 9, so most will be [0-1]
        a1s = a1score_preds / 9.0  # 95pth percentile for A1 pChEMBL values is 9, so most will be [0-1]
        hergs = 1.0 - (hergscore_preds / 9.0)  # 95pth percentile for hERG pChEMBL values is 9, so most will be [0-1]

        flat_rewards = torch.stack([a2as, a1s, hergs], 1)

        return FlatRewards(flat_rewards), is_valid


class DrugExV2FragTrainer(BaseTrainer):
    def setup(self):
        hps = self.hps
        RDLogger.DisableLog('rdApp.*')
        self.rng = np.random.default_rng(142857)
        self.env_ = GraphBuildingEnv()
        self.ctx_ = FragBasedGraphContext(max_frags=9, num_cond_dim=hps['num_cond_dim'])
        self.training_data_ = []
        self.test_data_ = []
        self.offline_ratio = 0
        self.valid_offline_ratio = 0
        self.setup_algo()
        self.setup_task()
        self.setup_model()

        # Separate Z parameters from non-Z to allow for LR decay on the former
        Z_params = []
        non_Z_params = [i for i in self.model_.parameters() if all(id(i) != id(j) for j in Z_params)]
        self.opt = torch.optim.Adam(non_Z_params, hps['learning_rate'], (hps['momentum'], 0.999),
                                    weight_decay=hps['weight_decay'], eps=hps['adam_eps'])
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / hps['lr_decay']))

        self.sampling_tau = hps['sampling_tau']
        if self.sampling_tau > 0:
            self.sampling_model_ = copy.deepcopy(self.model_)
        else:
            self.sampling_model_ = self.model_
        eps = hps['tb_epsilon']
        hps['tb_epsilon'] = ast.literal_eval(eps) if isinstance(eps, str) else eps

        self.mb_size_ = hps['global_batch_size']
        self.clip_grad_param = hps['clip_grad_param']
        self.clip_grad_callback = {
            'value': (lambda params: torch.nn.utils.clip_grad_value_(params, self.clip_grad_param)),
            'norm': (lambda params: torch.nn.utils.clip_grad_norm_(params, self.clip_grad_param)),
            'none': (lambda x: None)
        }[hps['clip_grad_type']]

        self.sampling_hooks.append(MultiObjectiveStatsHook(256, self.hps['log_dir']))
        if self.hps['preference_type'] == 'dirichlet':
            valid_preferences = metrics.generate_simplex(NUM_PREFS, 5)  # This yields 35 points of dimension 4
        elif self.hps['preference_type'] == 'seeded_single':
            seeded_prefs = np.random.default_rng(142857 + int(self.hps['seed'])).dirichlet([1] * NUM_PREFS, 10)
            valid_preferences = seeded_prefs[int(self.hps['single_pref_target_idx'])].reshape((1, NUM_PREFS))
            self.task_.seeded_preference = valid_preferences[0]
        elif self.hps['preference_type'] == 'seeded_many':
            valid_preferences = np.random.default_rng(142857 + int(self.hps['seed'])).dirichlet([1] * NUM_PREFS, 10)
        else:
            raise ValueError(f"Preference type {hps['preference_type']} is not supported.")
        self._top_k_hook = TopKHook(10, 128, len(valid_preferences))
        self.test_data_ = RepeatedPreferenceDataset(valid_preferences, 128)
        self.valid_sampling_hooks.append(self._top_k_hook)
        self.algo_.task = self.task_

    def step(self, loss: torch.Tensor):
        loss.backward()
        for i in self.model_.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.lr_sched.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model_.parameters(), self.sampling_model_.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))

    def setup_algo(self):
        hps = self.hps
        if hps['algo'] == 'SUBTB':
            self.algo_ = SubTrajectoryBalance(self.env_, self.ctx_, self.rng, hps, max_nodes=9)
        else:
            raise ValueError("Only supports 'SUBTB' rn, sorry.")

    def setup_task(self):
        self.task_ = DrugExV2Task(self.training_data_, self.hps['temperature_sample_dist'],
                                  ast.literal_eval(self.hps['temperature_dist_params']), wrap_model=self._wrap_model_mp)

    def setup_model(self):
        model = FragBasedGraphGFN(self.ctx_, num_emb=self.hps['num_emb'], num_layers=self.hps['num_layers'],
                                  estimate_init_state_flow=False)
        self.model_ = model

    def build_callbacks(self):
        parent = self

        class TopKMetricCB:
            def on_validation_end(self, metrics: Dict[str, Any]):
                top_k = parent._top_k_hook.finalize()
                for i in range(len(top_k)):
                    metrics[f'topk_rewards_{i}'] = top_k[i]
                print('validation end', metrics)

        return {'topk': TopKMetricCB()}

    def default_hps(self) -> Dict[str, Any]:
        return {
            'bootstrap_own_reward': False,
            'learning_rate': 1e-4,
            'global_batch_size': 128,
            'num_emb': 128,
            'num_layers': 6,
            'tb_epsilon': None,
            'illegal_action_logreward': -75,
            'reward_loss_multiplier': 1,
            'temperature_sample_dist': 'uniform',
            'temperature_dist_params': '(32, 32)',
            'weight_decay': 1e-8,
            'num_data_loader_workers': 4,
            'momentum': 0.9,
            'adam_eps': 1e-8,
            'lr_decay': 20_000,
            'clip_grad_type': 'norm',
            'clip_grad_param': 10,
            'random_action_prob': 0.05,
            'num_cond_dim': 32 + NUM_PREFS,
            'sampling_tau': 0.0,
            'seed': 0,
            'preference_type': 'seeded_many',
            'algo': 'SUBTB',
            'log_dir': str(here() / f'logs/moo/drugex_v2_subtb/'),
            'num_training_steps': 10_000,
            'validate_every': 250,
            'valid_sample_cond_info': False,
            'lambda': 0.99
        }


def main():
    device = "cuda"
    trial = DrugExV2FragTrainer({}, torch.device(device))
    trial.verbose = True
    trial.run()


if __name__ == "__main__":
    main()
