import ast
import copy
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from pyprojroot import here
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import QED
from rdkit.Chem.rdchem import Mol as RDMol
import scipy.stats as stats
import torch
from torch import Tensor
from torch.distributions.dirichlet import Dirichlet
import torch.nn as nn
from torch.utils.data import Dataset

from rxitect.gflownet.algorithms.sub_trajectory_balance import SubTrajectoryBalance
from rxitect.gflownet.algorithms.trajectory_balance import TrajectoryBalance
from rxitect.gflownet.base_trainer import BaseTrainer
from rxitect.gflownet.contexts.envs.graph_building_env import GraphBuildingEnv
from rxitect.gflownet.contexts.frag_graph_context import FragBasedGraphContext
from rxitect.gflownet.hooks.multiobjective_hooks import MultiObjectiveStatsHook, TopKHook
from rxitect.gflownet.models.frag_graph_gfn import FragBasedGraphGFN
from rxitect.gflownet.tasks.interfaces.graph_task import IGraphTask
from rxitect.gflownet.utils import metrics
from rxitect.gflownet.utils.types import FlatRewards, RewardScalar
from rxitect.scorers import sascore, rascore
from rxitect.scorers.a2ascore import Predictor
from rxitect.scorers.rascore import load_rascore_model


class SEHFragTrainer(BaseTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            'bootstrap_own_reward': False,
            'learning_rate': 1e-4,
            'global_batch_size': 64,
            'num_emb': 128,
            'num_layers': 4,
            'tb_epsilon': None,
            'illegal_action_logreward': -75,
            'reward_loss_multiplier': 1,
            'temperature_sample_dist': 'uniform',
            'temperature_dist_params': '(.5, 32)',
            'weight_decay': 1e-8,
            'num_data_loader_workers': 8,
            'momentum': 0.9,
            'adam_eps': 1e-8,
            'lr_decay': 20000,
            'Z_lr_decay': 20000,
            'clip_grad_type': 'norm',
            'clip_grad_param': 10,
            'random_action_prob': 0.,
            'sampling_tau': 0.,
            'num_cond_dim': 32,
        }

    def setup_algo(self):
        pass

    def setup_task(self):
        pass

    def setup_model(self):
        pass

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

    def step(self, loss: Tensor):
        loss.backward()
        for i in self.model_.parameters():
            self.clip_grad_callback(i)
        self.opt.step()
        self.opt.zero_grad()
        self.lr_sched.step()
        if self.sampling_tau > 0:
            for a, b in zip(self.model_.parameters(), self.sampling_model_.parameters()):
                b.data.mul_(self.sampling_tau).add_(a.data * (1 - self.sampling_tau))


class SEHMOOTask(IGraphTask):
    """Sets up a multiobjective task where the rewards are (functions of):
    - the binding energy of a molecule to Soluble Epoxide Hydrolases.
    - its QED
    - its synthetic accessibility
    - its molecular weight

    The proxy is pretrained, and obtained from the original GFlowNet paper, see `gflownet.models.bengio2021flow`.
    """

    def __init__(self, dataset: Dataset, temperature_distribution: str, temperature_parameters: Tuple[float],
                 wrap_model: Callable[[nn.Module], nn.Module] = None):
        self._wrap_model = wrap_model
        self.models = self._load_task_models()
        self.dataset = dataset
        self.temperature_sample_dist = temperature_distribution
        self.temperature_dist_params = temperature_parameters
        self.seeded_preference = None
        self.experimental_dirichlet = False
        self.rng = None

    def flat_reward_transform(self, y: Union[float, Tensor]) -> FlatRewards:
        return FlatRewards(torch.as_tensor(y))

    def inverse_flat_reward_transform(self, rp):
        return rp

    def _load_task_models(self):
        model = load_rascore_model(ckpt_filepath=here() / "models/rascore_26102022.ckpt", device="cuda")
        model, self.device = self._wrap_model(model)
        a2a = Predictor(path=here() / "models/RF_REG_CHEMBL251.pkg")
        a2a.model.n_jobs = 1
        return {'rascore': model, 'a2a': a2a}

    def sample_conditional_information(self, n):
        beta = None
        if self.temperature_sample_dist == 'gamma':
            loc, scale = self.temperature_dist_params
            beta = self.rng.gamma(loc, scale, n).astype(np.float32)
            upper_bound = stats.gamma.ppf(0.95, loc, scale=scale)
        elif self.temperature_sample_dist == 'uniform':
            beta = self.rng.uniform(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = self.temperature_dist_params[1]
        elif self.temperature_sample_dist == 'beta':
            beta = self.rng.beta(*self.temperature_dist_params, n).astype(np.float32)
            upper_bound = 1
        beta_enc = thermometer(torch.tensor(beta), 32, 0, upper_bound)  # TODO: hyperparameters
        if self.seeded_preference is not None:
            preferences = torch.tensor([self.seeded_preference] * n).float()
        elif self.experimental_dirichlet:
            a = np.random.dirichlet([1] * 4, n)
            b = np.random.exponential(1, n)[:, None]
            preferences = Dirichlet(torch.tensor(a * b)).sample([1])[0].float()
        else:
            m = Dirichlet(torch.FloatTensor([1.] * 4))
            preferences = m.sample([n])
        encoding = torch.cat([beta_enc, preferences], 1)
        return {'beta': torch.tensor(beta), 'encoding': encoding, 'preferences': preferences}

    def encode_conditional_information(self, info):
        # This assumes we're using a constant (max) beta and that info is the preferences
        encoding = torch.cat([torch.ones((len(info), 32)), info], 1)
        return {
            'beta': torch.ones(len(info)) * self.temperature_dist_params[-1],
            'encoding': encoding.float(),
            'preferences': info.float()
        }

    def cond_info_to_reward(self, cond_info: Dict[str, Tensor], flat_reward: FlatRewards) -> RewardScalar:
        if isinstance(flat_reward, list):
            if isinstance(flat_reward[0], Tensor):
                flat_reward = torch.stack(flat_reward)
            else:
                flat_reward = torch.tensor(flat_reward)
        scalar_reward = (flat_reward * cond_info['preferences']).sum(1)
        return scalar_reward ** cond_info['beta']

    def compute_flat_rewards(self, mols: List[RDMol]) -> Tuple[FlatRewards, Tensor]:
        # graphs = [bengio2021flow.mol2graph(i) for i in mols]
        # is_valid = torch.tensor([i is not None for i in graphs]).bool()

        # batch = gd.Batch.from_data_list([i for i in graphs if i is not None])
        # batch.to(self.device)
        # seh_preds = self.models['seh'](batch).reshape((-1,)).clip(1e-4, 100).data.cpu() / 8
        # seh_preds[seh_preds.isnan()] = 0
        fcfps = torch.tensor(np.array([rascore.mol2fcfp(i) for i in mols]), dtype=torch.float, device=self.device)
        enhanced_fps = Predictor.calc_fp(mols=mols)
        is_valid = torch.tensor([i is not None for i in fcfps]).bool()
        if not is_valid.any():
            return FlatRewards(torch.zeros((0, 4))), is_valid
        # fcfps.to(self.device)
        rascore_preds = self.models['rascore'](fcfps)  # is already between 0-1, no need to normalize
        rascore_preds = rascore_preds.reshape(-1).detach().cpu()
        a2ascore_preds = self.models['a2a'](enhanced_fps)
        a2ascore_preds = torch.from_numpy(a2ascore_preds.astype(np.float32))

        def safe(f, x, default):
            try:
                return f(x)
            except Exception as e:
                print(f"The following exception occurred for function {f}: {e}. Return default.")
                return default

        qeds = torch.tensor([safe(QED.qed, i, 0) for i, v in zip(mols, is_valid) if v.item()])
        sas = torch.tensor([safe(sascore.calculateScore, i, 10) for i, v in zip(mols, is_valid) if v.item()])
        sas = (10 - sas) / 9  # Turn into a [0-1] reward
        a2as = a2ascore_preds / 9.0  # 95pth percentile for A2A pChEMBL values is 9, so most will be [0-1]
        # molwts = torch.tensor([safe(Descriptors.MolWt, i, 1000) for i, v in zip(mols, is_valid) if v.item()])
        # molwts = ((300 - molwts) / 700 + 1).clip(0, 1)  # 1 until 300 then linear decay to 0 until 1000
        # flat_rewards = torch.stack([rascore_preds, sas, qeds, a2as], 1)
        flat_rewards = torch.stack([qeds, qeds, qeds, qeds], 1)
        return FlatRewards(flat_rewards), is_valid


class SEHMOOFragTrainer(SEHFragTrainer):
    def default_hps(self) -> Dict[str, Any]:
        return {
            **super().default_hps(),
            'use_fixed_weight': False,
            'num_cond_dim': 32 + 4,  # thermometer encoding of beta + 4 preferences
            'sampling_tau': 0.95,
            'valid_sample_cond_info': False,
            'preference_type': 'dirichlet',
        }

    def setup_algo(self):
        hps = self.hps
        if hps['algo'] == 'SUBTB':
            self.algo_ = SubTrajectoryBalance(self.env_, self.ctx_, self.rng, hps, max_nodes=9)
        else:
            raise ValueError("Only supports SUBTB rn, sorry.")

    def setup_task(self):
        self.task_ = SEHMOOTask(self.training_data_, self.hps['temperature_sample_dist'],
                                ast.literal_eval(self.hps['temperature_dist_params']), wrap_model=self._wrap_model_mp)

    def setup_model(self):
        model = FragBasedGraphGFN(self.ctx_, num_emb=self.hps['num_emb'], num_layers=self.hps['num_layers'],
                                  estimate_init_state_flow=False)

        if self.hps['algo'] in ['A2C', 'MOQL']:
            model.do_mask = False
        self.model_ = model

    def setup(self):
        super().setup()
        self.sampling_hooks.append(MultiObjectiveStatsHook(256, self.hps['log_dir']))
        if self.hps['preference_type'] == 'dirichlet':
            valid_preferences = metrics.generate_simplex(4, 5)  # This yields 35 points of dimension 4
        elif self.hps['preference_type'] == 'seeded_single':
            seeded_prefs = np.random.default_rng(142857 + int(self.hps['seed'])).dirichlet([1] * 4, 10)
            valid_preferences = seeded_prefs[int(self.hps['single_pref_target_idx'])].reshape((1, 4))
            self.task_.seeded_preference = valid_preferences[0]
        elif self.hps['preference_type'] == 'seeded_many':
            valid_preferences = np.random.default_rng(142857 + int(self.hps['seed'])).dirichlet([1] * 4, 10)
        self._top_k_hook = TopKHook(10, 128, len(valid_preferences))
        self.test_data_ = RepeatedPreferenceDataset(valid_preferences, 128)
        self.valid_sampling_hooks.append(self._top_k_hook)

        self.algo_.task = self.task_

    def build_callbacks(self):
        parent = self

        class TopKMetricCB:
            def on_validation_end(self, metrics: Dict[str, Any]):
                top_k = parent._top_k_hook.finalize()
                for i in range(len(top_k)):
                    metrics[f'topk_rewards_{i}'] = top_k[i]
                print('validation end', metrics)

        return {'topk': TopKMetricCB()}


class RepeatedPreferenceDataset:
    def __init__(self, preferences, repeat):
        self.prefs = preferences
        self.repeat = repeat

    def __len__(self):
        return len(self.prefs) * self.repeat

    def __getitem__(self, idx):
        assert 0 <= idx < len(self)
        return torch.tensor(self.prefs[int(idx // self.repeat)])


def thermometer(v: Tensor, n_bins: int = 50, vmin: float = 0, vmax: float = 1) -> Tensor:
    """Thermometer encoding of a scalar quantity.
    Parameters
    ----------
    v: Tensor
        Value(s) to encode. Can be any shape
    n_bins: int
        The number of dimensions to encode the values into
    vmin: float
        The smallest value, below which the encoding is equal to torch.zeros(n_bins)
    vmax: float
        The largest value, beyond which the encoding is equal to torch.ones(n_bins)
    Returns
    -------
    encoding: Tensor
        The encoded values, shape: `v.shape + (n_bins,)`
    """
    bins = torch.linspace(vmin, vmax, n_bins)
    gap = bins[1] - bins[0]
    return (v[..., None] - bins.reshape((1,) * v.ndim + (-1,))).clamp(0, gap.item()) / gap


def main():
    """Example of how this model can be run outside Determined"""
    hps = {
        'lr_decay': 10_000,
        'log_dir': str(here() / f'logs/subtbtest/run_tmp/'),
        'num_training_steps': 10_000,
        'validate_every': 500,
        'sampling_tau': 0,
        'num_layers': 6,
        'num_data_loader_workers': 0,
        'temperature_dist_params': '(16, 16)',
        'global_batch_size': 64,
        'algo': 'SUBTB',
        'sql_alpha': 0.01,
        'seed': 0,
        'preference_type': 'seeded_many',
        'lambda': 1.0
    }
    trial = SEHMOOFragTrainer(hps, torch.device('cpu'))
    trial.verbose = True
    trial.run()


if __name__ == '__main__':
    main()
