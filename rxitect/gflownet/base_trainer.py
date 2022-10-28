from pathlib import Path
from typing import Dict, Any, List, Callable

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch

from rxitect.gflownet.algorithms.interfaces.graph_algorithm import IGraphAlgorithm
from rxitect.gflownet.contexts.envs.graph_building_env import GraphBuildingEnv
from rxitect.gflownet.contexts.interfaces.graph_context import IGraphContext
from rxitect.gflownet.tasks.interfaces.graph_task import IGraphTask
from rxitect.gflownet.utils.graph import GraphActionCategorical
from rxitect.gflownet.utils.multiproc import wrap_model_mp
from rxitect.gflownet.utils.sampling_iterator import SamplingIterator


class BaseTrainer:
    def __init__(self, hps: Dict[str, Any], device: torch.device) -> None:
        """A GFlowNet trainer base class. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        hps:
            A dictionary of hyperparameters. These override default values obtained by the `default_hps` method.
        device:
            The torch device of the main worker.
        """
        # self.setup should at least set these up:
        self.training_data_: Dataset
        self.test_data_: Dataset
        self.model_: nn.Module
        # `sampling_model` is used by the data workers to sample new objects from the model. Can be
        # the same as `model`.
        self.sampling_model_: nn.Module
        self.mb_size_: int
        self.env_: GraphBuildingEnv
        self.ctx_: IGraphContext
        self.task_: IGraphTask
        self.algo_: IGraphAlgorithm

        # Override default hyperparameters with the constructor arguments
        self.hps = {**self.default_hps(), **hps}
        self.device = device
        # The number of processes spawned to sample object and do CPU work
        self.num_workers: int = self.hps.get('num_data_loader_workers', 0)
        # The ratio of samples drawn from `self.training_data` during training. The rest is drawn from
        # `self.sampling_model`.
        self.offline_ratio: float = 0.5
        # idem, but from `self.test_data` during validation.
        self.valid_offline_ratio: float = 1.0
        # If True, print messages during training
        self.verbose: bool = False
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []
        self.valid_sampling_hooks: List[Callable] = []

        self.setup()
        self._check_valid_setup_attrs()

    def default_hps(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()

    def step(self, loss: torch.Tensor):
        raise NotImplementedError()

    def _wrap_model_mp(self, model):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers.  """
        model.to(self.device)
        if self.num_workers > 0:
            placeholder = wrap_model_mp(model, self.num_workers, cast_types=(Batch, GraphActionCategorical))
            return placeholder, torch.device('cpu')
        return model, self.device

    def build_callbacks(self):
        return {}

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.sampling_model_)
        iterator = SamplingIterator(self.training_data_, model, self.mb_size_ * 2, self.ctx_, self.algo_, self.task_,
                                    dev,
                                    offline_ratio=self.offline_ratio, log_dir=self.hps['log_dir'])
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0)

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.model_)
        iterator = SamplingIterator(self.test_data_, model, self.mb_size_, self.ctx_, self.algo_, self.task_, dev,
                                    offline_ratio=self.valid_offline_ratio, stream=False,
                                    sample_cond_info=self.hps.get('valid_sample_cond_info', True))
        for hook in self.valid_sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(iterator, batch_size=None, num_workers=self.num_workers,
                                           persistent_workers=self.num_workers > 0)

    def train_batch(self, batch: Batch, epoch_idx: int, batch_idx: int) -> Dict[str, Any]:
        loss, info = self.algo_.compute_batch_losses(self.model_, batch, num_bootstrap=self.mb_size_)
        self.step(loss)
        if hasattr(batch, 'extra_info'):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()}

    def evaluate_batch(self, batch: Batch, epoch_idx: int = 0, batch_idx: int = 0) -> Dict[str, Any]:
        loss, info = self.algo_.compute_batch_losses(self.model_, batch, num_bootstrap=batch.num_offline)
        if hasattr(batch, 'extra_info'):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, 'item') else v for k, v in info.items()}

    def run(self):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        self.model_.to(self.device)
        self.sampling_model_.to(self.device)
        epoch_length = max(len(self.training_data_), 1)
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        callbacks = self.build_callbacks()
        for it, batch in zip(range(1, 1 + self.hps['num_training_steps']), train_dl):
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx)
            if self.verbose:
                print(it, ' '.join(f'{k}:{v:.2f}' for k, v in info.items()))
            self.log(info, it, 'train')

            if it % self.hps['validate_every'] == 0:
                for vbatch in valid_dl:
                    info = self.evaluate_batch(vbatch.to(self.device), epoch_idx, batch_idx)
                    self.log(info, it, 'valid')
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, 'on_validation_end'):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, 'valid_end')
                torch.save({
                    'models_state_dict': [self.model_.state_dict()],
                    'hps': self.hps,
                }, open(Path(self.hps['log_dir']) / 'model_state.pt', 'wb'))

    def log(self, info, index, key) -> None:
        if not hasattr(self, '_summary_writer'):
            self._summary_writer = SummaryWriter(self.hps['log_dir'])
        for k, v in info.items():
            self._summary_writer.add_scalar(f'{key}_{k}', v, index)

    def _check_valid_setup_attrs(self) -> None:
        """
        Raises
        ------
        AttributeError
        """
        missing_attrs = []
        for attr in ["training_data_",
                     "test_data_",
                     "model_",
                     "sampling_model_",
                     "mb_size_",
                     "env_",
                     "ctx_",
                     "task_",
                     "algo_", ]:
            if not hasattr(self, attr):
                missing_attrs.append(attr)
        if missing_attrs:
            raise AttributeError(f"The following attrs are missing: {missing_attrs}. Please make sure your setup method"
                                 f" properly adds these to the your trainer class.")
