from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

import torch
from rdkit.Chem.rdchem import Mol
from torch import nn
from torch.types import Device
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch

from rxitect.algorithms.gfn_algorithm import GFNAlgorithm
from rxitect.envs.contexts import ActionCategorical, GraphEnvContext
from rxitect.tasks.gfn_task import GFNTask


class GFNTrainer:
    def __init__(self, hps: Dict[str, Any], device: Device):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.
        Parameters
        ----------
        hps: Dict[str, Any]
            A dictionary of hyperparameters. These override default values obtained by the `default_hps` method.
        device: Device
            The torch device of the main worker.
        """
        # self.setup should at least set these up:
        self.training_data: Dataset = None
        self.test_data: Dataset = None
        self.model: nn.Module = None
        # `sampling_model` is used by the data workers to sample new objects from the model. Can be
        # the same as `model`.
        self.sampling_model: nn.Module = None
        self.mb_size: int = None
        self.ctx: GraphEnvContext = None
        self.task: GFNTask = None
        self.algo: GFNAlgorithm = None

        # Override default hyperparameters with the constructor arguments
        self.hps = {**self.default_hps(), **hps}
        self.device = device
        # The number of processes spawned to sample object and do CPU work
        self.num_workers: int = self.hps.get("num_data_loader_workers", 0)
        # The offline_ratio of samples drawn from `self.training_data` during training. The rest is drawn from
        # `self.sampling_model`.
        self.offline_ratio: float = 0.5
        # idem, but from `self.test_data` during validation.
        self.valid_offline_ratio: float = 1
        # If True, print messages during training
        self.verbose: bool = False
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []

        self.setup()

    def default_hps(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def setup(self):
        raise NotImplementedError()

    def step(self, loss: torch.Tensor):
        raise NotImplementedError()

    def _wrap_model_mp(self, model):
        """Wraps a nn.Module instance so that it can be shared to `DataLoader` workers."""
        model.to(self.device)
        if self.num_workers > 0:
            placeholder = wrap_model_mp(
                model, self.num_workers, cast_types=(Batch, ActionCategorical)
            )
            return placeholder, torch.device("cpu")
        return model, self.device

    def build_training_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.sampling_model)
        iterator = SamplingIterator(
            self.training_data,
            model,
            self.mb_size * 2,
            self.ctx,
            self.algo,
            self.task,
            dev,
            ratio=self.offline_ratio,
            log_dir=self.hps["log_dir"],
        )
        for hook in self.sampling_hooks:
            iterator.add_log_hook(hook)
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def build_validation_data_loader(self) -> DataLoader:
        model, dev = self._wrap_model_mp(self.model)
        iterator = SamplingIterator(
            self.test_data,
            model,
            self.mb_size,
            self.ctx,
            self.algo,
            self.task,
            dev,
            ratio=self.valid_offline_ratio,
            stream=False,
        )
        return torch.utils.data.DataLoader(
            iterator,
            batch_size=None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_batch(
        self, batch: Batch, epoch_idx: int, batch_idx: int
    ) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(
            self.model, batch, num_bootstrap=self.mb_size
        )
        self.step(loss)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def evaluate_batch(
        self, batch: Batch, epoch_idx: int = 0, batch_idx: int = 0
    ) -> Dict[str, Any]:
        loss, info = self.algo.compute_batch_losses(
            self.model, batch, num_bootstrap=batch.num_offline
        )
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def run(self):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        epoch_length = max(len(self.training_data), 1)
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        for it, batch in zip(range(1, 1 + self.hps["num_training_steps"]), train_dl):
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx)
            if self.verbose:
                print(it, " ".join(f"{k}:{v:.2f}" for k, v in info.items()))
            self.log(info, it, "train")

            if it % self.hps["validate_every"] == 0:
                for batch in valid_dl:
                    info = self.evaluate_batch(
                        batch.to(self.device), epoch_idx, batch_idx
                    )
                    self.log(info, it, "valid")
                torch.save(
                    {
                        "models_state_dict": [self.model.state_dict()],
                        "hps": self.hps,
                    },
                    open(pathlib.Path(self.hps["log_dir"]) / "model_state.pt", "wb"),
                )

    def log(self, info, index, key):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(
                self.hps["log_dir"]
            )
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{key}_{k}", v, index)
