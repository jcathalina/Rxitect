from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from torch.utils.data import Dataset, IterableDataset

if TYPE_CHECKING:
    from torch.types import Device

    from rxitect.algorithms.gfn_algorithm import GFNAlgorithm
    from rxitect.envs.contexts import GraphEnvContext
    from rxitect.tasks.gfn_task import GFNTask


class SamplingIterator(IterableDataset):
    """This class allows us to parallelize and train faster.
    By separating sampling data/the model and building torch geometric
    graphs from training the model, we can do the former in different
    processes, which is much faster since much of graph construction
    is CPU-bound.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: nn.Module,
        batch_size: int,
        ctx: GraphEnvContext,
        algo: GFNAlgorithm,
        task: GFNTask,
        device: Device,
        offline_ratio: bool = 0.5,
        stream: bool = True,
        log_dir: Optional[str] = None,
    ):
        """Parameters
        ----------
        dataset: Dataset
            A dataset instance
        model: nn.Module
            The model we sample from (must be on CUDA already or share_memory() must be called so that
            parameters are synchronized between each worker)
        batch_size: int
            The number of trajectories, each trajectory will be composed of many graphs, so this is
            _not_ the batch size in terms of the number of graphs (that will depend on the task)
        algo:
            The training algorithm, e.g. a TrajectoryBalance instance
        task: ConditionalTask
        offline_ratio: float
            The offline_ratio of offline trajectories in the batch.
        stream: bool
            If True, data is sampled iid for every batch. Otherwise, this is a normal in-order
            dataset iterator.
        log_dir: str
            If not None, logs each SamplingIterator worker's generated molecules to that file.
        """
        self.data = dataset
        self.model = model
        self.batch_size = batch_size
        self.offline_batch_size = int(np.ceil(batch_size * offline_ratio))
        self.online_batch_size = int(np.floor(batch_size * (1 - offline_ratio)))
        self.offline_ratio = offline_ratio
        self.ctx = ctx
        self.algo = algo
        self.task = task
        self.device = device
        self.stream = stream
        self.log_dir = log_dir if self.offline_ratio < 1 and self.stream else None
        # This SamplingIterator instance will be copied by torch DataLoaders for each worker, so we
        # don't want to initialize per-worker things just yet, such as the log the worker writes
        # to. This must be done in __iter__, which is called by the DataLoader once this instance
        # has been copied into a new python process.
        # self.log = SQLiteLog()  # Make generic logger that writes to txt file
        self.log_hooks: List[Callable] = []

    def add_log_hook(self, hook: Callable):
        self.log_hooks.append(hook)

    def _idx_iterator(self):
        RDLogger.DisableLog("rdApp.*")
        if self.stream:
            # If we're streaming data, just sample `offline_batch_size` indices
            while True:
                yield self.rng.integers(0, len(self.data), self.offline_batch_size)
        else:
            # Otherwise, figure out which indices correspond to this worker
            worker_info = torch.utils.data.get_worker_info()
            n = len(self.data)
            if n == 0:
                yield np.arange(0, 0)
                return
            if worker_info is None:
                start, end, wid = 0, n, -1
            else:
                nw = worker_info.num_workers
                wid = worker_info.id
                start, end = int(np.floor(n / nw * wid)), int(
                    np.ceil(n / nw * (wid + 1))
                )
            bs = self.offline_batch_size
            if end - start < bs:
                yield np.arange(start, end)
                return
            for i in range(start, end - bs, bs):
                yield np.arange(i, i + bs)
            if i + bs < end:
                yield np.arange(i + bs, end)

    def __len__(self):
        if self.stream:
            return int(1e6)
        return len(self.data)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        self._wid = worker_info.id if worker_info is not None else 0
        # Now that we know we are in a worker instance, we can initialize per-worker things
        self.rng = self.algo.rng = self.task.rng = np.random.default_rng(
            142857 + self._wid
        )
        self.ctx.device = self.device
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_path = f"{self.log_dir}/generated_mols_{self._wid}.db"
            # self.log.connect(self.log_path)

        for idcs in self._idx_iterator():
            num_offline = idcs.shape[0]  # This is in [0, self.offline_batch_size]
            # Sample conditional info such as temperature, trade-off weights, etc.

            cond_info = self.task.sample_conditional_information(
                num_offline + self.online_batch_size
            )
            is_valid = torch.ones(cond_info["beta"].shape[0]).bool()

            # Sample some dataset data
            mols, flat_rewards = (
                map(list, zip(*[self.data[i] for i in idcs])) if len(idcs) else ([], [])
            )
            flat_rewards = list(
                self.task.flat_reward_transform(torch.tensor(flat_rewards))
            )
            graphs = [self.ctx.mol_to_graph(m) for m in mols]
            trajs = self.algo.create_training_data_from_graphs(graphs)
            # Sample some on-policy data
            if self.online_batch_size > 0:
                with torch.no_grad():
                    trajs += self.algo.create_training_data_from_own_samples(
                        self.model,
                        self.online_batch_size,
                        cond_info["encoding"][num_offline:],
                    )
                if self.algo.bootstrap_own_reward:
                    # The model can be trained to predict its own reward,
                    # i.e. predict the output of cond_info_to_reward
                    pred_reward = [
                        i["reward_pred"].cpu().item() for i in trajs[num_offline:]
                    ]
                    flat_rewards += pred_reward
                else:
                    # Otherwise, query the task for flat rewards
                    valid_idcs = torch.tensor(
                        [
                            i + num_offline
                            for i in range(self.online_batch_size)
                            if trajs[i + num_offline]["is_valid"]
                        ]
                    ).long()
                    # fetch the valid trajectories endpoints
                    mols = [
                        self.ctx.graph_to_mol(trajs[i]["traj"][-1][0])
                        for i in valid_idcs
                    ]
                    # ask the task to compute their reward
                    preds, m_is_valid = self.task.compute_flat_rewards(mols)
                    # The task may decide some mols are invalid, we have to again filter those
                    valid_idcs = valid_idcs[m_is_valid]
                    pred_reward = torch.zeros((self.online_batch_size, preds.shape[1]))
                    pred_reward[valid_idcs - num_offline] = preds
                    # if preds.shape[0] > 0:
                    #     for i in range(self.number_of_objectives):
                    #         pred_reward[valid_idcs - num_offline, i] = preds[range(preds.shape[0]), i]
                    is_valid[num_offline:] = False
                    is_valid[valid_idcs] = True
                    flat_rewards += list(pred_reward)
                    # Override the is_valid key in case the task made some mols invalid
                    for i in range(self.online_batch_size):
                        trajs[num_offline + i]["is_valid"] = is_valid[
                            num_offline + i
                        ].item()
            flat_rewards = torch.stack(flat_rewards)
            # Compute scalar rewards from conditional information & flat rewards
            rewards = self.task.cond_info_to_reward(cond_info, flat_rewards)
            rewards[torch.logical_not(is_valid)] = np.exp(
                self.algo.illegal_action_logreward
            )
            # Construct batch
            batch = self.algo.construct_batch(trajs, cond_info["encoding"], rewards)
            batch.num_offline = num_offline
            batch.num_online = self.online_batch_size
            batch.flat_rewards = flat_rewards
            batch.mols = mols

            if self.online_batch_size > 0 and self.log_dir is not None:
                self.log_generated(
                    trajs[num_offline:],
                    rewards[num_offline:],
                    flat_rewards[num_offline:],
                    {k: v[num_offline:] for k, v in cond_info.items()},
                )
            if self.online_batch_size > 0:
                extra_info = {}
                for hook in self.log_hooks:
                    extra_info.update(hook(trajs, rewards, flat_rewards, cond_info))
                batch.extra_info = extra_info
            yield batch

    def log_generated(self, trajs, rewards, flat_rewards, cond_info):
        mols = [
            Chem.MolToSmiles(self.ctx.graph_to_mol(trajs[i]["traj"][-1][0]))
            if trajs[i]["is_valid"]
            else ""
            for i in range(len(trajs))
        ]

        flat_rewards = (
            flat_rewards.reshape((len(flat_rewards), -1)).data.numpy().tolist()
        )
        rewards = rewards.data.numpy().tolist()
        preferences = (
            cond_info.get("preferences", torch.zeros((len(mols), 0)))
            .data.numpy()
            .tolist()
        )
        logged_keys = [
            k for k in sorted(cond_info.keys()) if k not in ["encoding", "preferences"]
        ]

        data = [
            [mols[i], rewards[i]]
            + flat_rewards[i]
            + preferences[i]
            + [cond_info[k][i].item() for k in logged_keys]
            for i in range(len(trajs))
        ]
        data_labels = (
            ["smi", "r"]
            + [f"fr_{i}" for i in range(len(flat_rewards[0]))]
            + [f"pref_{i}" for i in range(len(preferences[0]))]
            + [f"ci_{k}" for k in logged_keys]
        )
        # self.log.insert_many(data, data_labels)
