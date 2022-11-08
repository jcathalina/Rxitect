from collections import defaultdict
import pathlib
import queue
import threading
from typing import List

import numpy as np
import torch
from torch import Tensor
import torch.multiprocessing as mp

from rxitect.gflownet.utils import metrics


class MultiObjectiveStatsHook:
    def __init__(self, num_to_keep: int, log_dir: str, save_every=50):
        # This __init__ is only called in the main process. This object is then (potentially) cloned
        # in pytorch data worker processed and __call__'ed from within those processes. This means
        # each process will compute its own Pareto front, which we will accumulate in the main
        # process by pushing local fronts to self.pareto_queue.
        self.num_to_keep = num_to_keep
        self.all_flat_rewards: List[Tensor] = []
        self.all_smi: List[str] = []
        self.hsri_epsilon = 0.3
        self.compute_hsri = False
        self.compute_normed = False
        self.pareto_queue: mp.Queue = mp.Queue()
        self.pareto_front = None
        self.pareto_front_smi = None
        self.pareto_metrics = mp.Array('f', 4)
        self.stop = threading.Event()
        self.save_every = save_every
        self.log_path = pathlib.Path(log_dir) / 'pareto.pt'
        self.pareto_thread = threading.Thread(target=self._run_pareto_accumulation, daemon=True)
        self.pareto_thread.start()

    def __del__(self):
        self.stop.set()

    def _run_pareto_accumulation(self):
        num_updates = 0
        while not self.stop.is_set():
            try:
                r, smi = self.pareto_queue.get(True, 1)  # Block for a second then check if we've stopped
            except queue.Empty:
                continue
            except ConnectionError:
                break
            if self.pareto_front is None:
                p = self.pareto_front = r
                psmi = smi
            else:
                p = np.concatenate([self.pareto_front, r], 0)
                psmi = self.pareto_front_smi + smi
            idcs = metrics.is_pareto_efficient(-p, False)
            self.pareto_front = p[idcs]
            self.pareto_front_smi = [psmi[i] for i in idcs]
            self.pareto_metrics[0] = metrics.get_hypervolume(torch.tensor(self.pareto_front), zero_ref=True)
            num_updates += 1
            if num_updates % self.save_every == 0:
                torch.save(
                    {
                        'pareto_front': self.pareto_front,
                        'pareto_metrics': list(self.pareto_metrics),
                        'pareto_front_smi': self.pareto_front_smi,
                    }, open(self.log_path, 'wb'))

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        self.all_flat_rewards = self.all_flat_rewards + list(flat_rewards)
        self.all_smi = self.all_smi + list([i.get('smi', None) for i in trajs])
        if len(self.all_flat_rewards) > self.num_to_keep:
            self.all_flat_rewards = self.all_flat_rewards[-self.num_to_keep:]
            self.all_smi = self.all_smi[-self.num_to_keep:]

        flat_rewards = torch.stack(self.all_flat_rewards).numpy()
        target_min = flat_rewards.min(0).copy()
        target_range = flat_rewards.max(0).copy() - target_min
        hypercube_transform = metrics.Normalizer(
            loc=target_min,
            scale=target_range,
        )
        pareto_idces = metrics.is_pareto_efficient(-flat_rewards, return_mask=False)
        gfn_pareto = flat_rewards[pareto_idces]
        pareto_smi = [self.all_smi[i] for i in pareto_idces]

        self.pareto_queue.put((gfn_pareto, pareto_smi))
        unnorm_hypervolume_with_zero_ref = metrics.get_hypervolume(torch.tensor(gfn_pareto), zero_ref=True)
        unnorm_hypervolume_wo_zero_ref = metrics.get_hypervolume(torch.tensor(gfn_pareto), zero_ref=False)
        info = {
            'UHV with zero ref': unnorm_hypervolume_with_zero_ref,
            'UHV w/o zero ref': unnorm_hypervolume_wo_zero_ref,
        }
        if self.compute_normed:
            normed_gfn_pareto = hypercube_transform(gfn_pareto)
            hypervolume_with_zero_ref = metrics.get_hypervolume(torch.tensor(normed_gfn_pareto), zero_ref=True)
            hypervolume_wo_zero_ref = metrics.get_hypervolume(torch.tensor(normed_gfn_pareto), zero_ref=False)
            info = {
                **info,
                'HV with zero ref': hypervolume_with_zero_ref,
                'HV w/o zero ref': hypervolume_wo_zero_ref,
            }
        if self.compute_hsri:
            upper = np.zeros(gfn_pareto.shape[-1]) + self.hsri_epsilon
            lower = np.ones(gfn_pareto.shape[-1]) * -1 - self.hsri_epsilon
            hsr_indicator = metrics.HSR_Calculator(lower, upper)
            try:
                hsri_w_pareto, x = hsr_indicator.calculate_hsr(-1 * gfn_pareto)
            except Exception:
                hsri_w_pareto = 0
            try:
                hsri_on_flat, _ = hsr_indicator.calculate_hsr(-1 * flat_rewards)
            except Exception:
                hsri_on_flat = 0
            info = {
                **info,
                'hsri_with_pareto': hsri_w_pareto,
                'hsri_on_flat_rew': hsri_on_flat,
            }
        info['lifetime_hv0'] = self.pareto_metrics[0]

        return info


class TopKHook:
    def __init__(self, k, repeats, num_preferences):
        self.queue: mp.Queue = mp.Queue()
        self.k = k
        self.repeats = repeats
        self.num_preferences = num_preferences

    def __call__(self, trajs, rewards, flat_rewards, cond_info):
        self.queue.put([(i['data_idx'], r) for i, r in zip(trajs, rewards)])
        return {}

    def finalize(self):
        data = []
        while not self.queue.empty():
            try:
                data += self.queue.get(True, 1)
            except queue.Empty:
                print("Warning, TopKHook queue timed out!")
                break
        repeats = defaultdict(list)
        for idx, r in data:
            repeats[idx // self.repeats].append(r)
        top_ks = [np.mean(sorted(i)[-self.k:]) for i in repeats.values()]
        assert len(top_ks) == self.num_preferences  # Make sure we got all of them?
        return top_ks