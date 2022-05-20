from typing import Union
import numpy as np
import torch
from rdkit import DataStructs
from abc import abstractclassmethod, ABC, abstractmethod

from globals import device


class SortingAlgorithm(ABC):
    def __init__(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def sort(cls, scoring_mat: Union[np.ndarray, torch.Tensor], fps: np.ndarray) -> np.ndarray:
        """
        Args:
            scoring_mat (np.ndarray): m x n scoring matrix, where m is the number of samples
                and n is the number of objectives.
            fps (np.ndarray): m-d vector as fingerprints for all the molecules

        Returns:
            rank (np.array): m-d vector as the index of well-ranked solutions.
        """
        pass

class NSGA_2(SortingAlgorithm):
    def __init__(self, use_gpu: bool = True) -> None:
        super().__init__()
        self.use_gpu = use_gpu

    def check_domination(solution_1: np.ndarray, solution_2: np.ndarray) -> bool:
        """
        Determine if solution 1 is dominated by solution 2.
        Args:
            solution_1 (np.ndarray): m-d vector represented the socres of a solution for all of objectives.
            solution_2 (np.ndarray): m-d vector represented the socres of a solution for all of objectives.

        Returns:
            True if solution 1 is dominated by solution 2, otherwise False.
        """
        assert solution_1.shape == solution_2.shape
        all = np.all(solution_1 <= solution_2)
        any = np.any(solution_1 < solution_2)
        return all & any

    def cpu_non_dominated_sort(self, scoring_mat: np.ndarray):
        """
        The CPU version of non-dominated sorting algorithms
        Args:
            scoring_mat (np.ndarray): m x n scoring matrix, where m is the number of samples
                and n is the number of objectives.

        Returns:
            fronts (List): a list of Pareto fronts, in which the dominated solutions are on the top,
                and non-dominated solutions are on the bottom.
        """
        domina = [[] for _ in range(len(scoring_mat))]
        front = []
        count = np.zeros(len(scoring_mat), dtype=int)
        ranks = np.zeros(len(scoring_mat), dtype=int)
        for p, ind1 in enumerate(scoring_mat):
            for q in range(p + 1, len(scoring_mat)):
                ind2 = scoring_mat[q]
                if self.check_domination(ind1, ind2):
                    domina[p].append(q)
                    count[q] += 1
                elif self.check_domination(ind2, ind1):
                    domina[q].append(p)
                    count[p] += 1
            if count[p] == 0:
                ranks[p] = 0
                front.append(p)

        fronts = [np.sort(front)]
        i = 0
        while len(fronts[i]) > 0:
            temp = []
            for f in fronts[i]:
                for d in domina[f]:
                    count[d] -= 1
                    if count[d] == 0:
                        ranks[d] = i + 1
                        temp.append(d)
            i = i + 1
            fronts.append(np.sort(temp))
        del fronts[len(fronts) - 1]
        return fronts

    def gpu_non_dominated_sort(self, scoring_mat: torch.Tensor):
        """
        The GPU version of non-dominated sorting algorithms
        Args:
            scoring_mat (np.ndarray): m x n scoring matrix, where m is the number of samples
                and n is the number of objectives.

        Returns:
            fronts (List): a list of Pareto fronts, in which the dominated solutions are on the top,
                and non-dominated solutions are on the bottom.
        """
        domina = (scoring_mat.unsqueeze(1) <= scoring_mat.unsqueeze(0)).all(-1)
        domina_any = (scoring_mat.unsqueeze(1) < scoring_mat.unsqueeze(0)).any(-1)
        domina = (domina & domina_any).half()

        fronts = []
        while (domina.diag() == 0).any():
            count = domina.sum(dim=0)
            front = torch.where(count == 0)[0]
            fronts.append(front)
            domina[front, :] = 0
            domina[front, front] = -1
        return fronts
        
    @classmethod
    def sort(self, scoring_mat: Union[np.ndarray, torch.Tensor], fps: np.ndarray) -> np.ndarray:
        """
        Revised cowding distance algorithm to rank the solutions in the same frontier with Tanimoto-distance.
        Args:
            scoring_mat (np.ndarray): m x n scoring matrix, where m is the number of samples
                and n is the number of objectives.
            fps (np.ndarray): m-d vector as fingerprints for all the molecules

        Returns:
            rank (np.array): m-d vector as the index of well-ranked solutions.
        """
        if self.is_gpu:
            scoring_mat = torch.tensor(scoring_mat, device=device)
            fronts = self.gpu_non_dominated_sort(scoring_mat)
        else:
            fronts = self.cpu_non_dominated_sort(scoring_mat)
        rank = []
        for i, front in enumerate(fronts):
            fp = [fps[f] for f in front]
            if len(front) > 2 and None not in fp:
                dist = np.zeros(len(front))
                for j in range(len(front)):
                    tanimoto = 1 - np.array(DataStructs.BulkTanimotoSimilarity(fp[j], fp))
                    order = tanimoto.argsort()
                    dist[order[0]] += 0
                    dist[order[-1]] += 10 ** 4
                    for k in range(1, len(order) - 1):
                        dist[order[k]] += tanimoto[order[k + 1]] - tanimoto[order[k - 1]]
                fronts[i] = front[dist.argsort()]
            rank.extend(fronts[i].tolist())
        return rank
