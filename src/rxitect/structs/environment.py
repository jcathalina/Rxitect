import copy
from dataclasses import dataclass
from enum import Enum
from multiprocessing.sharedctypes import Value
from typing import Callable, List, Union
import numpy as np
import pandas as pd
from rdkit import Chem
from rxitect import mol_utils, sorting, tensor_utils
from rxitect.models.vanilla.predictor import Predictor
from rdkit.Chem import Mol
from globals import root_path
from rxitect.structs.ra_scorer import RetrosyntheticAccessibilityScorer


@dataclass(order=True, frozen=True)
class Objective:
    predictor: Predictor
    modifier: Callable
    threshold: float
    key: str

class ScoringScheme(Enum):
    WS = 0  # Weighted Sum
    PR = 1  # Pareto Ranking

class Environment:
    def __init__(self, objectives: List[Objective], scoring_scheme: ScoringScheme = ScoringScheme.PR):
        """
        Initialized methods for the construction of environment.
        Args:
            objectives: a list of objectives.
            scoring_scheme: an enum denoting which scoring scheme to use
        """
        self.objectives = objectives
        self.ths = [obj.threshold for obj in objectives]
        self.keys = [obj.key for obj in objectives]
        self.scoring_scheme = scoring_scheme

    def get_preds(self, mols: Union[List[str], List[Mol]], is_smiles: bool = False, use_mods: bool = True) -> pd.DataFrame:
        """
        Calculate the predicted scores of all objectives for all of samples
        Args:
            mols (List): a list of molecules
            is_smiles (bool): if True, the type of element in mols should be SMILES sequence, otherwise
                it should be the Chem.Mol
            use_mods (bool): if True, the function of modifiers will work, otherwise
                the modifiers will ignore.
        Returns:
            preds (DataFrame): The scores of all objectives for all of samples which also includes validity
                and desirability for each SMILES.
        """
        preds = {}
        fps = None

        # Extra copy for predictors that rely on SMILES strings
        smiles_ = copy.deepcopy(mols) if is_smiles else [Chem.MolToSmiles(mols)]

        if is_smiles:
            mols = [Chem.MolFromSmiles(s) for s in mols]
        for objective in self.objectives:

            if isinstance(objective.predictor, Predictor):
                if fps is None:
                    fps = objective.predictor.calc_fp(mols)
                score = objective.predictor(fps)
            elif isinstance(objective.predictor, RetrosyntheticAccessibilityScorer):
                score = objective.predictor(smiles_)
            else:
                raise ValueError(f"Unsupported type of predictor: {type(objective.predictor)}")

            if use_mods and objective.modifier is not None:
                score = objective.modifier(score)
            preds[objective.key] = score

        preds = pd.DataFrame(preds)
        undesire = (preds < self.ths)
        preds['DESIRE'] = (undesire.sum(axis=1) == 0).astype(int)
        return preds

    @classmethod
    def calc_fps(cls, mols: List[str], fp_type='ECFP6') -> list:
        """Calculate fingerprints for all SMILES in list
        Args:
            mols: List of SMILES repr. molecules
            fp_type: string containing name of fingerprint type
        Returns:
            List of fingerprints per SMILES
        """
        fps = [mol_utils.get_fingerprint(mol, fp_type) for mol in mols]
        return fps

    def calc_reward(self, smiles: List[str]) -> np.ndarray:
        """
        Calculate the single value as the reward for each molecule used for reinforcement learning
        Args:
            smiles (List):  a list of SMILES-based molecules
            scheme (str): the label of different rewarding schemes, including
                'WS': weighted sum, 'PR': Pareto ranking with Tanimoto distance,
                and 'CD': Pareto ranking with crowding distance.
        Returns:
            rewards (np.ndarray): n-d array in which the element is the reward for each molecule, and
                n is the number of array which equals to the size of smiles.
        """
        # smiles = [sf.decoder(selfie) for selfie in selfies]
        mols = [Chem.MolFromSmiles(smile) for smile in smiles]
        preds = self.get_preds(mols)
        desire = preds["DESIRE"].sum()
        undesire = len(preds) - desire
        preds = preds[self.keys].values

        if self.scoring_scheme == ScoringScheme.PR:
            fps = self.calc_fps(mols)
            rewards = np.zeros((len(smiles), 1))
            ranks = sorting.similarity_sort(preds, fps, is_gpu=True)
            score = (np.arange(undesire) / undesire / 2).tolist() + (np.arange(desire) / desire / 2 + 0.5).tolist()
            rewards[ranks, 0] = score
     
        elif self.scoring_scheme == ScoringScheme.WS:
            weight = ((preds < self.ths).mean(axis=0, keepdims=True) + 0.01) / \
                        ((preds >= self.ths).mean(axis=0, keepdims=True) + 0.01)
            weight = weight / weight.sum()
            rewards = preds.dot(weight.T)
        
        else:
            raise ValueError(f"Scoring scheme {self.scoring_scheme} does not exist!")

        return rewards

    @classmethod
    def get_default_env(cls, scoring_scheme: ScoringScheme = ScoringScheme.PR) -> "Environment":
        A1_pred = Predictor(path=root_path / "models/RF_REG_CHEMBL226.pkg")
        A2A_pred = Predictor(path=root_path / "models/RF_REG_CHEMBL251.pkg")
        ERG_pred = Predictor(path=root_path / "models/RF_REG_CHEMBL240.pkg")

        if scoring_scheme == ScoringScheme.WS:
            mod1 = tensor_utils.ClippedScore(lower_x=3, upper_x=6.5)
            mod2 = tensor_utils.ClippedScore(lower_x=10, upper_x=3)
        else:
            mod1 = tensor_utils.ClippedScore(lower_x=3, upper_x=6.5)
            mod2 = tensor_utils.ClippedScore(lower_x=10, upper_x=6.5)


        A1 = Objective(predictor=A1_pred,
                       modifier=mod1,
                       threshold=0.5 if scoring_scheme == ScoringScheme.WS else 0.99,
                       key="A1")
        A2A = Objective(predictor=A2A_pred,
                        modifier=mod1,
                        threshold=0.5 if scoring_scheme == ScoringScheme.WS else 0.99,
                        key="A2A")
        ERG = Objective(predictor=ERG_pred,
                        modifier=mod2,
                        threshold=0.5 if scoring_scheme == ScoringScheme.WS else 0.99,
                        key="ERG")

        env = Environment([A1, A2A, ERG])
        return env


    @classmethod
    def get_ra_boosted_env(cls, scoring_scheme: ScoringScheme = ScoringScheme.PR, xgb: bool = False) -> "Environment":
        A1_pred = Predictor(path=root_path / "models/RF_REG_CHEMBL226.pkg")
        A2A_pred = Predictor(path=root_path / "models/RF_REG_CHEMBL251.pkg")
        ERG_pred = Predictor(path=root_path / "models/RF_REG_CHEMBL240.pkg")
        RA_pred = RetrosyntheticAccessibilityScorer(use_xgb_model=xgb)

        no_mod = lambda x: x
        if scoring_scheme == ScoringScheme.WS:
            mod1 = tensor_utils.ClippedScore(lower_x=3, upper_x=6.5)
            mod2 = tensor_utils.ClippedScore(lower_x=10, upper_x=3)
        else:
            mod1 = tensor_utils.ClippedScore(lower_x=3, upper_x=6.5)
            mod2 = tensor_utils.ClippedScore(lower_x=10, upper_x=6.5)


        A1 = Objective(predictor=A1_pred,
                       modifier=mod1,
                       threshold=0.5 if scoring_scheme == ScoringScheme.WS else 0.99,
                       key="A1")
        A2A = Objective(predictor=A2A_pred,
                        modifier=mod1,
                        threshold=0.5 if scoring_scheme == ScoringScheme.WS else 0.99,
                        key="A2A")
        ERG = Objective(predictor=ERG_pred,
                        modifier=mod2,
                        threshold=0.5 if scoring_scheme == ScoringScheme.WS else 0.99,
                        key="ERG")
        RA_SCORE = Objective(predictor=RA_pred,
                             modifier=no_mod,  # Already always returns score [0-1]
                             threshold=0.5 if scoring_scheme == ScoringScheme.WS else 0.99,
                             key="RA_SCORE")

        env = Environment([A1, A2A, ERG, RA_SCORE])
        return env