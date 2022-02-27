from typing import Optional, List

import numpy as np
import rdkit
from RAscore import RAscore_NN, RAscore_XGB

from globals import root_path


NN_MODEL_PATH = root_path / "models/rascore/DNN_chembl_fcfp_counts/model.h5"
XGB_MODEL_PATH = root_path / "models/rascore/XGB_chembl_ecfp_counts/model.pkl"


class RetrosyntheticAccessibilityScorer:
    def __init__(self, use_xgb_model: bool = False):
        self.scorer = (
            RAscore_XGB.RAScorerXGB(model_path=XGB_MODEL_PATH)
            if use_xgb_model
            else RAscore_NN.RAScorerNN(model_path=NN_MODEL_PATH)
        )

    def __call__(self, mols: List[str]):
        scores = np.zeros(shape=len(mols), dtype="float64")
        for i, mol in enumerate(mols):
            scores[i] = self.scorer.predict(mol)

        return scores

    @staticmethod
    def calculate_score(mol: str, use_xgb_model: bool = False) -> Optional[float]:
        """
        Given a SMILES string, returns a score in [0-1] that indicates how
        likely RA Score predicts it is to find a synthesis route.
        Args:
            mol: a SMILES string representing a molecule.
            use_xgb_model: Determines if the XGB-based model for RA Score
                                should be used instead of NN-based. False by default.

        Returns:
            A score between 0 and 1 indicating how likely a synthesis route is to be found by the underlying CASP tool (AiZynthFinder).
        """
        scorer = (
            RAscore_XGB.RAScorerXGB(model_path=XGB_MODEL_PATH)
            if use_xgb_model
            else RAscore_NN.RAScorerNN(model_path=NN_MODEL_PATH)
        )

        score = scorer.predict(smiles=mol)
        return score