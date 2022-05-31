from dataclasses import dataclass

import numpy as np


@dataclass
class LigandTrainingData:
    X: np.ndarray
    y: np.ndarray
