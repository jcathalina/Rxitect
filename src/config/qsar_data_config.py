from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Params:
    """Data Structure that defines parameters used
    for the processing of the ChEMBL data to create
    the QSAR model

    Args:
        targets: A list of ChEMBL IDs to create a dataset from.
        cols: A list of column names (all lowercase expected) to retain in the final dataset.
        classification: A flag that determines if the final pX values should be transformed into binary (if a classification model is being used).
        px_placeholder: The float value that should be used for rows with missing pX values.
        px_thresh: The pX value threshold where a compound is considered active, applicable when the data needs to be transformed into binary.
        train_size: The fraction of the data that should be held out for testing.
        val_size: The fraction of the data that should be held out for validation.
        random_seed: The random seed number used to reproduce experiments
    """

    targets: List[str]
    cols: List[str]
    classification: bool
    px_placeholder: float
    px_thresh: float
    train_size: float
    val_size: float
    random_seed: int


@dataclass
class Raw:
    """Data related to the raw files associated with the data used for QSAR model training.

    Args:
        path: The filepath to the raw data file.
    """

    path: str


@dataclass
class Processed:
    """Data related to the processed files associated with the data used for QSAR model training.

    Args:
        dir: The directory where processed files should be stored.
        name: The name that the processed file should have.
        path: The filepath to the raw data file.
    """

    dir: str
    name: str
    path: str


@dataclass
class Files:
    """Data related to the final files associated with the data used for QSAR model training.

    Args:
        dir: The directory where final files should be stored.
        train_data: The filepath of the file where the train data is stored.
        test_data: The filepath of the file where the test data is stored.
        val_data: The filepath of the file where the validation data is stored.
    """

    dir: str
    train_data: str


@dataclass
class QSARDataConfig:
    """Configuration object that defines all data necessary for QSAR processing"""

    params: Params
    raw: Raw
    processed: Processed
    files: Files
