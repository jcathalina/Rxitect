""" Module containing all types and type imports.
    Approach borrowed from https://github.com/MolecularAI/aizynthfinder.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from rdkit import Chem

StrDict = Dict[str, Any]
RDKitMol = Chem.rdchem.Mol
Fingerprints = Dict[Union[Tuple[int, int], Tuple[int]], np.ndarray]
