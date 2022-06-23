""" Module containing all types and type imports.
    Approach borrowed from https://github.com/MolecularAI/aizynthfinder.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import os
import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem

StrDict = Dict[str, Any]
ArrayDict = Dict[str, ArrayLike]
RDKitMol = Chem.rdchem.Mol
PathLike = Union[str, bytes, os.PathLike]