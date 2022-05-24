import logging
import pathlib

import joblib
import numpy as np
import rdkit.Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from src.structs.property import Property


class Predictor:
    def __init__(self, path: pathlib.Path):
        self.model = joblib.load(path)

    def __call__(self, fps):
        scores = self.model.predict(fps)
        return scores

    @classmethod
    def calc_fp(cls, mols, radius: int = 3, bit_len: int = 2048):
        ecfp = cls.calc_ecfp(mols, radius=radius, bit_len=bit_len)
        phch = cls.calc_physchem(mols)
        fps = np.concatenate([ecfp, phch], axis=1)
        return fps

    @classmethod
    def calc_ecfp(cls, mols, radius: int = 3, bit_len: int = 2048):
        fps = np.zeros((len(mols), bit_len))
        for i, mol in enumerate(mols):
            try:
                if isinstance(mol, str):
                    mol = rdkit.Chem.MolFromSmiles(mol)  # Make Mol object before calculating morgan FP or else it breaks.
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=bit_len)
                DataStructs.ConvertToNumpyArray(fp, fps[i, :])
            except Exception as e:
                logging.error(f"Something went wrong while creating fingerprints: {e}")
        return fps

    @classmethod
    def calc_physchem(cls, mols):
        prop_list = [
            "MW",
            "logP",
            "HBA",
            "HBD",
            "Rotable",
            "Amide",
            "Bridge",
            "Hetero",
            "Heavy",
            "Spiro",
            "FCSP3",
            "Ring",
            "Aliphatic",
            "Aromatic",
            "Saturated",
            "HeteroR",
            "TPSA",
            "Valence",
            "MR",
        ]
        fps = np.zeros((len(mols), 19))
        props = Property()
        for i, prop in enumerate(prop_list):
            props.prop = prop
            fps[:, i] = props(mols)
        return fps
