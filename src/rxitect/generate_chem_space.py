import pickle
import time
from typing import List, Tuple

import selfies as sf
import yaml
from rdkit import Chem, RDLogger
from tqdm import tqdm

from rxitect import mol_utils

RDLogger.DisableLog("rdApp.*")


def generate_chem_space(smi: str, fp_type: str, num_random_samples: int, num_mutation_ls: List[int]) -> Tuple[List[str], List[float]]:
    """

    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise Exception("Invalid starting structure encountered")

    randomized_smile_orderings = [
        mol_utils.randomize_smiles(mol)
        for _ in tqdm(range(num_random_samples), desc="Randomizing SMILES orderings")
    ]

    # Convert all the molecules to SELFIES
    selfies_ls = [
        sf.encoder(x)
        for x in tqdm(
            randomized_smile_orderings, desc="Convert all molecules to SELFIES"
        )
    ]

    all_smiles_collect = []
    all_smiles_collect_broken = []

    for num_mutations in num_mutation_ls:
        # Mutate the SELFIES:
        selfies_mut = mol_utils.get_mutated_selfies(
            selfies_ls.copy(), num_mutations=num_mutations
        )

        # Convert back to SMILES:
        smiles_back = [
            sf.decoder(x)
            for x in tqdm(selfies_mut, desc="Converting SELFIES back to SMILES")
        ]
        all_smiles_collect = all_smiles_collect + smiles_back
        all_smiles_collect_broken.append(smiles_back)

    canon_smi_ls = []
    for item in all_smiles_collect:
        mol, smi_canon, did_convert = mol_utils.sanitize_smiles(item)
        if mol is None or smi_canon == "" or did_convert is False:
            raise Exception("Invalid smile string found")
        canon_smi_ls.append(smi_canon)
    canon_smi_ls = list(set(canon_smi_ls))

    canon_smi_ls_scores = mol_utils.get_fp_scores(
        canon_smi_ls, target_smi=smi, fp_type=fp_type
    )

    return canon_smi_ls, canon_smi_ls_scores
