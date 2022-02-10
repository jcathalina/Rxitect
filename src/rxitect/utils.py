import dataclasses
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import rdkit
import selfies as sf
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Mol, MolFromSmiles, MolToSmiles
from rdkit.Chem.AtomPairs.Sheridan import GetBPFingerprint, GetBTFingerprint
from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity

RDLogger.DisableLog("rdApp.*")


@dataclass(order=True, frozen=True)
class SanitizedSmiles:
    """A helper class containing information about a sanitized smile string.

    Args:
        mol: RdKit mol object (None if invalid smile string smi)
        smi_canon: Canonicalized SMILES representation of smi (None if invalid smile string smi)
        conversion_successful: True/False to indicate if conversion was  successful
    """

    mol: Mol
    canon_smi: str
    success: bool

    def __iter__(self) -> Iterator:
        return iter(dataclasses.astuple(self))


def randomize_smiles(mol: Mol) -> str:
    """Returns a random (dearomatized) SMILES given an rdkit Mol representation of a molecule.

    Args:
        mol: RdKit mol object (None if invalid smile string smi)

    Returns:
        mol: RdKit mol object  (None if invalid smile string smi)
    """
    if not mol:
        return None

    Chem.Kekulize(mol)
    return rdkit.Chem.MolToSmiles(
        mol, canonical=False, doRandom=True, isomericSmiles=False, kekuleSmiles=True
    )


def sanitize_smiles(smi: str) -> SanitizedSmiles:
    """Compute a canonical SMILES representation of smi, along with Mol and success data.

    Args:
        smi: smiles string to be canonicalized

    Returns:
        SanitizedSmiles: a canonical SMILES representation with Mol and success data.
    """
    try:
        mol = MolFromSmiles(smi, sanitize=True)
        canon_smi = MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return SanitizedSmiles(mol=mol, canon_smi=canon_smi, success=True)
    except:
        return SanitizedSmiles(mol=None, canon_smi=None, success=False)


def get_selfie_chars(selfie: str) -> List[str]:
    """Obtain a list of all selfie characters in string selfie
    Example:
    >>> get_selfie_chars('[C][=C][C][=C][C][=C][Ring1][Branch1_1]')
    ['[C]', '[=C]', '[C]', '[=C]', '[C]', '[=C]', '[Ring1]', '[Branch1_1]']

    Args:
        selfie: A selfie string - representing a molecule

    Returns:
        chars_selfie: list of selfie characters present in molecule selfie
    """
    chars_selfie = []  # A list of all SELFIE sybols from string selfie
    while selfie != "":
        chars_selfie.append(selfie[selfie.find("[") : selfie.find("]") + 1])
        selfie = selfie[selfie.find("]") + 1 :]
    return chars_selfie


class _FingerprintCalculator:
    """Calculate the fingerprint for a molecule, given the fingerprint type

    Args:
        mol: RdKit mol object (None if invalid smile string smi)
        fp_type: Fingerprint type  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)

    Returns:
        RDKit fingerprint object
    """

    def get_fingerprint(self, mol: Mol, fp_type: str):
        method_name = "get_" + fp_type
        method = getattr(self, method_name)
        if method is None:
            raise Exception(f"{fp_type} is not a supported fingerprint type.")
        return method(mol)

    def get_AP(self, mol: Mol):
        return AllChem.GetAtomPairFingerprint(mol, maxLength=10)

    def get_PHCO(self, mol: Mol):
        return Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)

    def get_BPF(self, mol: Mol):
        return GetBPFingerprint(mol)

    def get_BTF(self, mol: Mol):
        return GetBTFingerprint(mol)

    def get_PATH(self, mol: Mol):
        return AllChem.RDKFingerprint(mol)

    def get_ECFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2)

    def get_ECFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3)

    def get_FCFP4(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 2, useFeatures=True)

    def get_FCFP6(self, mol: Mol):
        return AllChem.GetMorganFingerprint(mol, 3, useFeatures=True)


def get_fingerprint(mol: Mol, fp_type: str):
    """Fingerprint getter method. Fingerprint is returned after using object of
    class '_FingerprintCalculator'

    Args:
        mol: RdKit mol object (None if invalid smile string smi)
        fp_type: Fingerprint type (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)

    Returns:
        RDKit fingerprint object

    """
    return _FingerprintCalculator().get_fingerprint(mol=mol, fp_type=fp_type)


def mutate_selfie(
    selfie: str, max_molecules_len: int, write_fail_cases: bool = False
) -> Tuple[str, str]:
    """Return a mutated selfie string (only one mutation on selfie is performed)
       Mutations are done until a valid molecule is obtained
       Rules of mutation: With a 33.3% propbabily, either:
        1. Add a random SELFIE character in the string
        2. Replace a random SELFIE character with another
        3. Delete a random character

    Args:
        selfie: SELFIE string to be mutated
        max_molecules_len: Mutations of SELFIE string are allowed up to this length
        write_fail_cases: If true, failed mutations are recorded in "selfie_failure_cases.txt"

    Returns:
        mut_selfie: Mutated SELFIE string
        canon_smi: canonical smile of mutated SELFIE string
    """
    valid = False
    fail_counter = 0
    chars_selfie = get_selfie_chars(selfie)

    while not valid:
        fail_counter += 1

        alphabet = list(sf.get_semantic_robust_alphabet())  # 34 SELFIE characters

        choice_ls = [1, 2, 3]  # 1=Insert; 2=Replace; 3=Delete
        random_choice = np.random.choice(choice_ls, 1)[0]

        # Insert a character in a Random Location
        if random_choice == 1:
            random_index = np.random.randint(len(chars_selfie) + 1)
            random_character = np.random.choice(alphabet, size=1)[0]

            selfie_mutated_chars = (
                chars_selfie[:random_index]
                + [random_character]
                + chars_selfie[random_index:]
            )

        # Replace a random character
        elif random_choice == 2:
            random_index = np.random.randint(len(chars_selfie))
            random_character = np.random.choice(alphabet, size=1)[0]
            if random_index == 0:
                selfie_mutated_chars = [random_character] + chars_selfie[
                    random_index + 1 :
                ]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index]
                    + [random_character]
                    + chars_selfie[random_index + 1 :]
                )

        # Delete a random character
        elif random_choice == 3:
            random_index = np.random.randint(len(chars_selfie))
            if random_index == 0:
                selfie_mutated_chars = chars_selfie[random_index + 1 :]
            else:
                selfie_mutated_chars = (
                    chars_selfie[:random_index] + chars_selfie[random_index + 1 :]
                )

        else:
            raise Exception("Invalid Operation trying to be performed")

        mut_selfie = "".join(x for x in selfie_mutated_chars)
        orig_selfie = "".join(x for x in chars_selfie)

        try:
            smiles = sf.decoder(mut_selfie)
            _mol, canon_smi, done = sanitize_smiles(smiles)
            if len(selfie_mutated_chars) > max_molecules_len or canon_smi == "":
                done = False
            if done:
                valid = True
            else:
                valid = False
        except:
            valid = False
            if fail_counter > 1 and write_fail_cases == True:
                f = open("selfie_failure_cases.txt", "a+")
                f.write(
                    "Tried to mutate SELFIE: "
                    + str(orig_selfie)
                    + " To Obtain: "
                    + str(mut_selfie)
                    + "\n"
                )
                f.close()

    return (mut_selfie, canon_smi)


def get_mutated_selfies(selfies: List[str], num_mutations: int) -> List[str]:
    """Mutates all the SELFIES in 'selfies' 'num_mutations' number of times.

    Args:
        selfies: A list of SELFIES
        num_mutations: number of mutations to perform on each selfie within 'selfies'

    Returns:
        selfies: A list of mutated SELFIES
    """
    for _ in range(num_mutations):
        selfie_ls_mut_ls = []
        for selfie in selfies:

            selfie_chars = get_selfie_chars(selfie)
            max_molecules_len = len(selfie_chars) + num_mutations

            selfie_mutated, _ = mutate_selfie(selfie, max_molecules_len)
            selfie_ls_mut_ls.append(selfie_mutated)

        selfies = selfie_ls_mut_ls.copy()
    return selfies


def get_fp_scores(smiles_back: List[str], target_smi: str, fp_type: str):
    """Calculate the Tanimoto fingerprint (using fp_type fingerint) similarity between a list
       of SMILES and a known target structure (target_smi).

    Args:
        smiles_back: A list of valid SMILES strings
        target_smi: A valid SMILES string. Each smile in 'smiles_back' will be compared to this stucture
        fp_type: Type of fingerprint  (choices: AP/PHCO/BPF,BTF,PAT,ECFP4,ECFP6,FCFP4,FCFP6)

    Returns:
        smiles_back_scores (list of floats) : List of fingerprint similarities
    """
    smiles_back_scores = []
    target = Chem.MolFromSmiles(target_smi)

    fp_target = get_fingerprint(target, fp_type)

    for item in smiles_back:
        mol = Chem.MolFromSmiles(item)
        fp_mol = get_fingerprint(mol, fp_type)
        score = TanimotoSimilarity(fp_mol, fp_target)
        smiles_back_scores.append(score)
    return smiles_back_scores
