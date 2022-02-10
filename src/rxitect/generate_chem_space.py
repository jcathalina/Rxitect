import pickle
import time

import selfies as sf
import yaml
from rdkit import Chem, RDLogger
from tqdm import tqdm

from globals import root_path
from rxitect import utils

RDLogger.DisableLog("rdApp.*")


def main():
    settings_path = root_path / "config/chem_space_settings.yml"
    settings = yaml.safe_load(open(settings_path, "r"))

    print(settings)
    data = settings["data"]
    params = settings["params"]

    smi = data["smiles"]
    fp_type = data["fp_type"]

    total_time = time.time()
    # num_random_samples = 50000 # For a more exhaustive search!
    num_random_samples = params["n_random_samples"]
    num_mutation_ls = params["n_mutations"]

    mol = Chem.MolFromSmiles(smi)
    if mol == None:
        raise Exception("Invalid starting structure encountered")

    start_time = time.time()
    randomized_smile_orderings = [
        utils.randomize_smiles(mol)
        for _ in tqdm(range(num_random_samples), desc="Randomizing SMILES orderings")
    ]

    # Convert all the molecules to SELFIES
    selfies_ls = [
        sf.encoder(x)
        for x in tqdm(
            randomized_smile_orderings, desc="Convert all molecules to SELFIES"
        )
    ]
    print("Randomized molecules (in SELFIES) time: ", time.time() - start_time)

    all_smiles_collect = []
    all_smiles_collect_broken = []

    start_time = time.time()
    for num_mutations in num_mutation_ls:
        # Mutate the SELFIES:
        selfies_mut = utils.get_mutated_selfies(
            selfies_ls.copy(), num_mutations=num_mutations
        )

        # Convert back to SMILES:
        smiles_back = [
            sf.decoder(x)
            for x in tqdm(selfies_mut, desc="Converting SELFIES back to SMILES")
        ]
        all_smiles_collect = all_smiles_collect + smiles_back
        all_smiles_collect_broken.append(smiles_back)

    print("Mutation obtainment time (back to smiles): ", time.time() - start_time)

    # Work on:  all_smiles_collect
    start_time = time.time()
    canon_smi_ls = []
    for item in all_smiles_collect:
        mol, smi_canon, did_convert = utils.sanitize_smiles(item)
        if mol == None or smi_canon == "" or did_convert == False:
            raise Exception("Invalid smile string found")
        canon_smi_ls.append(smi_canon)
    canon_smi_ls = list(set(canon_smi_ls))
    print("Unique mutated structure obtainment time: ", time.time() - start_time)

    start_time = time.time()
    canon_smi_ls_scores = utils.get_fp_scores(
        canon_smi_ls, target_smi=smi, fp_type=fp_type
    )
    print("Fingerprint calculation time: ", time.time() - start_time)
    print("Total time: ", time.time() - total_time)

    # Molecules with fingerprint similarity > 0.8
    indices_thresh_8 = [
        i
        for i, x in tqdm(
            enumerate(canon_smi_ls_scores), desc="Filter mols with similarity > 0.8"
        )
        if x > 0.8
    ]
    mols_8 = [Chem.MolFromSmiles(canon_smi_ls[idx]) for idx in indices_thresh_8]

    # # Molecules with fingerprint similarity > 0.6
    # indices_thresh_6 = [i for i,x in enumerate(canon_smi_ls_scores) if x > 0.6 and x < 0.8]
    # mols_6 = [Chem.MolFromSmiles(canon_smi_ls[idx]) for idx in indices_thresh_6]

    # # Molecules with fingerprint similarity > 0.4
    # indices_thresh_4 = [i for i,x in enumerate(canon_smi_ls_scores) if x > 0.4 and x < 0.6]
    # mols_4 = [Chem.MolFromSmiles(canon_smi_ls[idx]) for idx in indices_thresh_4]

    with open("mols_8.dat", "wb") as f:
        pickle.dump(mols_8, f)


if __name__ == "__main__":
    main()
