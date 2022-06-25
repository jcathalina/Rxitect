import gzip
from logging import Logger
from typing import List, Optional

from rdkit import Chem
from tqdm import tqdm

from rxitect.utils.types import PathLike


def process_gzipped_chembl(
    path_to_chembl_gz: PathLike,
    output_path: PathLike,
    return_smiles: bool = False,
    n_compounds: Optional[int] = None,
) -> Optional[List[str]]:
    """Handle gzipped sdf file with RDkit"""
    assert (
        path_to_chembl_gz.endswith(".gz"),
        "The file that was passed is not gzipped.",
    )
    inf = gzip.open(path_to_chembl_gz)
    fsuppl = Chem.ForwardSDMolSupplier(inf)
    smiles = []
    for mol in tqdm(fsuppl, total=n_compounds, desc="Processing ChEMBL molecules"):
        try:
            smiles.append(Chem.MolToSmiles(mol))
        except Exception as e:
            Logger.warning(f"Was not able to convert {mol} to smiles: {e}")

    open(output_path, "w").write("\n".join(smi for smi in smiles))

    if return_smiles:
        return smiles


if __name__ == "__main__":
    # process_gzipped_chembl()
    pass  # TODO: Use e.g. click to pass command line args for this script.
