import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def smiles_list_from_text_file(smiles_filepath: Path) -> List[str]:
    with open(file=smiles_filepath, mode="r") as f:
        smiles = f.read()
        smiles_list = smiles.splitlines()

    return smiles_list


@hydra.main(config_path="../configs", config_name="corpus.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from rxitect.generator.corpus import generate_smiles_corpus

    data_dir = Path(config.data_dir)
    smiles = smiles_list_from_text_file(data_dir / "raw/chembl_30_smiles.txt")
    output_dir = Path(config.output_dir)
    standardize = config.standardize
    create_voc_file = config.create_voc_file
    return_dataframe = config.return_dataframe
    is_fine_tune_batch = config.is_fine_tune_batch
    min_token_len = config.min_token_len
    max_token_len = config.max_token_len
    n_jobs = config.n_jobs

    generate_smiles_corpus(
        smiles,
        output_dir,
        standardize,
        create_voc_file,
        return_dataframe,
        is_fine_tune_batch,
        min_token_len,
        max_token_len,
        n_jobs,
    )


if __name__ == "__main__":
    main()
