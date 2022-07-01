Module rxitect.preprocessing.generate_smiles_corpus
===================================================

Functions
---------


`generate_smiles_corpus(smiles: List[str], output_dir: pathlib.Path, standardize: bool = True, return_dataframe: bool = False, is_fine_tune_batch: bool = False, min_token_len: int = 10, max_token_len: int = 100, n_jobs: int = 1) ‑> Optional[pandas.core.frame.DataFrame]`
:   This method constructs a dataset with molecules represented as SMILES.

    Each molecule will be decomposed
    into a series of tokens. In the end, all the tokens will be put into one set as vocabulary.
    Args:
        smiles (List[str]): The file path of input, either .sdf file or tab-delimited file
        output_dir (Path): The directory to write the created files to.
        standardize (bool): If the molecule is should be standardized. The charges will be
                            removed and only the largest fragment will be kept. True by default.
        return_dataframe (bool): If the dataframe created during corpus generation should
                                 be returned in addition to being written to file.
                                 False by default.
        is_fine_tune_batch (bool): If the smiles passed should be used for creating the fine tuning corpus.
                                   False by default.
        min_token_len (int): The shortest token length allowed for SMILES included in the final corpus.
        max_token_len (int): The longest token length allowed for SMILES included in the final corpus.
        n_jobs (int): The number of workers that should be used for the methods
                      that can be embarassingly parallelized using joblib.
                      Set this to -1 to use all workers. Defaults to 1 worker (sequential).
    Returns:
        pd.DataFrame (optional): The dataframe containing the final corpus.
        Only gets returned if `return_dataframe` was set to True.
