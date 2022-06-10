Module rxitect.process_qsar_data
================================

Functions
---------

    
`construct_qsar_dataset(raw_data_path: str, targets: List[str], cols: List[str], px_placeholder: float = 3.99, random_state: int = 42, tsplit_year: Optional[int] = None, negative_samples: bool = True, out_dir: Optional[str] = None) ‑> rxitect.data.utils.QSARDataset`
:   Method that constructs a dataset from ChEMBL data to train QSAR regression models on,
    using a temporal split to create a hold-out test dataset for evaluation.
    
    Args:
        raw_data_path: filepath of the raw data
        targets: ChEMBL IDs that are relevant for the dataset creation
        cols: relevant columns for current dataset creation
        px_placeholder: pChEMBL value to use for negative examples
        random_state: seed integer to ensure reproducibility when randomness is involved
        tsplit_year (optional): year at which the temporal split should happen to create the held-out test set
        negative_samples: boolean flag that determines if negative samples should be included, default is True
        out_dur (optional): filepath where the processed data should be saved to, default is None
    
    Returns:
        A QSARDataset object - a convenient abstraction.

    
`main(cfg: omegaconf.dictconfig.DictConfig)`
: