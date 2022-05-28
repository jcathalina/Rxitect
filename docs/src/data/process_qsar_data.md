Module src.data.process_qsar_data
=================================

Functions
---------

    
`main(cfg: src.structs.qsar_data_config.QSARDataConfig)`
:   

    
`process_qsar_data(cfg: src.structs.qsar_data_config.QSARDataConfig) ‑> pandas.core.frame.DataFrame`
:   Function that loads and processed ChEMBL data for specific ligands
    to be used for training QSAR models.
    
    Args:
        cfg (QSARDataConfig): A dictionary used to configure the parameters and filepaths
                              used for the processing of the ChEMBL data into usable features
                              for an eventual QSAR model.
    
    Returns:
        A DataFrame that contains structured information with SMILES and their respective
        features that will be used in the training of a QSAR model.

    
`write_training_data(df: pandas.core.frame.DataFrame, dir: str, train_data: str, test_data: str, val_data: str, test_size: float = 0.2, val_size: float = 0.1) ‑> None`
:   Function that divides the training data based on chemical diversity, into the
    appropriate train/test/val sets and writes them to their respective files.
    
    Args:
        df: A DataFrame containing the cleaned ChEMBL data prepared for QSAR model training.
        train_data: Filepath to where the train data should be stored.
        test_data: Filepath to where the test data should be stored.
        val_data: Filepath to where the validation data should be stored.
        test_size: Fraction of data that should be held out for the test set.
        val_size: Fraction of data that should be held out for the validation set.