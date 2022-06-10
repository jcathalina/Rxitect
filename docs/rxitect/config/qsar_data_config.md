Module rxitect.config.qsar_data_config
======================================

Classes
-------

`Files(dir: str, train_data: str)`
:   Data related to the final files associated with the data used for QSAR model training.
    
    Args:
        dir: The directory where final files should be stored.
        train_data: The filepath of the file where the train data is stored.
        test_data: The filepath of the file where the test data is stored.
        val_data: The filepath of the file where the validation data is stored.

    ### Class variables

    `dir: str`
    :

    `train_data: str`
    :

`Params(targets: List[str], cols: List[str], classification: bool, px_placeholder: float, px_thresh: float, random_seed: int)`
:   Data Structure that defines parameters used
    for the processing of the ChEMBL data to create
    the QSAR model
    
    Args:
        targets: A list of ChEMBL IDs to create a dataset from.
        cols: A list of column names (all lowercase expected) to retain in the final dataset.
        classification: A flag that determines if the final pX values should be transformed into binary (if a classification model is being used).
        px_placeholder: The float value that should be used for rows with missing pX values.
        px_thresh: The pX value threshold where a compound is considered active, applicable when the data needs to be transformed into binary.
        random_seed: The random seed number used to reproduce experiments

    ### Class variables

    `classification: bool`
    :

    `cols: List[str]`
    :

    `px_placeholder: float`
    :

    `px_thresh: float`
    :

    `random_seed: int`
    :

    `targets: List[str]`
    :

`Processed(dir: str, name: str, path: str)`
:   Data related to the processed files associated with the data used for QSAR model training.
    
    Args:
        dir: The directory where processed files should be stored.
        name: The name that the processed file should have.
        path: The filepath to the raw data file.

    ### Class variables

    `dir: str`
    :

    `name: str`
    :

    `path: str`
    :

`QSARDataConfig(params: rxitect.config.qsar_data_config.Params, raw: rxitect.config.qsar_data_config.Raw, processed: rxitect.config.qsar_data_config.Processed, files: rxitect.config.qsar_data_config.Files)`
:   Configuration object that defines all data necessary for QSAR processing

    ### Class variables

    `files: rxitect.config.qsar_data_config.Files`
    :

    `params: rxitect.config.qsar_data_config.Params`
    :

    `processed: rxitect.config.qsar_data_config.Processed`
    :

    `raw: rxitect.config.qsar_data_config.Raw`
    :

`Raw(path: str)`
:   Data related to the raw files associated with the data used for QSAR model training.
    
    Args:
        path: The filepath to the raw data file.

    ### Class variables

    `path: str`
    :