Module rxitect.data.utils
=========================

Functions
---------

    
`train_test_val_split(X: ArrayLike, y: ArrayLike, train_size: float, test_size: float, val_size: float, random_state: int = 42) ‑> Tuple[Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]], ...]`
:   Helper function that extends scikit-learn's train test split to also accomodate validation set creation.
    
    Args:
        X: Array-like containing the full dataset without labels
        y: Array-like containing all the dataset labels
        train_size: The fraction of the data that should be reserved for training
        test_size: The fraction of the data that should be reserved for testing
        val_size: The fraction of the data that should be reserved for validation
        random_state: The random seed number used to enforce reproducibility
    
    Returns:
        A tuple containing all the train/test/val data in the following order:
        X_train, X_test, X_val, y_train, y_test, y_val

Classes
-------

`QSARDataset(df_train: pd.DataFrame, df_test: pd.DataFrame, targets: List[str])`
:   Class representing the dataset used to train QSAR models

    ### Class variables

    `df_test: pandas.core.frame.DataFrame`
    :

    `df_train: pandas.core.frame.DataFrame`
    :

    `targets: List[str]`
    :

    ### Static methods

    `load_from_file(train_file: str, test_file: str) ‑> rxitect.data.utils.QSARDataset`
    :

    ### Methods

    `X_test(self, target_chembl_id: str) ‑> numpy.ndarray`
    :   Lazily evaluates the test data points for a given target ChEMBL ID
        
        Args:
            target_chembl_id:
        
        Returns:
            An array containing the fingerprints of all test data points for the given target ChEMBL ID

    `X_train(self, target_chembl_id: str) ‑> numpy.ndarray`
    :   Lazily evaluates the train data points for a given target ChEMBL ID
        
        Args:
            target_chembl_id:
        
        Returns:
            An array containing the fingerprints of all train data points for the given target ChEMBL ID

    `get_classifier_labels(self, target_chembl_id: str) ‑> Tuple[Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]], Union[numpy._array_like._SupportsArray[numpy.dtype], numpy._nested_sequence._NestedSequence[numpy._array_like._SupportsArray[numpy.dtype]], bool, int, float, complex, str, bytes, numpy._nested_sequence._NestedSequence[Union[bool, int, float, complex, str, bytes]]]]`
    :

    `get_train_test_data(self, target_chembl_id: str) ‑> Tuple[numpy.ndarray, ...]`
    :

    `y_test(self, target_chembl_id: str) ‑> numpy.ndarray`
    :   Lazily evaluates the test labels for a given target ChEMBL ID
        
        Args:
            target_chembl_id:
        
        Returns:
            An array containing the pChEMBL value of all test data points for the given target ChEMBL ID

    `y_train(self, target_chembl_id: str) ‑> numpy.ndarray`
    :   Lazily evaluates the train labels for a given target ChEMBL ID
        
        Args:
            target_chembl_id:
        
        Returns:
            An array containing the pChEMBL value of all train data points for the given target ChEMBL ID

`QSARModel(value, names=None, *, module=None, qualname=None, type=None, start=1)`
:   An enumeration.

    ### Ancestors (in MRO)

    * builtins.str
    * enum.Enum

    ### Class variables

    `RF`
    :

    `SVR`
    :

    `XGB`
    :