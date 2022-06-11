
from typing import Tuple

from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split


def train_test_val_split(
    X: ArrayLike,
    y: ArrayLike,
    train_size: float,
    test_size: float,
    val_size: float,
    random_state: int = 42,
) -> Tuple[ArrayLike, ...]:
    """Helper function that extends scikit-learn's train test split to also accomodate validation set creation.

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
    """
    assert train_size + test_size + val_size == 1
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1 - train_size, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test,
        y_test,
        test_size=test_size / (test_size + val_size),
        random_state=random_state,
    )

    return X_train, X_test, X_val, y_train, y_test, y_val
