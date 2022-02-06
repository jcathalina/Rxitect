from unittest.mock import patch
import pytest
import pandas as pd

from rxitect.featurization import generate_selfies_encodings


@pytest.fixture()
def mock_smiles_df() -> pd.DataFrame:
    """Returns a mocked valid dataframe containing a SMILES column."""
    mock_df = pd.DataFrame({
        "index": pd.Series([0, 1, 2]),
        "smiles": pd.Series(["N1C2C1C1OC21", "CCC(=O)C#N", "OC1=NNN=N1"])
    })

    return mock_df


@patch("rxitect.featurization.pd.read_csv")
class TestGenerateSelfiesEncodings:
    """Class to test the functionality of generate_selfies_encodings."""

    def test_selfies_encodings_generated_for_valid_smiles_csv(self, mock_read_csv, mock_smiles_df):
        # setup mocked dataframe for testing
        mock_read_csv.return_value = mock_smiles_df
        
        # file path can be disregarded, return val is mocked
        encodings = generate_selfies_encodings(file_path="placeholder.csv")
        
        assert len(encodings.selfies_list) == 3