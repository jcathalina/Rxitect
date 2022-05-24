import pytest

from src.process_qsar_data import process_qsar_data
from hydra import compose, initialize


@pytest.mark.skip(reason="Need to refactor process_qsar_data() func to take explicit params.")
def test_expected_cols_in_processed_df() -> None:
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="qsar_data_config")
        print(cfg)
    df = process_qsar_data(cfg=cfg)
    assert df.columns == cfg.params.cols