import logging
import os
import hydra
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig

from rxitect.data.process import construct_qsar_dataset

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    abs_raw_path = abspath(cfg.qsar_dataset.raw.dir)
    abs_proc_dir = abspath(cfg.qsar_dataset.processed.dir)
    targets = cfg.qsar_dataset.targets
    usecols = cfg.qsar_dataset.cols
    dummy_pchembl_value = cfg.qsar_dataset.px_placeholder

    for target in targets:
        construct_qsar_dataset(
            raw_data_path=os.path.join(abs_raw_path, "qsar_data.csv"),
            target=target,
            usecols=usecols,
            dummy_pchembl_value=dummy_pchembl_value,
            out_dir=abs_proc_dir,
        )


if __name__ == "__main__":
    main()
