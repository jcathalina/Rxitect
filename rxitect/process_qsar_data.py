import logging

import hydra
from hydra.utils import to_absolute_path as abspath
from omegaconf import DictConfig
from rxitect.data.process import construct_qsar_dataset


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    abs_raw_path = abspath(cfg.qsar_dataset.raw.path)
    abs_proc_dir = abspath(cfg.qsar_dataset.processed.dir)
    targets = cfg.qsar_dataset.targets
    cols = cfg.qsar_dataset.cols
    px_placeholder = cfg.qsar_dataset.px_placeholder

    construct_qsar_dataset(
        raw_data_path=abs_raw_path,
        targets=targets,
        cols=cols,
        px_placeholder=px_placeholder,
        tsplit_year=2015,  # TODO: Add to config
        random_state=42,  # TODO: Add to config
        negative_samples=True,  # TODO: Add to config,
        out_dir=abs_proc_dir,
    )


if __name__ == "__main__":
    main()
