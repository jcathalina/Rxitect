import logging
import os
from pathlib import Path
from typing import Dict, Any

from globals import root_path
from rxitect.data.chembl_corpus import ChemblCorpus
from rxitect.structs.vocabulary import SelfiesVocabulary
from rxitect.models.lightning.generator import Generator
from rxitect.log_utils import print_auto_logged_info
import mlflow
import pytorch_lightning as pl
import yaml


logging.basicConfig(level=logging.INFO)


def train(
        hyperparams: Dict[str, Any],
        epochs: int = 50,
        dev: bool = False,
        n_workers: int = 4,
        n_gpus: int = 1,
):

    vocabulary = SelfiesVocabulary(
        vocabulary_file_path=root_path / "data/processed/selfies_voc.txt"
    )

    output_dir = root_path / "models"
    if not Path.exists(output_dir):
        logging.info(
            f"Creating directories to store pretraining output @ '{output_dir}'"
        )
        Path(output_dir).mkdir(parents=True)

    pretrained_rnn_path = output_dir / "pretrained_selfies_rnn.ckpt" if not dev else output_dir / "dev_pretrained_selfies_rnn.ckpt"

    print("Creating Generator")
    prior = Generator(**hyperparams,
                      vocabulary=vocabulary,)

    logging.info("Initiating ML Flow tracking...")
    mlflow.set_tracking_uri("https://dagshub.com/naisuu/Rxitect.mlflow")
    mlflow.pytorch.autolog()

    print("Loading Corpus...")
    chembl_dm = ChemblCorpus(vocabulary=vocabulary, n_workers=n_workers, dev_run=dev)
    logging.info("Setting up ChEMBL Data Module...")
    chembl_dm.setup(stage="fit")

    print("Creating Trainer...")
    pretrainer = pl.Trainer(
        gpus=n_gpus,
        log_every_n_steps=1 if dev else 50,
        max_epochs=epochs,
        default_root_dir=output_dir,
        # resume_from_checkpoint= root_path / "models/lightning_logs/version_4/checkpoints/epoch=14-step=44879.ckpt"
    )

    logging.info("Starting main pretraining run...")
    with mlflow.start_run() as run:
        pretrainer.fit(model=prior, datamodule=chembl_dm)

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    logging.info("Training finished, saving pretrained model checkpoint...")
    pretrainer.save_checkpoint(filepath=pretrained_rnn_path)


if __name__ == "__main__":
    settings_path = root_path / "config/selfies_rnn_settings.yml"
    if os.path.exists(settings_path):
        settings = yaml.safe_load(open(settings_path, "r"))
    else:
        raise FileNotFoundError("Expected a settings file but didn't find it.")

    train_settings = settings["train_settings"]  # TODO: Refactor to just take dictionary.

    train(epochs=train_settings["epochs"],
          dev=train_settings["dev"],
          n_workers=train_settings["n_workers"],
          n_gpus=train_settings["n_gpus"],
          hyperparams=settings["hyperparams"],)
