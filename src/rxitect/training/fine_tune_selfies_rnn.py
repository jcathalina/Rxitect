import logging
import pathlib
from pathlib import Path

import mlflow
import pytorch_lightning as pl

from globals import root_path
from rxitect.data.ligand_corpus import LigandCorpus
from rxitect.structs.vocabulary import SelfiesVocabulary
from rxitect.models.lightning.generator import Generator
from rxitect.log_utils import print_auto_logged_info


def fine_tune(
    epochs: int = 1_000,
    dev: bool = False,
    n_workers: int = 2,
    n_gpus: int = 1
):
    vocabulary = SelfiesVocabulary(
        vocabulary_file_path=(root_path / "data/processed/selfies_voc.txt")
    )
    
    output_dir = root_path / "models"
    pretrained_lstm_path = output_dir / "pretrained_selfies_rnn.ckpt"
    fine_tuned_lstm_path = output_dir / "fine_tuned_selfies_rnn.ckpt"
    
    logging.info("Initiating ML Flow tracking...")
    mlflow.set_tracking_uri("https://dagshub.com/naisuu/Rxitect.mlflow")
    mlflow.pytorch.autolog()
    
    logging.info("Loading pre-trained LSTM model for fine-tuning...")
    # NOTE: DrugEx V2 uses a lower learning rate for fine-tuning the LSTM, which is why we do it here as well.
    prior = Generator.load_from_checkpoint(pretrained_lstm_path, vocabulary=vocabulary, lr=1e-4)

    ligand_dm = LigandCorpus(vocabulary=vocabulary, n_workers=n_workers)  # Setting multiple dataloaders is actually slower due to the small dataset.
    logging.info("Setting up Ligand Data Module...")
    ligand_dm.setup(stage="fit")

    fine_tuner = pl.Trainer(
        gpus=n_gpus,
        log_every_n_steps=1 if dev else 50,
        max_epochs=epochs,
        fast_dev_run=dev,
        default_root_dir=output_dir,
    )

    logging.info("Starting main fine-tuning run...")
    with mlflow.start_run() as run:
        fine_tuner.fit(model=prior, datamodule=ligand_dm)

    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    logging.info("Fine-tuning finished, saving fine-tuned LSTM checkpoint...")
    fine_tuner.save_checkpoint(filepath=fine_tuned_lstm_path)


if __name__ == "__main__":
    fine_tune()