import logging
from pathlib import Path

from globals import root_path
from rxitect.data.chembl_corpus import ChemblCorpus
from rxitect.structs.vocabulary import SelfiesVocabulary
from rxitect.models.lightning.generator import Generator
from rxitect.log_utils import print_auto_logged_info
import pytorch_lightning as pl


def train(
        epochs: int = 50,
        dev: bool = False,
        n_workers: int = 4,
        n_gpus: int = 1
):
    # load_dotenv()

    vocabulary = SelfiesVocabulary(
        vocabulary_file_path=root_path / "data/processed/selfies_voc.txt"
    )

    output_dir = root_path / "models"
    if not Path.exists(output_dir):
        logging.info(
            f"Creating directories to store pretraining output @ '{output_dir}'"
        )
        Path(output_dir).mkdir(parents=True)

    pretrained_lstm_path = output_dir / "pretrained_selfies_lstm.ckpt" if not dev else output_dir / "dev_pretrained_selfies_lstm.ckpt"

    print("Creating Generator")
    prior = Generator(vocabulary=vocabulary)

    # logging.info("Initiating ML Flow tracking...")
    # mlflow.set_tracking_uri("https://dagshub.com/naisuu/drugex-plus-r.mlflow")
    # mlflow.pytorch.autolog()

    print("Corpus time...")
    chembl_dm = ChemblCorpus(vocabulary=vocabulary, n_workers=n_workers, dev_run=dev)
    logging.info("Setting up ChEMBL Data Module...")
    chembl_dm.setup(stage="fit")

    print("Creating Trainer...")
    pretrainer = pl.Trainer(
        gpus=n_gpus,
        log_every_n_steps=1 if dev else 50,
        max_epochs=epochs,
        default_root_dir=output_dir,
    )

    # logging.info("Starting main pretraining run...")
    # with mlflow.start_run() as run:
    #     pretrainer.fit(model=prior, datamodule=chembl_dm)
    pretrainer.fit(model=prior, datamodule=chembl_dm)

    # print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))
    print("Training finished, saving pretrained LSTM checkpoint...")
    pretrainer.save_checkpoint(filepath=pretrained_lstm_path)


if __name__ == "__main__":
    train(dev=True, n_workers=1)
