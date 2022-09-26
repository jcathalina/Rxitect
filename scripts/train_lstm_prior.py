import pytorch_lightning as pl

from rxitect.models import LSTMGenerator
from rxitect.data import SmilesDataModule
from pytorch_lightning.profilers import AdvancedProfiler
from pyprojroot import here
# from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    # cli = LightningCLI(LSTMGenerator)
    lr = 1e-3
    epochs = 5

    net = LSTMGenerator(
        vocabulary_filepath=here() / "data/processed/chembl_v30_smi_voc.txt",
    )
    dm = SmilesDataModule(
        dataset_filepath=here() / "data/processed/chembl_v30_clean.smi",
        tokenizer=net.tokenizer,
        num_workers=4,
    )

    profiler = AdvancedProfiler(dirpath=here() / "logs", filename="perf_logs_lstm")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        profiler=profiler,
        check_val_every_n_epoch=1,
    )
    trainer.fit(net, datamodule=dm)
