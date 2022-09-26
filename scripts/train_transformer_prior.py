import pytorch_lightning as pl

from rxitect.models import TransformerGenerator
from rxitect.data import SmilesDataModule
from rxitect.tokenizers import get_tokenizer
from pytorch_lightning.profilers import AdvancedProfiler
from pyprojroot import here


if __name__ == "__main__":
    lr = 1e-3
    epochs = 5

    tokenizer = get_tokenizer("smiles", here() / "data/processed/chembl_v30_smi_voc.txt")
    net = TransformerGenerator(n_tokens=tokenizer.vocabulary_size_,
                               d_model=512,
                               max_lr=8e-4)
    dm = SmilesDataModule(
        dataset_filepath=here() / "data/processed/chembl_v30_clean.smi",
        tokenizer=tokenizer,
        num_workers=4,
        batch_size=256,
    )

    profiler = AdvancedProfiler(dirpath=here() / "logs", filename="perf_logs_transformer")
    lr_logger =pl.callbacks.lr_monitor.LearningRateMonitor(logging_interval="epoch")
    tb_logger = pl.loggers.TensorBoardLogger('tensorboard_logs/')

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=epochs,
        profiler=profiler,
        check_val_every_n_epoch=1,
        callbacks=[lr_logger],
        logger=tb_logger,
    )
    trainer.fit(net, datamodule=dm)
