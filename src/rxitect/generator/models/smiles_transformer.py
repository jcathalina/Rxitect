from typing import Optional
import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric, Metric
from torchmetrics.classification.accuracy import Accuracy

from rxitect.utils.types import StrDict


# TODO: Pass sampler into transformer (init only with voc.), and define in hydra as well.
class SmilesTransformer(LightningModule):
    def __init__(self,
                net: torch.nn.Module,
                max_lr: float = 1e-3,
                ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self):
        """
        Helper function that makes sure our accuracy tracker does not store accuracy from
        post-training pytorch lightning validation sanity checks.
        """
        pass

    def step(self, batch: torch.Tensor):
        batch = batch.to(self.device)
        preds = self(batch[:-1])  # self.forward(batch[:-1])  ## I don't think we should be manually calling forward outside of inf.
        loss = self.criterion(preds.transpose(0, 1).transpose(1, 2),
                              batch[1:].transpose(0, 1))
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        loss = self.step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        loss = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        loss = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self) -> None:
        "Helper function that resets metrics at the end of every train epoch"
        pass

    def on_validation_epoch_end(self) -> None:
        "Helper function that resets metrics at the end of every val epoch"
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        epochs = self.trainer.max_epochs
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.max_lr, 
                       total_steps=None, epochs=epochs, steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
                       pct_start=0.1, anneal_strategy='cos', cycle_momentum=True, 
                       base_momentum=0.85, max_momentum=0.95,
                       div_factor=1e3, final_div_factor=1e3, last_epoch=-1)
        scheduler = {"scheduler": scheduler, "interval" : "step" }
        return [optimizer], [scheduler]