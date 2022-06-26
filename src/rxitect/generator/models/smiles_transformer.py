import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from rxitect.utils.types import StrDict


class SmilesTransformer(LightningModule):
    def __init__(self,
                net: torch.nn.Module,
                # n_tokens: int,
                # d_model: int = 256,
                # nhead: int = 8,
                # num_encoder_layers: int = 4,
                # dim_feedforward: int = 1024,
                # dropout: float = 0.1,
                # activation: str = "relu",
                # max_length: int = 1000,
                max_lr: float = 1e-3,
                ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])

        self.net = net
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def on_train_start(self):
        """
        Helper function that makes sure our accuracy tracker does not store accuracy from
        post-training pytorch lightning validation sanity checks.
        """
        self.val_acc_best.reset()

    def step(self, batch: torch.Tensor):
        batch = batch.to(self.device)
        preds = self.forward(batch[:-1])
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
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def on_validation_epoch_end(self) -> None:
        "Helper function that resets metrics at the end of every val epoch"
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

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