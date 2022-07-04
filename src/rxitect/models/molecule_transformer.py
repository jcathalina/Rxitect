import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule


class MoleculeTransformer(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        max_lr: float = 1e-3,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def step(self, batch: torch.Tensor):
        batch = batch.to(self.device)
        preds = self.forward(batch[:-1])
        loss = F.cross_entropy(preds.transpose(0, 1).transpose(1, 2), batch[1:].transpose(0, 1))
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

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), self.hparams.max_lr)
        # epochs = self.trainer.max_epochs
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer,
        #     max_lr=self.hparams.max_lr,
        #     total_steps=None,
        #     epochs=epochs,
        #     steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
        #     pct_start=0.1,
        #     anneal_strategy="cos",
        #     cycle_momentum=True,
        #     base_momentum=0.85,
        #     max_momentum=0.95,
        #     div_factor=1e3,
        #     final_div_factor=1e3,
        #     last_epoch=-1,
        # )
        # scheduler = {"scheduler": scheduler, "interval": "step"}
        # return [optimizer], [scheduler]
        return optimizer
