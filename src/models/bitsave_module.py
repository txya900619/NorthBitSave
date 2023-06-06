from typing import Any, Tuple

import torch
from lightning import LightningModule
from torch import Tensor
from torchjpeg import dct
from torchmetrics import (
    MeanMetric,
    MinMetric,
    MultiScaleStructuralSimilarityIndexMeasure,
)


def dct_loss(tensor: Tensor):
    # TODO: add n mask in paper
    dct_tensor = dct.batch_dct(tensor)
    threshold = torch.mean(torch.abs(dct_tensor))
    dct_tensor = torch.where(
        torch.abs(dct_tensor) >= threshold, torch.zeros_like(dct_tensor), dct_tensor
    )
    # return torch.mean(torch.abs(dct_tensor-torch.zeros_like(dct_tensor)))
    return torch.mean(torch.abs(dct_tensor))


class BitSaveLitModule(LightningModule):
    """A LightningModule organizes your PyTorch code into 6 sections:

        - Initialization (__init__)
        - Train Loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        lambda1: float = 10.0,
        lambda2: float = 0.1,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.l1_loss = torch.nn.L1Loss()
        self.y_ms_ssim_loss = MultiScaleStructuralSimilarityIndexMeasure(kernel_size=9)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.net(x)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Any) -> Tensor:
        input_y, input_uv, ans_y, ans_uv = batch
        logits_y, logits_uv = self.forward((input_y, input_uv))
        loss_y = (
            self.l1_loss(logits_y, ans_y)
            + self.lambda1 * dct_loss(logits_y)
            + self.lambda2 * (1 - self.y_ms_ssim_loss(logits_y, ans_y))
        )
        loss_uv = self.l1_loss(logits_uv, ans_uv) + self.lambda1 * dct_loss(logits_uv)
        loss = (2 * loss_y + loss_uv) / 3
        return loss

    def training_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = BitSaveLitModule(None, None, None)
