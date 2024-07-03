from typing import Tuple

import torch
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric, MinMetric
from torchvision.transforms.v2 import Compose

from src.data.components.transforms import rgb_to_ycbcr


def to_255(x: Tensor) -> Tensor:
    return x.mul(torch.iinfo(torch.uint8).max + 1.0 - 1e-3)


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
        proxy: torch.nn.Module,
        augmentation: Compose,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        rate_weight: float,
        # lambda1: float = 10.0,
        # lambda2: float = 0.1,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.proxy = proxy
        self.augmentation = augmentation

        self.rate_weight = rate_weight

        # loss function
        self.l1_loss = torch.nn.L1Loss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_l1_loss = MeanMetric()
        self.train_un_compressed_l1_loss = MeanMetric()
        self.train_rate = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_l1_loss = MeanMetric()
        self.val_un_compressed_l1_loss = MeanMetric()
        self.val_rate = MeanMetric()

        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def process_batch(self, batch: Tensor):
        batch = self.augmentation(batch)
        batch = rgb_to_ycbcr(batch)
        batch_y = batch[:, :1, ...]

        return batch_y

    def forward(self, x: Tensor) -> Tensor:
        processed_y = self.net(x)

        compressed_y = rearrange(processed_y, "b c h w -> b h w c")

        compressed_y = to_255(compressed_y)
        compressed_y, rate = self.proxy(compressed_y)
        rate = rate.sum() / x.shape.numel()
        compressed_y = compressed_y * (1.0 / torch.iinfo(torch.uint8).max)

        compressed_y = rearrange(compressed_y, "b h w c -> b c h w")

        return processed_y, compressed_y, rate

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_l1_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Tuple[Tensor]) -> Tensor:
        original_y = batch  # only y

        processed_y, compressed_y, rate = self.forward(original_y)

        rate = rate.sum() / original_y.shape.numel()

        l1_loss = self.l1_loss(compressed_y, original_y)
        un_compressed_l1_loss = self.l1_loss(processed_y, original_y)
        loss = l1_loss + self.rate_weight * processed_y
        return loss, l1_loss, un_compressed_l1_loss, rate

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        batch = self.process_batch(batch)
        loss, l1_loss, un_compressed_l1_loss, rate = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.train_l1_loss(l1_loss)
        self.train_rate(rate)
        self.train_un_compressed_l1_loss(un_compressed_l1_loss)

        loss_dict = {
            "train/l1_loss": l1_loss,
            "train/rate": rate,
            "train/un_compressed_l1_loss": un_compressed_l1_loss,
        }

        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        loss, l1_loss, un_compressed_l1_loss, rate = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        self.val_l1_loss(l1_loss)
        self.val_un_compressed_l1_loss(un_compressed_l1_loss)
        self.val_rate(rate)

        loss_dict = {
            "val/l1_loss": l1_loss,
            "val/un_compressed_l1_loss": un_compressed_l1_loss,
            "val/rate": rate,
        }

        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_epoch_end(self):
        loss = self.val_loss.compute()  # get current val acc
        self.val_loss_best(loss)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/loss_best", self.val_loss_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        loss, _, _, _ = self.model_step(batch)

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
