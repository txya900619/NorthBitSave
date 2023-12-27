from typing import Dict, List, Tuple

import pyiqa
import torch
import torchvision
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric, MinMetric

from src.data.components.transforms import ycbcr_to_rgb
from src.models.components.fvc import FVC, load_model


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(
        self,
        input: Tensor,
        target: Tensor,
        feature_layers: List[int] = [0, 1, 2, 3],
        style_layers: List[int] = [],
    ) -> Tensor:
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode="bilinear", size=(224, 224), align_corners=False)
            target = self.transform(target, mode="bilinear", size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss


class WithFVCLitModule(LightningModule):
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
        FVC_model_path: str,
        train_lambda: float = 0.001,
        train_gamma: float = 0.01,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.train_lambda = train_lambda
        self.train_gamma = train_gamma

        self.fvc = FVC(self.train_lambda)
        load_model(self.fvc, FVC_model_path)
        self.fvc = self.fvc.eval().cuda()
        for param in self.fvc.parameters():
            param.requires_grad = False

        # for averaging loss across batches
        self.train_rd_loss = MeanMetric()
        self.train_fidelity_loss = MeanMetric()
        self.train_bpp = MeanMetric()
        self.train_perceptual_loss = MeanMetric()
        self.train_m0_l1_loss = MeanMetric()

        self.val_rd_loss = MeanMetric()
        self.val_fidelity_loss = MeanMetric()
        self.val_bpp = MeanMetric()
        self.val_perceptual_loss = MeanMetric()
        self.val_m0_l1_loss = MeanMetric()

        self.test_rd_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_rd_loss_best = MinMetric()

    def setup(self, stage: str) -> None:
        result = super().setup(stage)
        self.hparams.topiq = pyiqa.create_metric(
            "topiq_fr", device=self.trainer.strategy.root_device, as_loss=True
        )
        return result

    def forward(
        self,
        ref_yuv: Tuple[Tensor, Tensor],
        input_yuv: Tuple[Tensor, Tensor],
        input_rgb: Tensor,
    ) -> Dict[str, Tensor]:
        ref_y, ref_uv = ref_yuv
        enh_ref_y = self.net(ref_y)
        enh_ref_rgb = ycbcr_to_rgb(enh_ref_y, ref_uv)

        input_y, input_uv = input_yuv
        enh_input_y = self.net(input_y)
        m0_l1_loss = torch.nn.functional.l1_loss(enh_input_y, input_y)
        enh_input_rgb = ycbcr_to_rgb(enh_input_y, input_uv)

        output_rgb, out = self.fvc(enh_ref_rgb, enh_input_rgb, input_rgb)

        out["fidelity_loss"] = torch.nn.functional.l1_loss(output_rgb, input_rgb)
        out["perceptual_loss"] = 1 - self.hparams.topiq(output_rgb, input_rgb)
        out["m0_l1_loss"] = m0_l1_loss
        out["rd_loss"] = (
            out["fidelity_loss"]
            + m0_l1_loss
            + out["bpp"] * self.train_lambda
            + out["perceptual_loss"] * self.train_gamma
        )

        return out

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_rd_loss.reset()
        self.val_fidelity_loss.reset()
        self.val_bpp.reset()
        self.val_rd_loss_best.reset()
        self.val_perceptual_loss.reset()
        self.val_m0_l1_loss.reset()

    def model_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]
    ) -> Dict[str, Tensor]:
        input_y, input_uv, ref_y, ref_uv, input_rgb = batch
        out = self.forward((ref_y, ref_uv), (input_y, input_uv), input_rgb)

        return out

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        out = self.model_step(batch)

        # update and log metrics
        self.train_rd_loss(out["rd_loss"])
        self.train_fidelity_loss(out["fidelity_loss"])
        self.train_bpp(out["bpp"])
        self.train_perceptual_loss(out["perceptual_loss"])
        self.train_m0_l1_loss(out["m0_l1_loss"])

        self.log(
            "train/rd_loss",
            self.train_rd_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/fidelity_loss",
            self.train_fidelity_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("train/bpp", self.train_bpp, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "train/perceptual_loss",
            self.train_perceptual_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "train/m0_l1_loss",
            self.train_m0_l1_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        # return loss or backpropagation will fail
        return out["rd_loss"]

    def on_train_epoch_end(self):
        pass

    def validation_step(
        self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int
    ):
        out = self.model_step(batch)

        # update and log metrics
        self.val_rd_loss(out["rd_loss"])
        self.val_fidelity_loss(out["fidelity_loss"])
        self.val_bpp(out["bpp"])
        self.val_perceptual_loss(out["perceptual_loss"])
        self.val_m0_l1_loss(out["m0_l1_loss"])

        self.log("val/rd_loss", self.val_rd_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/fidelity_loss",
            self.val_fidelity_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log("val/bpp", self.val_bpp, on_step=False, on_epoch=True, prog_bar=False)
        self.log(
            "val/perceptual_loss",
            self.val_perceptual_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "val/m0_l1_loss",
            self.val_m0_l1_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_epoch_end(self):
        val_rd_loss = self.val_rd_loss.compute()  # get current val rd loss
        self.val_rd_loss_best(val_rd_loss)  # update best so far val rd loss
        # log `val_rd_loss_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss_best",
            self.val_rd_loss_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        out = self.model_step(batch)

        # update and log metrics
        self.test_rd_loss(out["rd_loss"])
        self.log(
            "test/rd_loss",
            self.test_rd_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/rd_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = WithFVCLitModule(None, None, None, None, None)
