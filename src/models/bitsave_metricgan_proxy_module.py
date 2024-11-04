import json
import subprocess
from itertools import chain
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Tuple

import numpy as np
import torch
import wandb
import yuvio
from einops import rearrange
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric
from torchvision.transforms.v2 import Compose

from src.data.components.transforms import rgb_to_ycbcr
from src.models.components.JPEG_proxy.encode_decode_intra import EncodeDecodeIntra

vmaf_command = "vmaf --reference {reference} --distorted {distorted} -w {width} -h {height} -p 420 -b 8 --threads {threads} --json -o {output}"


def to_255(x: Tensor) -> Tensor:
    # Scale the tensor to the range [0, 255] for uint8 representation
    return x.mul(torch.iinfo(torch.uint8).max + 1.0 - 1e-3)


def vmaf_metric(distorted_y: Tensor, reference_y: Tensor) -> Tensor:
    """Calculates the Video Multi-Method Assessment Fusion (VMAF) metric between distorted and
    reference frames.

    Args:
        distorted_y (Tensor): Tensor containing distorted y. (B, 1, H, W)
        reference_y (Tensor): Tensor containing reference y. (B, 1, H, W)

    Returns:
        Tensor: Tensor containing the VMAF scores for each pair of distorted and reference frames. (B,)
    """
    frame_size = distorted_y.shape[-2:]
    u = np.zeros((frame_size[1] // 2, frame_size[0] // 2), dtype=np.uint8)
    v = np.zeros((frame_size[1] // 2, frame_size[0] // 2), dtype=np.uint8)

    vmaf_list = []
    with TemporaryDirectory() as d:
        d = Path(d)

        for dis_y, ref_y in zip(distorted_y, reference_y):
            distorted_writer = yuvio.get_writer(
                d / "distorted", frame_size[1], frame_size[0], "yuv420p"
            )
            reference_writer = yuvio.get_writer(
                d / "reference", frame_size[1], frame_size[0], "yuv420p"
            )

            dis_y = dis_y.detach().cpu()
            ref_y = ref_y.detach().cpu()
            dis_y = to_255(dis_y).to(torch.uint8).numpy()
            ref_y = to_255(ref_y).to(torch.uint8).numpy()
            distorted_writer.write(yuvio.frame((dis_y[0], u, v), "yuv420p"))
            reference_writer.write(yuvio.frame((ref_y[0], u, v), "yuv420p"))

            check_call(
                vmaf_command.format(
                    reference=str(d / "reference"),
                    distorted=str(d / "distorted"),
                    width=frame_size[1],
                    height=frame_size[0],
                    output=str(d / "report"),
                    threads=4,
                ).split(),
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
            )

            with (d / "report").open() as f:
                report = json.load(f)

            vmaf_list.append(report["pooled_metrics"]["vmaf"]["mean"])

    return torch.Tensor(vmaf_list).to(distorted_y.device)


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
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        augmentation: Compose,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler,
        scheduler_d: torch.optim.lr_scheduler,
        proxy: torch.nn.Module = None,
        gan_loss_weight: float = 0.01,
        rate_weight: float = 0.01,
        un_compress_weight: float = 0.01,
    ):
        """Initializes the BitSaveMetricGAN model.

        Args:
            generator (torch.nn.Module): The generator model.
            discriminator (torch.nn.Module): The discriminator model.
            augmentation (Compose): The data augmentation pipeline.
            optimizer_g (torch.optim.Optimizer): The optimizer for the generator.
            optimizer_d (torch.optim.Optimizer): The optimizer for the discriminator.
            scheduler_g (torch.optim.lr_scheduler): The learning rate scheduler for the generator.
            scheduler_d (torch.optim.lr_scheduler): The learning rate scheduler for the discriminator.
            proxy (torch.nn.Module, optional): The proxy model. Defaults to None.
            gan_loss_weight (float, optional): The weight of the GAN loss. Defaults to 0.01.
            rate_weight (float, optional): The weight of the rate loss. Defaults to 0.01.
            un_compress_weight (float, optional): The weight of the uncompressed loss. Defaults to 0.01.
        """
        super().__init__()

        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        if proxy is None:
            proxy = EncodeDecodeIntra()

        self.generator = generator
        self.discriminator = discriminator
        self.proxy = proxy
        self.augmentation = augmentation
        self.gan_loss_weight = gan_loss_weight
        self.rate_weight = rate_weight
        self.un_compress_weight = un_compress_weight

        # loss function
        self.l1_loss = torch.nn.L1Loss()

        # for averaging loss across batches
        self.train_g_total_loss = MeanMetric()
        self.train_g_l1_loss = MeanMetric()
        self.train_g_adversarial_loss = MeanMetric()
        self.train_g_un_compress_l1_loss = MeanMetric()
        self.train_rate = MeanMetric()
        self.train_d_total_loss = MeanMetric()
        self.train_d_compressed_loss = MeanMetric()
        self.train_d_original_loss = MeanMetric()

        self.val_g_l1_loss = MeanMetric()
        self.val_g_un_compress_l1_loss = MeanMetric()
        self.val_rate = MeanMetric()
        self.val_compressed_vmaf = MeanMetric()

        self.test_compressed_vmaf = MeanMetric()
        self.test_g_l1_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_compressed_vmaf_best = MaxMetric()

    def process_batch(self, batch: Tensor):
        batch = self.augmentation(batch)
        batch = rgb_to_ycbcr(batch)
        batch_y = batch[:, :1, ...]

        return batch_y

    def forward(self, x: Tensor) -> Tensor:
        processed_y = self.generator(x)

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
        self.val_g_l1_loss.reset()
        self.val_compressed_vmaf.reset()
        self.val_compressed_vmaf_best.reset()

    def discriminator_step(self, compressed_y: Tensor, original_y: Tensor) -> Tensor:
        original_est_score = self.discriminator(torch.cat([original_y, original_y], dim=1))
        compressed_est_score = self.discriminator(
            torch.cat([compressed_y.detach(), original_y], dim=1)
        )

        original_real_score = torch.ones_like(
            original_est_score
        )  # use original_y as distorted vmaf not 100, is 99.7
        compressed_real_score = vmaf_metric(compressed_y, original_y) / 100
        compressed_real_score = compressed_real_score.unsqueeze(1)

        original_loss = self.l1_loss(original_est_score, original_real_score)
        compressed_loss = self.l1_loss(compressed_est_score, compressed_real_score)
        total_loss = original_loss + compressed_loss

        return total_loss, compressed_loss, original_loss

    def generator_step(
        self, processed_y: Tensor, compressed_y: Tensor, original_y: Tensor, rate
    ) -> Tensor:
        """Performs a single step of the generator during training.

        Args:
            processed_y (Tensor): The generated output tensor. (B, 1, H, W)
            compressed_y (Tensor): The compressed output tensor. (B, 1, H, W)
            original_y (Tensor): The original output tensor. (B, 1, H, W)

        Returns:
            Tensor: The total loss, L1 loss, and adversarial loss.
        """
        l1_loss = self.l1_loss(compressed_y, original_y)
        un_compress_l1_loss = self.l1_loss(processed_y, original_y)
        compressed_est_score = self.discriminator(torch.cat([compressed_y, original_y], dim=1))
        compressed_target_score = torch.ones_like(compressed_est_score)
        adversarial_loss = self.l1_loss(compressed_est_score, compressed_target_score)
        total_loss = (
            l1_loss
            + adversarial_loss * self.gan_loss_weight
            + rate * self.rate_weight
            + un_compress_l1_loss * self.un_compress_weight
        )

        return total_loss, l1_loss, adversarial_loss, un_compress_l1_loss

    def training_step(self, batch: Tensor, batch_idx: int):
        optimizer_g, optimizer_d = self.optimizers()

        original_y = self.process_batch(batch)

        processed_y, compressed_y, rate = self.forward(original_y)

        optimizer_d.zero_grad()
        d_total_loss, d_compressed_loss, d_original_loss = self.discriminator_step(
            compressed_y, original_y
        )
        self.manual_backward(d_total_loss)
        self.clip_gradients(optimizer_d, gradient_clip_val=5.0, gradient_clip_algorithm="norm")
        optimizer_d.step()

        optimizer_g.zero_grad()
        g_total_loss, g_l1_loss, g_adversarial_loss, g_un_compress_l1_loss = self.generator_step(
            processed_y,
            compressed_y,
            original_y,
            rate,
        )
        self.manual_backward(g_total_loss)
        self.clip_gradients(optimizer_g, gradient_clip_val=5.0, gradient_clip_algorithm="norm")
        optimizer_g.step()

        self.train_g_total_loss(g_total_loss)
        self.train_g_l1_loss(g_l1_loss)
        self.train_g_adversarial_loss(g_adversarial_loss)
        self.train_g_un_compress_l1_loss(g_un_compress_l1_loss)
        self.train_rate(rate)
        self.train_d_total_loss(d_total_loss)
        self.train_d_compressed_loss(d_compressed_loss)
        self.train_d_original_loss(d_original_loss)

        # Log all losses
        self.log(
            "train/g_total_loss",
            self.train_g_total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        loss_dict = {
            "train/g_l1_loss": self.train_g_l1_loss,
            "train/g_adversarial_loss": self.train_g_adversarial_loss,
            "train/g_un_compress_l1_loss": self.train_g_un_compress_l1_loss,
            "train/rate": self.train_rate,
            "train/d_total_loss": self.train_d_total_loss,
            "train/d_compressed_loss": self.train_d_compressed_loss,
            "train/d_original_loss": self.train_d_original_loss,
        }
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self):
        pass

    def validation_step(self, batch: Tensor, batch_idx: int):
        original_y = self.process_batch(batch)
        (
            processed_y,
            compressed_y,
            rate,
        ) = self.forward(original_y)
        g_l1_loss = self.l1_loss(compressed_y, original_y)
        un_compress_l1_loss = self.l1_loss(processed_y, original_y)

        compressed_real_vmaf = vmaf_metric(compressed_y, original_y).mean()

        # update and log metrics
        self.val_g_l1_loss(g_l1_loss)
        self.val_g_un_compress_l1_loss(un_compress_l1_loss)
        self.val_rate(rate)
        self.val_compressed_vmaf(compressed_real_vmaf)

        loss_dict = {
            "val/g_l1_loss": self.val_g_l1_loss,
            "val/g_un_compress_l1_loss": self.val_g_un_compress_l1_loss,
            "val/rate": self.val_rate,
            "val/generated_vmaf": self.val_compressed_vmaf,
        }

        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False)

        if batch_idx == 0:
            self.logger.experiment.log(
                {
                    "val/y": [
                        wandb.Image(original_y[:10, ...], caption="Original Y"),
                        wandb.Image(compressed_y[:10, ...], caption="Compressed Y"),
                    ]
                }
            )

    def on_validation_epoch_end(self):
        un_compress_l1_loss = self.val_g_un_compress_l1_loss.compute()
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step(un_compress_l1_loss)
        scheduler_d.step(un_compress_l1_loss)

        compressed_vmaf = self.val_compressed_vmaf.compute()  # get current val acc
        self.val_compressed_vmaf_best(compressed_vmaf)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch

        self.log(
            "val/loss_best",
            self.val_compressed_vmaf_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        original_y = self.process_batch(batch)

        processed_y, compressed_y, rate = self.forward(original_y)
        compressed_real_vmaf = vmaf_metric(compressed_y, original_y).mean()
        g_l1_loss = self.l1_loss(compressed_y, original_y)

        # update and log metrics
        self.test_compressed_vmaf(compressed_real_vmaf)
        self.test_g_l1_loss(g_l1_loss)
        self.log(
            "test/generated_vmaf",
            self.test_compressed_vmaf,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test/g_l1_loss",
            self.test_g_l1_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_test_epoch_end(self):
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer_g = self.hparams.optimizer_g(
            params=chain(self.generator.parameters(), self.proxy.parameters())
        )
        optimizer_d = self.hparams.optimizer_d(params=self.discriminator.parameters())
        scheduler_g = self.hparams.scheduler_g(optimizer=optimizer_g)
        scheduler_d = self.hparams.scheduler_d(optimizer=optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


if __name__ == "__main__":
    _ = BitSaveLitModule(None, None, None)
