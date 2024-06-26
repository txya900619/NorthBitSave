import json
import subprocess
from pathlib import Path
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Tuple

import numpy as np
import torch
import yuvio
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MaxMetric, MeanMetric

vmaf_command = "vmaf --reference {reference} --distorted {distorted} -w {width} -h {height} -p 420 -b 8 --json -o {output}"


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
            dis_y = (dis_y * 255).clamp(min=0, max=255).numpy().astype(np.uint8)
            ref_y = (ref_y * 255).clamp(min=0, max=255).numpy().astype(np.uint8)
            distorted_writer.write(yuvio.frame((dis_y[0], u, v), "yuv420p"))
            reference_writer.write(yuvio.frame((ref_y[0], u, v), "yuv420p"))

            check_call(
                vmaf_command.format(
                    reference=str(d / "reference"),
                    distorted=str(d / "distorted"),
                    width=frame_size[1],
                    height=frame_size[0],
                    output=str(d / "report"),
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
        proxy: torch.nn.Module,
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        scheduler_g: torch.optim.lr_scheduler,
        scheduler_d: torch.optim.lr_scheduler,
        gan_loss_weight: float = 0.01,
        rate_weight: float = 0.01,
    ):
        """Initializes the BitSaveMetricGAN model.

        Args:
            generator (torch.nn.Module): The generator model.
            discriminator (torch.nn.Module): The discriminator model.
            optimizer_g (torch.optim.Optimizer): The optimizer for the generator.
            optimizer_d (torch.optim.Optimizer): The optimizer for the discriminator.
            scheduler_g (torch.optim.lr_scheduler): The learning rate scheduler for the generator.
            scheduler_d (torch.optim.lr_scheduler): The learning rate scheduler for the discriminator.
            gamma (float, optional): The scaler for the adversarial loss. Defaults to 0.01.
        """
        super().__init__()

        self.automatic_optimization = False

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.generator = generator
        self.discriminator = discriminator
        self.proxy = proxy
        self.gan_loss_weight = gan_loss_weight
        self.rate_weight = rate_weight

        # loss function
        self.l1_loss = torch.nn.L1Loss()

        # for averaging loss across batches
        self.train_g_total_loss = MeanMetric()
        self.train_g_l1_loss = MeanMetric()
        self.train_g_adversarial_loss = MeanMetric()
        self.train_rate = MeanMetric()
        self.train_d_total_loss = MeanMetric()
        self.train_d_generated_loss = MeanMetric()
        self.train_d_original_loss = MeanMetric()

        self.val_g_l1_loss = MeanMetric()
        self.val_rate = MeanMetric()
        self.val_generated_vmaf = MeanMetric()

        self.test_generated_vmaf = MeanMetric()
        self.test_g_l1_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_generated_vmaf_best = MaxMetric()

    def forward(self, x: Tensor) -> Tensor:
        generated_y = self.generator(x)

        generated_y = generated_y.permute(0, 2, 3, 1)  # to (B, H, W, C)

        generated_y = generated_y.mul(torch.iinfo(torch.uint8).max + 1.0 - 1e-3)
        generated_y, rate = self.proxy(generated_y)
        generated_y = generated_y / (1.0 / torch.iinfo(torch.uint8).max)

        generated_y = generated_y.permute(0, 3, 1, 2)  # to (B, C, H, W)

        return generated_y, rate

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_g_l1_loss.reset()
        self.val_generated_vmaf.reset()
        self.val_generated_vmaf_best.reset()

    def discriminator_step(self, generated_y: Tensor, original_y: Tensor) -> Tensor:
        original_est_score = self.discriminator(torch.cat([original_y, original_y], dim=1))
        generated_est_score = self.discriminator(
            torch.cat([generated_y.detach(), original_y], dim=1)
        )

        original_real_score = torch.ones_like(
            original_est_score
        )  # use original_y as distorted vmaf not 100, is 99.7
        generated_real_score = vmaf_metric(generated_y, original_y) / 100
        generated_real_score = generated_real_score.unsqueeze(1)

        original_loss = self.l1_loss(original_est_score, original_real_score)
        generated_loss = self.l1_loss(generated_est_score, generated_real_score)
        total_loss = original_loss + generated_loss

        return total_loss, generated_loss, original_loss

    def generator_step(self, generated_y: Tensor, original_y: Tensor, rate) -> Tensor:
        """Performs a single step of the generator during training.

        Args:
            generated_y (Tensor): The generated output tensor. (B, 1, H, W)
            original_y (Tensor): The original output tensor. (B, 1, H, W)

        Returns:
            Tensor: The total loss, L1 loss, and adversarial loss.
        """
        l1_loss = self.l1_loss(generated_y, original_y)
        generated_est_score = self.discriminator(
            torch.cat([generated_y.detach(), original_y], dim=1)
        )
        generated_targe_score = torch.ones_like(generated_est_score)
        adversarial_loss = self.l1_loss(generated_est_score, generated_targe_score)
        total_loss = (
            l1_loss + adversarial_loss * self.gan_loss_weight + rate.mean() * self.rate_weight
        )

        return total_loss, l1_loss, adversarial_loss

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        optimizer_g, optimizer_d = self.optimizers()

        original_y, _, _, _ = batch
        generated_y, rate = self.forward(original_y)

        optimizer_d.zero_grad()
        d_total_loss, d_generated_loss, d_original_loss = self.discriminator_step(
            generated_y, original_y
        )
        self.manual_backward(d_total_loss)
        self.clip_gradients(optimizer_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        optimizer_d.step()

        optimizer_g.zero_grad()
        g_total_loss, g_l1_loss, g_adversarial_loss = self.generator_step(
            generated_y, original_y, rate
        )
        self.manual_backward(g_total_loss)
        self.clip_gradients(optimizer_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        optimizer_g.step()

        self.train_g_total_loss(g_total_loss)
        self.train_g_l1_loss(g_l1_loss)
        self.train_g_adversarial_loss(g_adversarial_loss)
        self.train_rate(rate)
        self.train_d_total_loss(d_total_loss)
        self.train_d_generated_loss(d_generated_loss)
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
            "train/rate": self.train_rate,
            "train/d_total_loss": self.train_d_total_loss,
            "train/d_generated_loss": self.train_d_generated_loss,
            "train/d_original_loss": self.train_d_original_loss,
        }
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False)

    def on_train_epoch_end(self):
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        original_y, _, _, _ = batch
        generated_y, rate = self.forward(original_y)
        g_l1_loss = self.l1_loss(generated_y, original_y)

        generated_real_vmaf = vmaf_metric(generated_y, original_y).mean()

        # update and log metrics
        self.val_g_l1_loss(g_l1_loss)
        self.val_generated_vmaf(generated_real_vmaf)

        self.log(
            "val/g_l1_loss",
            self.val_g_l1_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            "val/rate",
            self.val_rate,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.log(
            "val/generated_vmaf",
            self.val_generated_vmaf,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def on_validation_epoch_end(self):
        generated_vmaf = self.val_generated_vmaf.compute()  # get current val acc
        self.val_generated_vmaf_best(generated_vmaf)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log(
            "val/loss_best",
            self.val_generated_vmaf_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx: int):
        original_y, _, _, _ = batch
        generated_y, rate = self.forward(original_y)
        generated_real_vmaf = vmaf_metric(generated_y, original_y).mean()
        g_l1_loss = self.l1_loss(generated_y, original_y)

        # update and log metrics
        self.test_generated_vmaf(generated_real_vmaf)
        self.test_g_l1_loss(g_l1_loss)
        self.log(
            "test/generated_vmaf",
            self.test_generated_vmaf,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "test/g_l1_loss",
            self.test_g_l1_loss,
            on_step=False,
            on_epoch=False,
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
        optimizer_g = self.hparams.optimizer_g(params=self.generator.parameters())
        optimizer_d = self.hparams.optimizer_d(params=self.discriminator.parameters())
        scheduler_g = self.hparams.scheduler_g(optimizer=optimizer_g)
        scheduler_d = self.hparams.scheduler_d(optimizer=optimizer_d)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


if __name__ == "__main__":
    _ = BitSaveLitModule(None, None, None)
