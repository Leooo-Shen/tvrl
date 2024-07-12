import subprocess
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl

from models.scheduler import build_scheduler
from models.mae_helper import VideoMAEModel


class VideoMAE(pl.LightningModule):
    def __init__(self, cfg, mask_ratio=0.9):
        super().__init__()
        self.save_hyperparameters()
        self.mask_ratio = mask_ratio
        self.lr = float(cfg.solver.lr) * float(cfg.data.train_batch_size) / 256
        self.epochs = cfg.trainer.epochs
        self.weight_decay = float(cfg.solver.weight_decay)
        self.optimizer_type = cfg.solver.optimizer
        self.lr_scheduler_type = cfg.solver.lr_scheduler
        self.warmup_epochs = cfg.solver.warmup_epochs

        img_size = cfg.data.img_size
        patch_spatial = cfg.model.vit.patch_spatial
        patch_temporal = cfg.model.vit.patch_temporal
        num_channels = cfg.data.channels
        num_frames = cfg.data.clip_frames
        vit_arch = cfg.model.vit.arch
        num_patches_per_frame = (img_size // patch_spatial) ** 2
        self.seq_length = (num_frames // patch_temporal) * num_patches_per_frame
        self.num_frames = num_frames

        self.mae = VideoMAEModel(
            frame_size=img_size,
            channels=num_channels,
            num_frames=num_frames,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            encoder_cfg=vit_arch,
            decoder_cfg="small",
        )

    def forward_loss(self, target, pred, mask):
        if self.mae.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def _shared_step(self, batch, stage="train"):
        x = batch["clip"]
        if not isinstance(x, list):
            x = [x]

        loss = 0
        for xi in x:
            target, pred, mask = self.mae(xi, mask_ratio=self.mask_ratio)
            loss = self.forward_loss(target, pred, mask)
        loss = loss.mean()
        self.log(f"{stage}.loss", loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        lr_scheduler = build_scheduler(
            optimizer, self.lr_scheduler_type, max_epochs=self.epochs, warmup_epochs=self.warmup_epochs
        )
        return [optimizer], [lr_scheduler]
