import subprocess

import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics
from models.scheduler import build_scheduler


class Evaluator(pl.LightningModule):
    """A linear layer on top of the latent representation.
    If classification, then use CrossEntropyLoss.
    If regression, then use MSELoss.
    """

    def __init__(self, encoder, cfg, is_regression=False, ensemble=False, encoder_feat_op="cls"):
        super().__init__()
        self.encoder = encoder
        in_dim = encoder.embed_dim
        self.lr = float(cfg.solver.lr) * float(cfg.data.train_batch_size) / 256
        self.optimizer_type = cfg.solver.optimizer
        self.weight_decay = float(cfg.solver.weight_decay)
        self.lr_scheduler_type = cfg.solver.lr_scheduler
        self.epochs = cfg.trainer.epochs
        self.warmup_epochs = cfg.solver.warmup_epochs
        self.task = "regression" if is_regression else "classification"
        self.is_regression = is_regression
        self.ensemble = ensemble
        self.encoder_feat_op = encoder_feat_op

        if self.task == "classification":
            num_classes = cfg.data.num_classes
            self.criterion = nn.CrossEntropyLoss()
            self.head = nn.Linear(in_dim, num_classes)
            task = "binary" if num_classes == 2 else "multiclass"
            self.auc_train = torchmetrics.AUROC(task=task, num_classes=num_classes)
            self.auc_val = torchmetrics.AUROC(task=task, num_classes=num_classes)
            self.auc_test = torchmetrics.AUROC(task=task, num_classes=num_classes)

        elif self.task == "regression":
            self.criterion = nn.MSELoss()
            self.head = nn.Linear(in_dim, 1)
            self.mae_train = torchmetrics.MeanAbsoluteError()
            self.mae_val = torchmetrics.MeanAbsoluteError()
            self.mae_test = torchmetrics.MeanAbsoluteError()

    def forward(self, x, attn_mask=None, time_step=None):
        x = self.encoder(x, attn_mask=attn_mask, time_step=time_step, feat_op=self.encoder_feat_op)
        return self.head(x)

    def _shared_step(self, batch, batch_idx, stage="train"):
        x = batch["clip"]
        y = batch["label"]
        attn_mask = batch["pad_mask"]
        time_step = batch["time_step"]

        y_hat = self(x, attn_mask, time_step).squeeze()
        if isinstance(y, list):
            y = torch.cat(y, dim=0)
        loss = self.criterion(y_hat, y)

        if stage != "test":
            self.log(f"{stage}.loss", loss, on_epoch=True, on_step=False)

        if self.task == "classification":
            y_hat = torch.softmax(y_hat.detach(), dim=1)
            y_hat = y_hat[:, 1]
            if stage == "train":
                self.auc_train(y_hat, y)
            elif stage == "val":
                self.auc_val(y_hat, y)
            elif stage == "test":
                self.auc_test(y_hat, y)

        elif self.task == "regression":
            y_hat = y_hat.detach()
            if stage == "train":
                self.mae_train(y_hat, y)
            elif stage == "val":
                self.mae_val(y_hat, y)
            elif stage == "test":
                self.mae_test(y_hat, y)

        # self._gpu_util()
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, stage="val")

    def test_step(self, batch, batch_idx):
        # average the logits from multiple clips during testing
        if self.ensemble:
            x = batch["clip"]
            y = batch["label"]
            attn_mask = batch["pad_mask"]
            time_step = batch["time_step"]

            B, CLIPS, C, T, H, W = x.shape
            x = x.reshape(B * CLIPS, C, T, H, W)
            attn_mask = attn_mask.reshape(B * CLIPS, T)
            time_step = time_step.reshape(B * CLIPS, T)

            y_hat = self(x, attn_mask, time_step).detach()

            # average the logits from multiple clips
            y_hat = y_hat.reshape(B, CLIPS, -1)
            y_hat = torch.mean(y_hat, dim=1).squeeze()

            if self.task == "classification":
                y_hat = torch.softmax(y_hat, dim=1)
                y_hat = y_hat[:, 1]
                self.auc_test(y_hat, y)
            elif self.task == "regression":
                self.mae_test(y_hat, y)

            # average the softmax from multiple clips
            # y_hat = torch.softmax(y_hat, dim=1)
            # y_hat = y_hat[:, 1]
            # y_hat = y_hat.reshape(B, CLIPS, -1)
            # y_hat = torch.mean(y_hat, dim=1).squeeze()
            # self.auc_test(y_hat, y)

        else:
            self._shared_step(batch, batch_idx, stage="test")

    def on_train_epoch_end(self):
        if self.task == "classification":
            self.log("train.auc", self.auc_train.compute())
        elif self.task == "regression":
            self.log("train.mae", self.mae_train.compute())

    def on_validation_epoch_end(self):
        if self.task == "classification":
            self.log("val.auc", self.auc_val.compute())
        elif self.task == "regression":
            self.log("val.mae", self.mae_val.compute())

    def on_test_epoch_end(self):
        if self.task == "classification":
            self.log("test.auc", self.auc_test.compute())
        elif self.task == "regression":
            self.log("test.mae", self.mae_test.compute())

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.is_regression:
            mode = "min"
            monitor = "val.mae"
        else:
            mode = "max"
            monitor = "val.auc"

        lr_scheduler = build_scheduler(
            optimizer,
            self.lr_scheduler_type,
            max_epochs=self.epochs,
            warmup_epochs=self.warmup_epochs,
            mode=mode,
            monitor=monitor,
        )
        return [optimizer], [lr_scheduler]

    def _gpu_util(self):
        # Display GPU memory usage and utilization rate
        command = "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,nounits,noheader"
        output = subprocess.check_output(command.split()).decode("utf-8").strip().split("\n")
        for line in output:
            used_memory, total_memory, gpu_utilization = line.split(",")
            self.log(
                "GPU Mem",
                int(used_memory) / int(total_memory),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
            self.log(
                "GPU Usage",
                float(gpu_utilization),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
