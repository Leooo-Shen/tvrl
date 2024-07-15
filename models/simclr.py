import subprocess

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from lightly.loss import NegativeCosineSimilarity, NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from torch import nn

from models.scheduler import build_scheduler
from models.utils import InfoNCELoss, token_mapping
from models.videovit import TimeEmbedding
from models.vit_helper import ViT_Backbone


def ortho_penalty(t):
    eye = torch.eye(t.shape[0]).to(t.device)
    return ((t @ t.T - eye) ** 2).sum()


class SimCLR(pl.LightningModule):
    """A standard SimCLR model with InfoNCE loss.

    Additionally, it supports Latent Time Navigation (https://arxiv.org/abs/2305.06437) for temporal encoder.
    """

    def __init__(self, encoder, cfg):
        super(SimCLR, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.lr = float(cfg.solver.lr) * float(cfg.data.train_batch_size) / 256
        self.epochs = cfg.trainer.epochs
        self.weight_decay = float(cfg.solver.weight_decay)
        self.optimizer_type = cfg.solver.optimizer
        self.lr_scheduler_type = cfg.solver.lr_scheduler
        self.warmup_epochs = cfg.solver.warmup_epochs

        self.encoder = encoder
        in_dim = encoder.embed_dim
        self.head = SimCLRProjectionHead(input_dim=in_dim)
        self.criterion = InfoNCELoss(temperature=cfg.simclr.temperature)

        self.use_ltn = cfg.get("use_ltn", False)
        if self.use_ltn:
            # v1: random init, run QR decomposition at each forward pass, as implemented in the paper
            self.Dt = nn.Parameter(torch.randn(64, in_dim))

            # v2: init with orthogonal matrix, use orthogonal loss.
            # self.Dt = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(64, in_dim)))
            # self.ortho_loss = ortho_penalty

            self.ltn_embed = nn.Sequential(nn.Linear(in_dim * 2, 2048), nn.ReLU(), nn.Linear(2048, 64))
            self.time_embed = TimeEmbedding(in_dim, learnable=True)

    def forward_ltn(self, feature, delta_t):
        time_embed = self.time_embed(delta_t).squeeze(1)
        ltn_input = torch.cat([feature, time_embed], dim=-1)
        ltn_output = self.ltn_embed(ltn_input)

        # v1: run QR decomposition at each forward pass
        Q, _ = torch.linalg.qr(self.Dt.T)
        feature += ltn_output @ Q.T

        # v2: directly on Dt
        # feature += ltn_output @ self.Dt
        return feature

    def forward(self, x, attn_mask=None, time_step=None):
        x = self.encoder(x, attn_mask=attn_mask, time_step=time_step).flatten(start_dim=1)
        z = self.head(x)
        return z

    def forward_feature(self, x, attn_mask=None, time_step=None):
        return self.encoder(x, attn_mask=attn_mask, time_step=time_step).flatten(start_dim=1)

    def _shared_step(self, batch, stage=None):
        clips, masks, time_steps = batch
        x0, x1 = clips
        m0, m1 = masks
        t0, t1 = time_steps
        z0 = self.forward_feature(x0, attn_mask=m0, time_step=t0)
        z1 = self.forward_feature(x1, attn_mask=m1, time_step=t1)

        if self.use_ltn:
            delta_t = torch.abs(t1[:, 0] - t0[:, 0]).unsqueeze(-1)
            z1 = self.forward_ltn(z1, delta_t)

        z0 = self.head(z0)
        z1 = self.head(z1)
        loss, sim_argsort = self.criterion(z0, z1)

        # v2: use orthogonal loss
        # if self.use_ltn:
        #     loss += self.ortho_loss(self.Dt)

        self.log(stage + ".loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log(stage + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(stage + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(stage + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        # self._gpu_util()
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        if self.optimizer_type == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = build_scheduler(
            optimizer, self.lr_scheduler_type, max_epochs=self.epochs, warmup_epochs=self.warmup_epochs
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


class SimCLRMaskLM(pl.LightningModule):
    """SimCLR with MIM (masked image modeling) training for temporal encoder.

    Given a sequence of frame-level tokens output by our spatial encoder, we
    randomly mask out a subset of tokens and let the temporal encoder to
    reconstruct them in the feature space. See Section 3.5 for more details.
    """

    def __init__(self, encoder, cfg):
        super(SimCLRMaskLM, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.lr = float(cfg.solver.lr) * float(cfg.data.train_batch_size) / 256
        self.epochs = cfg.trainer.epochs
        self.weight_decay = float(cfg.solver.weight_decay)
        self.optimizer_type = cfg.solver.optimizer
        self.lr_scheduler_type = cfg.solver.lr_scheduler
        self.warmup_epochs = cfg.solver.warmup_epochs
        self.mlm_w = cfg.mlm_w
        self.mask_mode = cfg.mask_mode

        # setup contrastive
        self.encoder = encoder
        self.temporal_dim = encoder.temporal_dim
        self.simclr_head = SimCLRProjectionHead(input_dim=self.temporal_dim, batch_norm=False)
        self.criterion_contrastive = InfoNCELoss(temperature=cfg.simclr.temperature)

        # setup MLM
        self.mask_ratio = cfg.mlm.mask_ratio
        self.interchangeable = cfg.interchangeable
        print(f"[*] Using mask ratio: {self.mask_ratio}")
        print(f"[*] Interchangeable MLM training: {self.interchangeable}")
        self.norm = nn.LayerNorm(self.temporal_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.temporal_dim))

        self.token_prediction_head = SimCLRProjectionHead(
            input_dim=self.temporal_dim, batch_norm=False, output_dim=self.temporal_dim
        )
        self.criterion_mlm = NegativeCosineSimilarity()

    def log_sample_similarities(self, x, attn_mask, stage):
        inter_sim = self.criterion_mlm(x[1:, 0:1], x[:-1, 0:1])
        intra_sim = self.criterion_mlm(x[1][attn_mask[1]][1:], x[1][attn_mask[1]][:-1])
        self.log(
            stage + "_inter_sample_cosine_similarity",
            inter_sim,
            on_step=True,
            on_epoch=False,
            logger=True,
        )
        self.log(
            stage + "_intra_sample_cosine_similarity",
            intra_sim,
            on_step=True,
            on_epoch=False,
            logger=True,
        )

    def generate_token_mask(self, attn_mask, mask_ratio, mask_value=token_mapping["mask"]):
        """Helper function to generate token masks only at the location where the attn_mask is 1."""
        attn_mask = attn_mask.int()
        ones_positions = attn_mask == 1
        random_masks = torch.rand(attn_mask.shape, device=attn_mask.device)
        random_masks = ones_positions * (random_masks < mask_ratio)
        token_mask = attn_mask + ones_positions * random_masks * (mask_value - attn_mask)
        return token_mask

    def randomly_mask_tokens(self, x, attn_mask, mask_ratio):
        token_mask = self.generate_token_mask(attn_mask, mask_ratio)
        x[token_mask == token_mapping["mask"]] = self.mask_token
        return x, token_mask

    def mlm_reconstruct(self, x_masked, time_step, attn_mask):
        # reconstruct the sequence
        x_recon = self.encoder.forward_temporal_encoder(x_masked, attn_mask=attn_mask, time_step=time_step)[:, 1:, :]
        x_recon = self.norm(x_recon)
        x_recon = self.token_prediction_head(x_recon)
        return x_recon

    def simclr_no_mask(self, batch, w, stage):
        """v1: mask after the simclr loss, meaning that simclr sees all tokens"""
        clips, attn_masks, time_steps = batch
        x0, x1 = clips
        m0, m1 = attn_masks
        t0, t1 = time_steps

        x0_spatial, x0 = self.encoder(x0, m0, t0, feat_op="cls", return_spatial=True)
        x1_spatial, x1 = self.encoder(x1, m1, t1, feat_op="cls", return_spatial=True)
        z0 = self.simclr_head(x0)
        z1 = self.simclr_head(x1)
        loss_contrastive, sim_argsort = self.criterion_contrastive(z0, z1)

        self.log_sample_similarities(x0_spatial, m0, stage)

        if w > 0:
            x0_masked = x0_spatial.clone().detach()
            x0_masked, token_mask0 = self.randomly_mask_tokens(x0_masked, m0, self.mask_ratio)
            x0_recon = self.mlm_reconstruct(x0_masked, t0, attn_mask=m0)

            x1_masked = x1_spatial.clone().detach()
            x1_masked, token_mask1 = self.randomly_mask_tokens(x1_masked, m1, self.mask_ratio)
            x1_recon = self.mlm_reconstruct(x1_masked, t1, attn_mask=m1)

            l_00 = self.criterion_mlm(
                x0_recon[token_mask0 == token_mapping["mask"]], x0_spatial[token_mask0 == token_mapping["mask"]]
            )
            l_11 = self.criterion_mlm(
                x1_recon[token_mask1 == token_mapping["mask"]], x1_spatial[token_mask1 == token_mapping["mask"]]
            )
            loss_mlm = (l_00 + l_11) * 0.5

            if not loss_mlm.isnan().any():
                loss = (1 - w) * loss_contrastive + w * loss_mlm
            else:
                loss = (1 - w) * loss_contrastive
        else:
            loss_mlm = None
            loss = loss_contrastive

        return loss, loss_contrastive, loss_mlm, sim_argsort

    def simclr_with_mask(self, batch, w, stage):
        """v2: mask before the simclr loss, meaning that simclr only sees the valid tokens"""
        clips, attn_masks, time_steps = batch
        x0, x1 = clips
        m0, m1 = attn_masks
        t0, t1 = time_steps

        # extract spatial features
        x0_spatial = self.encoder.forward_spatio_encoder(x0)
        x1_spatial = self.encoder.forward_spatio_encoder(x1)

        # randomly mask some tokens
        x0_spatial, token_mask0 = self.randomly_mask_tokens(x0_spatial, m0, self.mask_ratio)
        x1_spatial, token_mask1 = self.randomly_mask_tokens(x1_spatial, m1, self.mask_ratio)

        # simclr loss
        # need to ignore the masked tokens
        attn_mask0 = torch.where(token_mask0 != 1, torch.tensor(0), token_mask0)
        attn_mask1 = torch.where(token_mask1 != 1, torch.tensor(0), token_mask1)
        x0 = self.encoder.forward_temporal_encoder(x0_spatial, attn_mask=attn_mask0, time_step=t0)[:, 0, :]
        x1 = self.encoder.forward_temporal_encoder(x1_spatial, attn_mask=attn_mask1, time_step=t1)[:, 0, :]
        z0 = self.simclr_head(x0)
        z1 = self.simclr_head(x1)
        loss_contrastive, sim_argsort = self.criterion_contrastive(z0, z1)

        self.log_sample_similarities(x0_spatial, m0, stage)

        if w > 0:
            x0_masked = x0_spatial.clone().detach()
            x0_recon = self.mlm_reconstruct(x0_masked, t0, attn_mask=m0)

            x1_masked = x1_spatial.clone().detach()
            x1_recon = self.mlm_reconstruct(x1_masked, t1, attn_mask=m1)

            l_00 = self.criterion_mlm(
                x0_recon[token_mask0 == token_mapping["mask"]], x0_spatial[token_mask0 == token_mapping["mask"]]
            )
            l_11 = self.criterion_mlm(
                x1_recon[token_mask1 == token_mapping["mask"]], x1_spatial[token_mask1 == token_mapping["mask"]]
            )
            loss_mlm = (l_00 + l_11) * 0.5

            if not loss_mlm.isnan().any():
                loss = (1 - w) * loss_contrastive + w * loss_mlm
            else:
                loss = (1 - w) * loss_contrastive
        else:
            loss_mlm = None
            loss = loss_contrastive

        return loss, loss_contrastive, loss_mlm, sim_argsort

    def _shared_step(self, batch, stage=None):
        # w = cosine_schedule(self.current_epoch, self.epochs, 0, 0.5)
        w = self.mlm_w if self.current_epoch > 100 else 0.0
        if self.mask_mode == "v1":
            loss, loss_contrastive, loss_mlm, sim_argsort = self.simclr_no_mask(batch, w, stage)
        else:
            loss, loss_contrastive, loss_mlm, sim_argsort = self.simclr_with_mask(batch, w, stage)

        self.log(stage + ".loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log(stage + ".contrastive_loss", loss_contrastive, on_step=False, on_epoch=True, logger=True)
        self.log(stage + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(stage + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(stage + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        if loss_mlm is not None:
            self.log(stage + ".mlm_loss", loss_mlm, on_step=True, on_epoch=False, logger=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = build_scheduler(
            optimizer, self.lr_scheduler_type, max_epochs=self.epochs, warmup_epochs=self.warmup_epochs
        )
        return [optimizer], [lr_scheduler]


class SimCLRCausalLM(pl.LightningModule):
    """SimCLR with autoregressive training for temporal encoder. Not used in current paper."""

    def __init__(self, encoder, cfg):
        super(SimCLRCausalLM, self).__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        self.lr = float(cfg.solver.lr) * float(cfg.data.train_batch_size) / 256
        self.epochs = cfg.trainer.epochs
        self.weight_decay = float(cfg.solver.weight_decay)
        self.optimizer_type = cfg.solver.optimizer
        self.lr_scheduler_type = cfg.solver.lr_scheduler
        self.warmup_epochs = cfg.solver.warmup_epochs

        # setup contrastive
        self.encoder = encoder
        self.temporal_dim = encoder.temporal_dim
        self.simclr_head = SimCLRProjectionHead(input_dim=self.temporal_dim)
        self.criterion_contrastive = NTXentLoss(temperature=cfg.simclr.temperature)

        # setup CLM
        use_token_prediction_head = cfg.use_token_prediction_head
        self.interchangeable = cfg.interchangeable
        print(f"[*] Interchangeable CausalLM training: {self.interchangeable}")
        if use_token_prediction_head:
            self.token_prediction_head = nn.Sequential(
                nn.Linear(self.temporal_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.temporal_dim),
            )
        else:
            self.token_prediction_head = nn.Identity()
        self.norm = nn.LayerNorm(self.temporal_dim)
        self.criterion_clm = NegativeCosineSimilarity()

    def next_token_predict(self, x, time_step, attn_mask):
        x = self.encoder.prepare_temporal_token(x, time_step)
        x = self.encoder.temporal_encoder(x, attn_mask=attn_mask)
        x = self.norm(x)
        x = self.token_prediction_head(x)
        return x

    def _shared_step(self, batch, stage=None):
        # w = cosine_schedule(self.current_epoch, self.epochs, 0, 0.9)
        # w = 0.5
        w = 0.5 if self.current_epoch > 100 else 0.0

        clips, masks, time_steps = batch
        x0, x1 = clips
        m0, m1 = masks
        t0, t1 = time_steps

        # x_spatial is after mid_project
        x0_spatial, x0 = self.encoder(x0, m0, t0, feat_op="cls", return_spatial=True)
        x1_spatial, x1 = self.encoder(x1, m1, t1, feat_op="cls", return_spatial=True)
        z0 = self.simclr_head(x0)
        z1 = self.simclr_head(x1)
        loss_contrastive = self.criterion_contrastive(z0, z1)

        # causal masks, same for both x0 and x1
        # truncate with attention mask
        if w > 0:
            b, t, d = x0_spatial.shape
            causal_mask_ = torch.tril(torch.ones(b, t, t, device=x0_spatial.device))
            m0 = m0.unsqueeze(1).expand(-1, t, -1)
            m1 = m1.unsqueeze(1).expand(-1, t, -1)
            causal_mask0 = causal_mask_ * m0
            causal_mask1 = causal_mask_ * m1

            pred0 = self.next_token_predict(x0_spatial, t0, attn_mask=causal_mask0)
            pred1 = self.next_token_predict(x1_spatial, t1, attn_mask=causal_mask1)
            pred0 = pred0[:, :-1, :]
            pred1 = pred1[:, :-1, :]
            target0 = x0_spatial[:, 1:, :]
            target1 = x1_spatial[:, 1:, :]

            if self.interchangeable:
                l_01 = self.criterion_clm(pred0, target1)
                l_10 = self.criterion_clm(pred1, target0)
                loss_clm = (l_01 + l_10) * 0.5
            else:
                l_00 = self.criterion_clm(pred0, target0)
                l_11 = self.criterion_clm(pred1, target1)
                loss_clm = (l_00 + l_11) * 0.5

            loss = (1 - w) * loss_contrastive + w * loss_clm

        else:
            loss_clm = None
            loss = loss_contrastive

        self.log(stage + ".loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log(stage + ".contrastive_loss", loss_contrastive, on_step=False, on_epoch=True, logger=True)
        if loss_clm is not None:
            self.log(stage + ".clm_loss", loss_clm, on_step=False, on_epoch=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = build_scheduler(
            optimizer, self.lr_scheduler_type, max_epochs=self.epochs, warmup_epochs=self.warmup_epochs
        )
        return [optimizer], [lr_scheduler]
