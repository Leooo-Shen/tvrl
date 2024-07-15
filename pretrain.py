import os
import random
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from datamodule.cardiac_datamodule import CardiacDataModule
from datamodule.oct_datamodule import OCTDataModule
from models.mae import VideoMAE
from models.simclr import SimCLR, SimCLRCausalLM, SimCLRMaskLM
from models.spatiotemporal import SpatioTemporalEncoder

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

os.environ["WANDB__SERVICE_WAIT"] = "300"


def set_exp_name(cfg):
    lr_str = str(cfg.solver.lr)
    name = cfg.model.name + "_" + cfg.model.vit.arch
    clipframe = f"{cfg.data.n_clips}c{cfg.data.clip_frames}f{cfg.data.stride}s"

    use_ltn = cfg.get("use_ltn", False)
    ltn_str = "ltn" if use_ltn else ""
    time_str = "time" if cfg.get("use_time_embed", False) else "notime"

    interchangeable = cfg.get("interchangeable", False)
    inter_str = "inter" if interchangeable else "intra"

    m_ratio = ""
    if cfg.pretrain_framework == "mlm":
        m_ratio = "M" + str(int(cfg.mlm.mask_ratio * 100))

    data_percentage_str = str(int(cfg.data.get("data_percentage", 1.0) * 100))

    ckpt_path = os.path.join(
        *[
            cfg.data.name,
            "2d",
            cfg.pretrain_framework,
            name,
            clipframe,
            lr_str,
            ltn_str,
            inter_str,
            time_str,
            m_ratio,
            data_percentage_str,
        ]
    )

    if cfg.trainer.custom_save_ckpt_path is not None:
        ckpt_path = os.path.join(ckpt_path, cfg.trainer.custom_save_ckpt_path)
    return ckpt_path


def build_callbacks(cfg):
    exp_name = set_exp_name(cfg)
    ckpt_dir = os.path.join(cfg.ckpt_root, exp_name)
    print("[*] Checkpoint directory: ", ckpt_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="epoch",
        mode="max",
        filename="last",
        save_top_k=1,
        dirpath=ckpt_dir,
        save_last=False,
    )

    # grad_norm_callback = GradientNormCallback()
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop_callback = EarlyStopping(
        monitor="val.loss",
        mode="min",
        patience=cfg.trainer.epochs,
        check_finite=True,  # only stop when the loss is nan
    )
    callbacks = [checkpoint_callback, early_stop_callback]

    if not cfg.debug:
        callbacks.append(lr_monitor)
    return callbacks


def build_trainer(cfg):
    if cfg.trainer.num_gpus > 1:
        strategy = "ddp"
        sync_batchnorm = True
    else:
        strategy = "auto"
        sync_batchnorm = False

    exp_name = set_exp_name(cfg).replace("/", "_")

    if cfg.debug:
        print("[**] DEBUG MODE")
        logger = None
        epochs = 200
        enable_progress_bar = True
    else:
        logger = WandbLogger(project="spatiotemporal", name=exp_name)
        epochs = cfg.trainer.epochs
        enable_progress_bar = False

    callbacks = build_callbacks(cfg)
    trainer = pl.Trainer(
        devices=cfg.trainer.num_gpus,
        accelerator="gpu",
        max_epochs=epochs,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        sync_batchnorm=sync_batchnorm,
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
        check_val_every_n_epoch=5,
    )
    return trainer


def build_model(cfg):
    encoder = SpatioTemporalEncoder(cfg)
    if cfg.pretrain_framework == "simclr":
        model = SimCLR(encoder, cfg)
    elif cfg.pretrain_framework == "mae":
        model = VideoMAE(cfg)
    elif cfg.pretrain_framework == "mlm":
        model = SimCLRMaskLM(encoder, cfg)
    elif cfg.pretrain_framework == "clm":
        model = SimCLRCausalLM(encoder, cfg)
    else:
        raise NotImplementedError
    return model


def build_datamodule(cfg):
    if cfg.data.name == "cardiac":
        data_moduel = CardiacDataModule(cfg)
    elif cfg.data.name == "oct":
        data_moduel = OCTDataModule(cfg)
    else:
        raise NotImplementedError
    return data_moduel


@hydra.main(config_path="config", config_name="pretrain")
def main(cfg: DictConfig):
    # seed = cfg.get("seed", random.randint(0, 1000))
    seed = 2427633826
    pl.seed_everything(seed)

    model = build_model(cfg)
    data_module = build_datamodule(cfg)
    trainer = build_trainer(cfg)

    ckpt_path = None
    if cfg.trainer.auto_resume:
        exp_name = set_exp_name(cfg)
        ckpt_dir = os.path.join(cfg.ckpt_root, exp_name)
        ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
        if not os.path.exists(ckpt_path):
            ckpt_path = None
        print("[*] Resuming from checkpoint: ", ckpt_path)

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
