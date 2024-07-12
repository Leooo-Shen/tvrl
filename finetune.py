"""Supervised finetuning"""

import os
import re
import wandb
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from datamodule.cardiac_datamodule import CardiacDataModule
from datamodule.oct_datamodule import OCTDataModule
from datamodule.tasks import oct_task_list, cardiac_task_is_regression
from models.evaluator import Evaluator
from models.videovit import build_vit
from models.spatiotemporal import SpatioTemporalEncoder
from models.utils import reset_parameters_to_random


import hydra
from omegaconf import DictConfig
import warnings

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
os.environ["WANDB__SERVICE_WAIT"] = "300"


def find_pretrain_method(string):
    pattern = re.compile(r"2d/(.*?)/vit_")
    matches = pattern.findall(string)
    if matches:
        result = matches[0]
        print(f"Pretrain method: {result}")
    else:
        print("No match found.")
        result = None
    return result


def set_exp_name(cfg):
    lr_str = str(cfg.solver.lr)
    name = cfg.model.name + "_" + cfg.model.vit.arch
    clipframe = f"{cfg.data.n_clips}c{cfg.data.clip_frames}f{cfg.data.stride}s"

    init = "random"
    pretrained_weights = cfg.get("pretrained_weights", None)
    freeze_backbone = cfg.get("freeze_backbone", False)
    if pretrained_weights is not None:
        init = "finetune" if not freeze_backbone else "linear"
        pretrain_method = find_pretrain_method(pretrained_weights)
        if pretrain_method is not None:
            init = init + "_" + pretrain_method

    data_percentage_str = str(int(cfg.data.get("data_percentage", 1.0) * 100))

    ckpt_path = os.path.join(
        *[cfg.data.name, "2d", init, name, clipframe, lr_str, cfg.data.target, data_percentage_str]
    )

    if cfg.trainer.custom_save_ckpt_path is not None:
        ckpt_path = os.path.join(ckpt_path, cfg.trainer.custom_save_ckpt_path)
    return ckpt_path


def build_callbacks(cfg):
    exp_name = set_exp_name(cfg)
    ckpt_dir = os.path.join(cfg.ckpt_root, exp_name)
    print("[*] Checkpoint directory: ", ckpt_dir)

    if cfg.data.name == "cardiac":
        is_regression = cardiac_task_is_regression[cfg.data.target]
    elif cfg.data.name == "oct":
        is_regression, _ = oct_task_list[cfg.data.target]

    if not is_regression:
        # monitor = "val.loss"
        # mode = "min"
        monitor = "val.auc"
        mode = "max"
    else:
        monitor = "val.mae"
        mode = "min"

    patience = 15
    early_stop_callback = EarlyStopping(monitor=monitor, patience=patience, mode=mode, min_delta=1e-4)

    # checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        mode=mode,
        dirpath=ckpt_dir,
        filename="best",
        save_top_k=1,
        save_last=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks = [checkpoint_callback]

    if not cfg.debug:
        callbacks.append(early_stop_callback)
        callbacks.append(lr_monitor)
    return callbacks


def build_trainer(cfg):
    if cfg.trainer.num_gpus > 1:
        strategy = "ddp"
        sync_batchnorm = True
    else:
        strategy = None
        sync_batchnorm = False

    exp_name = set_exp_name(cfg).replace("/", "_")

    if cfg.debug:
        print("[**] DEBUG MODE")
        logger = None
        log_every_n_steps = 1
        epochs = 2
        check_val_every_n_epoch = 1
        enable_progress_bar = True
    else:
        logger = WandbLogger(project="st_finetune", name=exp_name)
        log_every_n_steps = 50
        epochs = cfg.trainer.epochs
        check_val_every_n_epoch = cfg.trainer.eval_period
        enable_progress_bar = False

    callbacks = build_callbacks(cfg)
    trainer = pl.Trainer(
        devices=cfg.trainer.num_gpus,
        accelerator="auto",
        max_epochs=epochs,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        sync_batchnorm=sync_batchnorm,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
    )

    return trainer, logger


def build_model(cfg):
    encoder = SpatioTemporalEncoder(cfg)
    pretrained_weights = cfg.get("pretrained_weights", None)
    encoder_feat_op = "cls"
    if pretrained_weights is not None and os.path.exists(pretrained_weights):
        print("[*] Loading pretrained weights from: ", pretrained_weights)
        state_dict = torch.load(pretrained_weights, map_location="cpu")["state_dict"]

        prefix = "encoder."
        prefix_st = "st."
        prefix_mae = "mae.encoder."

        # common prefix for the encoder
        if any(k.startswith(prefix) for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    k = k[len(prefix) :]
                    new_state_dict[k] = v
            reset_parameters_to_random(new_state_dict, ["cls_token"])

        elif any(k.startswith(prefix_st) for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix_st):
                    k = k[len(prefix_st) :]
                    new_state_dict[k] = v
            reset_parameters_to_random(new_state_dict, ["cls_token"])

        # mae does not use convstem, thus need a different encoder
        elif any(k.startswith(prefix_mae) for k in state_dict.keys()):
            encoder = build_vit(
                frame_size=cfg.data.img_size,
                channels=cfg.data.channels,
                num_frames=cfg.data.clip_frames,
                patch_spatial=cfg.model.vit.patch_spatial,
                patch_temporal=cfg.model.vit.patch_temporal,
                vit_config=cfg.model.vit.arch,
                drop_rate=cfg.model.vit.drop_rate,
                drop_path_rate=cfg.model.vit.drop_path_rate,
                use_convstem=False,
            )
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix_mae):
                    # k = k.replace(prefix_mae, "")
                    k = k[len(prefix_mae) :]
                    new_state_dict[k] = v
            reset_parameters_to_random(new_state_dict, ["cls_token"])

        encoder.load_state_dict(new_state_dict, strict=False)

        # use a randomly init downstream cls token
        freeze_backbone = cfg.get("freeze_backbone", False)
        if freeze_backbone:
            print("[*] Freezing backbone except cls tokens.")
            for name, param in encoder.named_parameters():
                if name not in ["cls_token", "downstream_cls_token"]:
                    param.requires_grad = False

    target = cfg.data.target
    if cfg.data.name == "cardiac":
        is_regression = cardiac_task_is_regression[target]
        ensemble = True
    elif cfg.data.name == "oct":
        is_regression, _ = oct_task_list[target]
        ensemble = False
    return Evaluator(encoder, cfg, is_regression=is_regression, ensemble=ensemble, encoder_feat_op=encoder_feat_op)


def build_datamodule(cfg, cache_dict=None):
    if cfg.data.name == "cardiac":
        data_moduel = CardiacDataModule(cfg, cache_dict=cache_dict)
    elif cfg.data.name == "oct":
        data_moduel = OCTDataModule(cfg, cache_dict=cache_dict)
    else:
        raise NotImplementedError
    return data_moduel


def run(cfg, cache_dict=None):
    # seeds = cfg.get("seeds", None)
    # num_runs = cfg.get("num_runs", 2)
    # if seeds is None:
    #     seeds = np.random.randint(0, 10000, size=num_runs)
    # seeds = [1234, 2345, 3456, 4567, 5678]
    seeds = [1234, 2345]

    for data_percentage in [0.01, 1.0]:
        test_results = []
        print(f"[*] Using {data_percentage * 100}% of the data")
        cfg.data.data_percentage = data_percentage

        for seed in seeds:
            pl.seed_everything(seed)
            model = build_model(cfg)
            data_module = build_datamodule(cfg, cache_dict=cache_dict)
            trainer, logger = build_trainer(cfg)

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
            res = trainer.test(model, datamodule=data_module)

            # to manually test
            # ckpt_path = ""
            # res = trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)

            test_results.append(res[0])
            if logger:
                wandb.finish()

        # compute mean and std
        df = pd.DataFrame(test_results)
        mean = df.mean().to_dict()
        std = df.std().to_dict()

        res_dict = {}
        for key, value in mean.items():
            key = key + ".mean"
            value = round(value, 3)
            res_dict[key] = value
        for key, value in std.items():
            value = round(value, 3)
            key = key + ".std"
            res_dict[key] = value

        print("*****************************")
        print(res_dict)


@hydra.main(config_path="config", config_name="finetune")
def main(cfg: DictConfig):
    # load dataset to RAM to speed up training
    cache_dir = f""
    train_cache = os.path.join(cache_dir, "train.pt")
    val_cache = os.path.join(cache_dir, "val.pt")
    test_cache = os.path.join(cache_dir, "test.pt")

    print(f"[*] Loading cached data from {cache_dir} ...")
    train = torch.load(train_cache)
    val = torch.load(val_cache)
    test = torch.load(test_cache)
    cache_dict = {"train": train, "val": val, "test": test}
    print(f"[*] Loaded cache")

    target = cfg.data.target
    if isinstance(target, str):
        target = [target]
    for t in target:
        cfg.data.target = t
        print(f"[*] Target: {t}")

        # use larger lr for regression
        is_regression = False
        if cfg.data.name == "cardiac":
            is_regression = cardiac_task_is_regression[cfg.data.target]
        elif cfg.data.name == "oct":
            is_regression, _ = oct_task_list[cfg.data.target]

        if is_regression:
            cfg.solver.lr *= 5
        print(f"[*] Learning rate: {cfg.solver.lr}")

        run(cfg, cache_dict=cache_dict)


if __name__ == "__main__":
    main()
