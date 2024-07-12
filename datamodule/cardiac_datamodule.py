import os
import torch
import pandas as pd
import pytorch_lightning as pl
from torchvision.transforms import v2

from transforms import cardiac_aug_mild, cardiac_aug_strong, cardiac_aug_mae
from torch.utils.data import DataLoader
from cardiac_datasets import (
    CardiacDataset,
    CardiacContrastiveDataset,
)
from tasks import cardiac_task_is_regression


class CardiacDataModule(pl.LightningDataModule):
    def __init__(self, cfg, cache_dict=None):
        super().__init__()
        self.cfg = cfg
        self.num_workers = cfg.loader.num_workers
        self.pin_memory = cfg.loader.pin_memory
        self.persistent_workers = cfg.loader.persistent_workers
        self.img_size = cfg.data.img_size
        self.pretrain_framework = cfg.get("pretrain_framework", None)
        self.train_batch_size = cfg.data.train_batch_size
        self.test_batch_size = cfg.data.test_batch_size
        self.collate_fn = None  # default collate_fn as None
        self.relative_time_embed = cfg.relative_time_embed
        if cfg.debug:
            self.train_batch_size = 8
            self.test_batch_size = 2
        target = cfg.data.get("target", None)
        csv_dir = cfg.data.csv_dir
        data_percentage = cfg.data.get("data_percentage", 1.0)
        train_path = "train.csv"

        # balanced dataset for supervised training and linear evaluation
        if target is not None:
            is_regression = cardiac_task_is_regression[target]
            if not is_regression:
                train_path = f"train_{target}.csv"
        self.train_data = pd.read_csv(os.path.join(csv_dir, train_path), dtype={"eid": str})
        self.val_data = pd.read_csv(os.path.join(csv_dir, "val.csv"), dtype={"eid": str})
        self.test_data = pd.read_csv(os.path.join(csv_dir, "test.csv"), dtype={"eid": str})

        # use a subset of the training data
        self.train_data = self.train_data[: int(len(self.train_data) * data_percentage)]
        print(f"[*] Using {data_percentage*100}% of {train_path} for training")

        self.cache_dict = cache_dict

    def setup(self, stage=None):
        # run on every GPU
        print("[*] Setting up DataModule ...")
        if self.pretrain_framework in ["simclr"]:
            self.setup_contrastive(stage, strict_half=False)
        elif self.pretrain_framework in ["mlm", "clm", "mlm_momentum", "clm_momentum", "simclr_mae", "simsiam_mlm"]:
            self.setup_contrastive(stage, strict_half=True)
        elif self.pretrain_framework == "mae":
            self.setup_mae(stage)
        else:
            self.setup_finetune(stage)

    def setup_contrastive(self, stage=None, strict_half=False):
        augment = cardiac_aug_strong(self.img_size)
        img_dir = self.cfg.data.img_dir
        if stage == "fit" or stage is None:
            self.train_set = self.construct_contrastive_dataset_with_cfg(
                self.train_data, img_dir, self.cfg, transforms=augment, stage="train", strict_half=strict_half
            )
            self.val_set = self.construct_contrastive_dataset_with_cfg(
                self.val_data, img_dir, self.cfg, transforms=augment, stage="val", strict_half=strict_half
            )

    def setup_finetune(self, stage=None):
        self.collate_fn = CardiacDataset.collate_fn
        augment = cardiac_aug_mild(self.img_size)
        basic_t = v2.Compose([v2.Resize(self.img_size), v2.ToDtype(torch.float32, scale=True)])
        img_dir = self.cfg.data.img_dir
        test_img_dir = img_dir

        if stage == "fit" or stage is None:
            self.train_set = self.construct_dataset_with_cfg(
                self.train_data, img_dir, self.cfg, transforms=augment, stage="train"
            )
            self.val_set = self.construct_dataset_with_cfg(
                self.val_data, test_img_dir, self.cfg, transforms=basic_t, stage="val"
            )

            # in case the train_batch_size is larger than the number of training samples
            if len(self.train_set) < self.train_batch_size:
                self.train_batch_size = len(self.train_set)
                print(f"[*] Setting train_batch_size to {self.train_batch_size}")

        elif stage == "test":
            # use dense sampling for testing
            self.test_set = self.construct_dataset_with_cfg(
                self.test_data, test_img_dir, self.cfg, transforms=basic_t, dense_sampling=True, stage="test"
            )

    def setup_mae(self, stage=None):
        self.collate_fn = CardiacDataset.collate_fn
        augment = cardiac_aug_mae(self.img_size, horizontal_flip=True)
        img_dir = self.cfg.data.img_dir
        if stage == "fit" or stage is None:
            self.train_set = self.construct_dataset_with_cfg(
                self.train_data, img_dir, self.cfg, transforms=augment, stage="train"
            )
            self.val_set = self.construct_dataset_with_cfg(
                self.val_data, img_dir, self.cfg, transforms=augment, stage="val"
            )

    def construct_dataset_with_cfg(self, df, img_dir, cfg, transforms, dense_sampling=False, stage=None):
        target = cfg.data.get("target", None)
        cache = self.cache_dict[stage] if self.cache_dict is not None else None
        return CardiacDataset(
            df,
            img_dir=img_dir,
            target=target,
            clip_frames=cfg.data.clip_frames,
            stride=cfg.data.stride,
            n_clips=cfg.data.n_clips,
            transforms=transforms,
            dense_sampling=dense_sampling,
            stage=stage,
            cache=cache,
            relative_time_embed=self.relative_time_embed,
        )

    def construct_contrastive_dataset_with_cfg(self, df, img_dir, cfg, transforms, stage=None, strict_half=False):
        sampling_cdf = cfg.data.get("sampling_cdf", None)
        make_clips_identical = cfg.data.get("make_clips_identical", False)
        cache = self.cache_dict[stage] if self.cache_dict is not None else None
        return CardiacContrastiveDataset(
            df,
            img_dir=img_dir,
            clip_frames=cfg.data.clip_frames,
            stride=cfg.data.stride,
            n_clips=cfg.data.n_clips,
            sampling_cdf=sampling_cdf,
            make_clips_identical=make_clips_identical,
            transforms=transforms,
            stage=stage,
            cache=cache,
            strict_half=strict_half,
            relative_time_embed=self.relative_time_embed,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            drop_last=True,
            collate_fn=self.collate_fn,
        )
