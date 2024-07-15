import ast
import os

import pandas as pd
import pytorch_lightning as pl
import torch
from oct_datasets import OCTContrastiveDataset, OCTDataset
from tasks import oct_task_list
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from transforms import oct_aug_mae, oct_aug_mild, oct_aug_strong


class OCTDataModule(pl.LightningDataModule):
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
        self.relative_time_embed = cfg.relative_time_embed
        self.collate_fn = None  # default collate_fn as None
        if cfg.debug:
            self.train_batch_size = 8
            self.test_batch_size = 2

        target = cfg.data.get("target", None)
        balance_classes = cfg.data.get("balance_classes", False)
        train_path = "train.csv"
        if target:
            is_regression, _ = oct_task_list[target]
            if not is_regression and balance_classes:
                train_path = f"train_{target}.csv"
        print(f"train_path: {train_path}")

        csv_dir = cfg.data.csv_dir
        data_percentage = cfg.data.get("data_percentage", 1.0)

        train_data = pd.read_csv(os.path.join(csv_dir, train_path))
        val_data = pd.read_csv(os.path.join(csv_dir, "val.csv"))
        test_data = pd.read_csv(os.path.join(csv_dir, "test.csv"))
        self.train_data = self.df_preprocess(train_data, target)
        self.val_data = self.df_preprocess(val_data, target)
        self.test_data = self.df_preprocess(test_data, target)

        # use a subset of the training data
        self.train_data = self.train_data[: int(len(self.train_data) * data_percentage)]
        print(f"[*] Using {data_percentage*100}% data for training")

        self.cache_dict = cache_dict

    def df_preprocess(self, df, target=None):
        df["Scan_history"] = df["Scan_history"].apply(lambda x: ast.literal_eval(x))
        df["Scan_lenght"] = df["Scan_history"].apply(len)
        df["Video"] = df["Video"].apply(lambda x: ast.literal_eval(x))
        if target:
            _, filter_condition = oct_task_list[target]
            df = df[df.apply(filter_condition, axis=1)].reset_index() if filter_condition else df

        # # some preprocess to run test on specific longtiudinal sequences
        # df = df[df["Scan_lenght"] > 8]
        # df = df[df["Converts_to_cRORA of 1000 um_within_3_years"] == 1]

        return df

    def setup(self, stage=None):
        # run on every GPU
        print("[*] Setting up DataModule ...")
        if self.pretrain_framework in [
            "simclr",
            "mlm",
            "clm",
        ]:
            self.setup_contrastive(stage)
        elif self.pretrain_framework == "mae":
            self.setup_mae(stage)
        else:
            self.setup_finetune(stage)

    def setup_contrastive(self, stage=None):
        augment = oct_aug_strong(self.img_size)
        img_dir = self.cfg.data.img_dir
        if stage == "fit" or stage is None:
            self.train_set = self.construct_contrastive_dataset_with_cfg(
                self.train_data, img_dir, self.cfg, transforms=augment, stage="train"
            )
            self.val_set = self.construct_contrastive_dataset_with_cfg(
                self.val_data, img_dir, self.cfg, transforms=augment, stage="val"
            )

    def setup_finetune(self, stage=None):
        self.collate_fn = OCTDataset.collate_fn
        basic_t = v2.Compose(
            [
                v2.CenterCrop(size=self.img_size),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )
        augment = oct_aug_mild(self.img_size)
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
            self.test_set = self.construct_dataset_with_cfg(
                self.test_data, test_img_dir, self.cfg, transforms=basic_t, stage="test"
            )

    def setup_mae(self, stage=None):
        self.collate_fn = OCTDataset.group_collate_fn
        augment = oct_aug_mae(self.img_size, horizontal_flip=True)
        img_dir = self.cfg.data.img_dir
        if stage == "fit" or stage is None:
            self.train_set = self.construct_dataset_with_cfg(
                self.train_data, img_dir, self.cfg, transforms=augment, pad_clip=False
            )
            self.val_set = self.construct_dataset_with_cfg(
                self.val_data, img_dir, self.cfg, transforms=augment, pad_clip=False
            )

    def construct_dataset_with_cfg(self, df, img_dir, cfg, transforms, stage=None, pad_clip=True):
        target = cfg.data.get("target", None)
        cache = self.cache_dict[stage] if self.cache_dict is not None else None
        return OCTDataset(
            df,
            img_dir=img_dir,
            clip_frames=cfg.data.clip_frames,
            stride=cfg.data.stride,
            target=target,
            transforms=transforms,
            stage=stage,
            pad_clip=pad_clip,
            cache=cache,
            relative_time_embed=self.relative_time_embed,
        )

    def construct_contrastive_dataset_with_cfg(self, df, img_dir, cfg, transforms, stage=None):
        cache = self.cache_dict[stage] if self.cache_dict is not None else None
        return OCTContrastiveDataset(
            df,
            img_dir=img_dir,
            clip_frames=cfg.data.clip_frames,
            stride=cfg.data.stride,
            n_clips=cfg.data.n_clips,
            transforms=transforms,
            stage=stage,
            cache=cache,
            relative_time_embed=self.relative_time_embed,
        )

    def train_dataloader(self):
        loader = DataLoader(
            self.train_set,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,
            drop_last=True,
            collate_fn=self.collate_fn,
        )

        return loader

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.test_batch_size,
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
