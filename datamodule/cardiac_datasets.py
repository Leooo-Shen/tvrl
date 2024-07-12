import os
import torch
import numpy as np

from sampler import ClipSamplingRandom, ClipSamplingWithCDF, ClipSamplingDense
from torch.utils.data import Dataset
from tasks import cardiac_task_is_regression
from datamodule.transforms import add_gaussian_noise


class CardiacDataset(Dataset):
    """
    Return the a list of clips and the label for supervised learning and linear evaluation.
    Each clip is of shape [1, frames, h, w]
    """

    def __init__(
        self,
        dataframe,
        img_dir,
        clip_frames,
        stride,
        n_clips=1,
        transforms=None,
        dense_sampling=False,
        target=None,
        stage=None,
        cache=None,
        relative_time_embed=False,
        return_original_video=False,
    ):
        self.img_dir = img_dir
        self.target = target
        is_regression = False
        if target:
            is_regression = cardiac_task_is_regression[target]
            if target.lower() == "lvef":
                self.target = "LVEF (%)"
            if target.lower() == "lvsv":
                self.target = "LVSV (mL)"
            if target.lower() == "lvedv":
                self.target = "LVEDV (mL)"
            if target.lower() == "lvesv":
                self.target = "LVESV (mL)"
            if target.lower() == "cindex":
                self.target = "Cardiac index-2.0"
            if target.lower() == "lvm":
                self.target = "LVM (g)"
            if target.lower() == "co":
                self.target = "Cardiac output during PWA-2.0"
            dataframe = dataframe.dropna(subset=[self.target])

        self.dataframe = dataframe
        self.target_dtype = torch.float if is_regression else torch.long
        self.transforms = transforms
        self.relative_time_embed = relative_time_embed

        # dense sample the video during testing
        if dense_sampling:
            self.clip_sampler = ClipSamplingDense(
                window_size=clip_frames * stride,
                window_step_size=1,
                data_sample_stride=stride,
            )
        else:
            self.clip_sampler = ClipSamplingRandom(
                clip_frames=clip_frames,
                n_clips=n_clips,
                stride=stride,
                make_clips_identical=False,
            )
        self.return_original_video = return_original_video
        self.cache = cache

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        """Support dense sampling"""
        folder = self.dataframe.iloc[index]["eid"]
        if self.cache:
            data = self.cache[folder]
        else:
            data_path = os.path.join(self.img_dir, f"{folder}.npy")
            data = np.load(data_path).astype(np.uint8)

        # sample a list of n clips, each in shape [f, h, w]
        clips, indices = self.clip_sampler(data)
        clips_transformed = []
        time_steps = []
        masks = []
        for c, idx in zip(clips, indices):
            c = torch.tensor(c).unsqueeze(1)
            if self.transforms:
                c = self.transforms(c).transpose(0, 1)
            clips_transformed.append(c)
            time_step = torch.tensor(idx, dtype=torch.float)
            if self.relative_time_embed:
                time_step = time_step - time_step[0]
            time_steps.append(time_step)
            masks.append(torch.ones_like(time_step, dtype=torch.bool))

        label = None
        if self.target:
            label = self.dataframe.iloc[index][self.target]
            label = torch.tensor(label, dtype=self.target_dtype)

        if self.return_original_video:
            return clips_transformed, label, masks, time_steps, torch.tensor(data, dtype=torch.uint8)

        return clips_transformed, label, masks, time_steps

    @staticmethod
    def collate_fn(batch):
        """
        Collate function with clip padding.
        Collate a batch of data. Batch_size is variable, bs = cfg.batch * n_clips_sampled.
        Returns:
            clips: [B, n_clips, C, T, H, W]
            labels: [B, 1]
        """
        clips, labels, masks, time_steps = zip(*batch)
        clips = [torch.stack(c, dim=0) for c in clips]
        clips = torch.stack(clips, dim=0).squeeze(1)
        time_steps = [torch.stack(t, dim=0) for t in time_steps]
        time_steps = torch.stack(time_steps, dim=0).squeeze(1)
        masks = [torch.stack(m, dim=0) for m in masks]
        masks = torch.stack(masks, dim=0).squeeze(1)
        labels = None if labels[0] is None else torch.stack(labels, dim=0)
        return {
            "clip": clips,
            "label": labels,
            "pad_mask": masks,
            "time_step": time_steps,
        }


class CardiacContrastiveDataset(Dataset):
    """
    Return the a list of clips for contrastive learning.
    Each clip is of shape [1, frames, h, w]
    """

    def __init__(
        self,
        dataframe,
        img_dir,
        clip_frames,
        stride,
        n_clips,
        sampling_cdf=None,
        make_clips_identical=False,
        transforms=None,
        stage=None,
        cache=None,
        strict_half=False,
        relative_time_embed=False,
    ):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transforms = transforms
        self.clip_frames = clip_frames
        self.stride = stride
        self.n_clips = n_clips
        self.clip_sampler = self._build_clip_sampler(sampling_cdf, clip_frames, stride, n_clips, make_clips_identical)
        self.strict_half = strict_half
        self.relative_time_embed = relative_time_embed

        self.cache = cache
        print(f"[*] {len(self.dataframe)} videos")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        folder = self.dataframe.iloc[index]["eid"]
        if self.cache:
            data = self.cache[folder]
        else:
            data_path = os.path.join(self.img_dir, f"{folder}.npy")
            data = np.load(data_path).astype(np.uint8)

        # take the first half or second half of the video before sampling
        if self.strict_half:
            data = data[:25] if np.random.rand() > 0.5 else data[25:]

        # sample a list of clips, each in shape [f, h, w]
        clips, indices = self.clip_sampler(data)
        if len(clips) == 1:
            clips *= 2
            indices *= 2

        clips_transformed = []
        time_steps = []
        for c, t in zip(clips, indices):
            # transformation need frame as batch
            c = torch.tensor(c).unsqueeze(1)
            # reshape to [1, f, h, w]
            c = self.transforms(c).transpose(0, 1)
            clips_transformed.append(c)

            if self.relative_time_embed:
                t = t - t[0]

            t = add_gaussian_noise(t)
            time_steps.append(torch.tensor(t, dtype=torch.long))

        masks = [torch.ones_like(t, dtype=torch.bool) for t in time_steps]
        return clips_transformed, masks, time_steps

    def _build_clip_sampler(self, sampling_cdf, clip_frames, stride, n_clips, make_clips_identical):
        if sampling_cdf in [
            "linear_increase",
            "quadratic_increase",
            "linear_decrease",
            "quadratic_decrease",
            "root_decrease",
            "constant",
        ]:
            return ClipSamplingWithCDF(
                clip_frames=clip_frames,
                cdf_type=sampling_cdf,
                stride=stride,
            )
        return ClipSamplingRandom(
            clip_frames=clip_frames,
            n_clips=n_clips,
            stride=stride,
            make_clips_identical=make_clips_identical,
        )
