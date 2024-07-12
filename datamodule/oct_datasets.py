import os
import imageio
import numbers
import torch
import pandas as pd
import numpy as np
from torchvision.transforms import v2

from torch.utils.data import Dataset
from tasks import oct_task_list
from sampler import PathSamplingRandom
from datamodule.transforms import PadVideo, add_gaussian_noise


def sample_scan_history(scan_history, length, stride=1):
    """sample n previous frames from the end (including the last one)"""
    sample_length = min(length * stride, len(scan_history))
    sampled = scan_history[-sample_length::stride]
    paths, unix_years = zip(*sampled)
    unix_years = np.array(unix_years)
    unix_years -= scan_history[0][1]
    return paths, unix_years


def random_sample_path(video_paths, sampler):
    """Random sample n consecutive frames from the paths"""
    sampled, idx = sampler(video_paths)
    assert len(sampled) == 1
    sampled = sampled[0]
    paths, unix_years = zip(*sampled)
    unix_years = np.array(unix_years)
    unix_years -= video_paths[0][1]
    return paths, unix_years


def load_images_into_video(image_paths, preprocess=None):
    """Load images into a video in shape [c, t, h, w]"""
    if preprocess:
        images = [preprocess(torch.tensor(imageio.imread(image_path)).unsqueeze(0)) for image_path in image_paths]
    else:
        images = [torch.tensor(imageio.imread(image_path)).unsqueeze(0) for image_path in image_paths]
    return torch.stack(images, dim=1)


def load_images_into_video_with_cache(image_paths, cache, preprocess=None):
    """Load images into a video in shape [c, t, h, w]"""
    if preprocess:
        images = [preprocess(cache[image_path]) for image_path in image_paths]
    else:
        images = [cache[image_path] for image_path in image_paths]
    return torch.stack(images, dim=1)


class OCTDataset(Dataset):
    """OCT dataset for finetuning"""

    def __init__(
        self,
        dataframe,
        img_dir,
        clip_frames,
        stride=1,
        transforms=None,
        pad_clip=True,
        target=None,
        stage=None,
        cache=None,
        relative_time_embed=False,
    ):
        self.img_dir = img_dir
        self.target = target
        self.clip_frames = clip_frames
        self.stride = stride
        self.target_dtype = torch.long
        if target:
            is_regression, _ = oct_task_list[target]
            self.target_dtype = torch.float if is_regression else torch.long
        self.dataframe = dataframe
        self.preprocess = v2.Resize(128, antialias=True)
        self.transforms = transforms
        self.pad_clip = pad_clip
        self.clip_padding = PadVideo(clip_frames)
        self.relative_time_embed = relative_time_embed
        self.cache = cache

        print(f"[*] {len(self.dataframe)} OCT data points")

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        paths, time_step = sample_scan_history(row["Scan_history"], self.clip_frames, stride=self.stride)

        # relative time step
        if self.relative_time_embed:
            time_step = time_step - time_step[0]

        # get the clips and masks
        if self.cache:
            clip = load_images_into_video_with_cache(paths, cache=self.cache, preprocess=self.preprocess)
        else:
            image_paths = [os.path.join(self.img_dir, img) for img in paths]
            clip = load_images_into_video(image_paths, preprocess=self.preprocess)

        # transformation need frame as batch
        if self.transforms:
            clip = self.transforms(clip.transpose(0, 1)).transpose(0, 1)

        # pad zero to clip
        if self.pad_clip:
            clip, mask = self.clip_padding(clip)
        else:
            mask = torch.ones(clip.shape[1], dtype=torch.bool)

        label = None
        if self.target:
            label = row[self.target]
            label = torch.tensor(label, dtype=self.target_dtype)

        # no gaussian noise for time step during finetuning
        # time_step = add_gaussian_noise(time_step)
        # pad zero to time_step
        if len(time_step) < self.clip_frames:
            time_step = np.pad(time_step, (0, self.clip_frames - len(time_step)), mode="constant", constant_values=1e5)
        time_step = torch.tensor(time_step, dtype=torch.float)
        return clip, label, mask, time_step

    @staticmethod
    def collate_fn(batch):
        """
        Collate function with clip padding.
        Collate a batch of data. Batch_size is variable, bs = cfg.batch * n_clips_sampled.
        Returns:
            clips: [B, C, T, H, W]
            labels: [B, 1]
        """
        clips, labels, masks, time_steps = zip(*batch)
        clips = torch.stack(clips, dim=0)
        labels = None if labels[0] is None else torch.stack(labels, dim=0)
        masks = torch.stack(masks, dim=0)
        time_steps = torch.stack(time_steps, dim=0)
        return {
            "clip": clips,
            "label": labels,
            "pad_mask": masks,
            "time_step": time_steps,
        }

    @staticmethod
    def group_collate_fn(batch):
        """
        Collate function without clip padding. Group the batch by clip length.
        Returns:
            clips: [B, C, T, H, W]
            labels: [B, 1]
        """
        clips, labels, masks, time_steps = zip(*batch)

        groupped_clips = {}
        groupped_labels = {}
        groupped_masks = {}
        groupped_time_steps = {}

        # group with the same length
        for clip, label, mask, time_step in zip(clips, labels, masks, time_steps):
            # clip in shape (1, t, 128, 128)
            t = clip.shape[1]
            if t not in groupped_clips:
                groupped_clips[t] = [clip]
                groupped_labels[t] = [label]
                groupped_masks[t] = [mask]
                groupped_time_steps[t] = [time_step]
            else:
                groupped_clips[t].append(clip)
                groupped_labels[t].append(label)
                groupped_masks[t].append(mask)
                groupped_time_steps[t].append(time_step)

        concat_clips = []
        concat_labels = []
        concat_masks = []
        concat_time_steps = []

        for t, clips in groupped_clips.items():
            concat_clips.append(torch.stack(clips, dim=0))
            concat_masks.append(torch.stack(groupped_masks[t], dim=0))
            concat_time_steps.append(torch.stack(groupped_time_steps[t], dim=0))
            if groupped_labels[t][0] is not None:
                concat_labels.append(torch.stack(groupped_labels[t], dim=0))
            else:
                concat_labels.append(None)

        return {
            "clip": concat_clips,
            "label": concat_labels,
            "pad_mask": concat_masks,
            "time_step": concat_time_steps,
        }


class OCTContrastiveDataset(Dataset):
    """ """

    def __init__(
        self,
        dataframe,
        img_dir,
        clip_frames,
        stride,
        n_clips=1,
        transforms=None,
        stage=None,
        cache=None,
        relative_time_embed=False,
    ):
        self.img_dir = img_dir
        self.clip_frames = clip_frames
        self.transforms = transforms
        self.dataframe = dataframe
        self.clip_frames = clip_frames
        self.stride = stride
        self.n_clips = n_clips

        self.clip_padding = PadVideo(clip_frames)
        self.preprocess = v2.Resize(128, antialias=True)
        self.random_sampler = PathSamplingRandom(clip_frames=clip_frames, n_clips=1, stride=stride)
        self.relative_time_embed = relative_time_embed

        # load cache, each image in shape [c, h, w]
        self.cache = cache
        print(f"[*] {len(self.dataframe)} OCT data points")

    def get_two_clips(self, index):
        row = self.dataframe.iloc[index]
        paths1, time_step1 = sample_scan_history(row["Scan_history"], self.clip_frames, stride=self.stride)
        paths2, time_step2 = random_sample_path(row["Video"], self.random_sampler)

        # relative time step
        if self.relative_time_embed:
            time_step1 = time_step1 - time_step1[0]
            time_step2 = time_step2 - time_step2[0]

        if self.cache:
            clip1 = load_images_into_video_with_cache(paths1, cache=self.cache, preprocess=self.preprocess)
            clip2 = load_images_into_video_with_cache(paths2, cache=self.cache, preprocess=self.preprocess)
        else:
            image_paths1 = [os.path.join(self.img_dir, img) for img in paths1]
            image_paths2 = [os.path.join(self.img_dir, img) for img in paths2]
            clip1 = load_images_into_video(image_paths1, preprocess=self.preprocess)
            clip2 = load_images_into_video(image_paths2, preprocess=self.preprocess)

        clip = [clip1, clip2]
        time_step = [time_step1, time_step2]
        clips_transformed = []
        masks = []
        time_steps = []
        for c, t in zip(clip, time_step):
            # transformation need frame as batch
            c = self.transforms(c.transpose(0, 1)).transpose(0, 1)
            c, mask = self.clip_padding(c)
            clips_transformed.append(c)
            masks.append(mask)

            t = add_gaussian_noise(t, std=0.1)
            if len(t) < self.clip_frames:
                t = np.pad(t, (0, self.clip_frames - len(t)), mode="constant", constant_values=1e5)
            t = torch.tensor(t, dtype=torch.float)
            time_steps.append(t)
        return clips_transformed, masks, time_steps

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        if self.n_clips == 2:
            return self.get_two_clips(index)

        row = self.dataframe.iloc[index]
        paths, time_step = sample_scan_history(row["Scan_history"], self.clip_frames, stride=self.stride)

        # relative time step
        if self.relative_time_embed:
            time_step = time_step - time_step[0]

        # get the clips and masks
        if self.cache:
            clip = load_images_into_video_with_cache(paths, cache=self.cache, preprocess=self.preprocess)
        else:
            image_paths = [os.path.join(self.img_dir, img) for img in paths]
            clip = load_images_into_video(image_paths, preprocess=self.preprocess)

        clip = [clip] * 2
        time_step = [time_step] * 2
        clips_transformed = []
        masks = []
        time_steps = []
        for c, t in zip(clip, time_step):
            # transformation need frame as batch
            c = self.transforms(c.transpose(0, 1)).transpose(0, 1)
            c, mask = self.clip_padding(c)
            clips_transformed.append(c)
            masks.append(mask)

            t = add_gaussian_noise(t, std=0.1)
            if len(t) < self.clip_frames:
                t = np.pad(t, (0, self.clip_frames - len(t)), mode="constant", constant_values=1e5)
            t = torch.tensor(t, dtype=torch.float)
            time_steps.append(t)

        return clips_transformed, masks, time_steps
