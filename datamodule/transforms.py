import torch
import numpy as np
import torchvision
from torchvision.transforms import v2


def add_gaussian_noise(data, mean=0, std=0.01):
    noise = np.random.normal(mean, std, data.shape)
    return data + noise


class PadVideo:
    def __init__(self, max_length, pad_side="right"):
        self.max_length = max_length
        assert pad_side in ["left", "right"]
        self.pad_side = pad_side

    def __call__(self, video):
        c, t, w, h = video.shape
        mask = torch.ones(self.max_length, dtype=torch.bool)
        if t < self.max_length:
            padding_tensor = torch.zeros((c, self.max_length - t, w, h), dtype=video.dtype)

            if self.pad_side == "right":
                video = torch.cat([video, padding_tensor], dim=1)
                mask[t:] = False
            elif self.pad_side == "left":
                video = torch.cat([padding_tensor, video], dim=1)
                mask[:t] = False
        return video, mask


def cardiac_aug_strong(img_size=128, scale=[0.08, 1.0], dytpe=torch.float32):
    return v2.Compose(
        [
            v2.RandomRotation(
                degrees=15,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            v2.RandomResizedCrop(size=img_size, scale=scale, ratio=[1.0, 1.0], antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.8, contrast=0.8, saturation=0, hue=0)],
                p=0.8,
            ),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0))], p=0.5),
            v2.ToDtype(dytpe, scale=True),
        ]
    )


def cardiac_aug_mild(img_size=128, scale=[0.9, 1.0], dytpe=torch.float32):
    return v2.Compose(
        [
            v2.RandomResizedCrop(size=img_size, scale=scale, ratio=[1.0, 1.0], antialias=True),
            v2.RandomHorizontalFlip(p=0.3),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
            v2.ToDtype(dytpe, scale=True),
        ]
    )


def cardiac_aug_mae(img_size=128, horizontal_flip=True, dytpe=torch.float32):
    return v2.Compose(
        [
            v2.Resize(size=img_size, antialias=True),
            v2.RandomHorizontalFlip(p=0.5) if horizontal_flip else None,
            v2.ToDtype(dytpe, scale=True),
        ]
    )


def oct_aug_strong(img_size=128, scale=[0.10, 1.0], dytpe=torch.float32):
    return v2.Compose(
        [
            v2.RandomRotation(
                degrees=5,
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            v2.RandomResizedCrop(size=img_size, scale=scale, ratio=[1.0, 1.0], antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.6, contrast=0.8, saturation=0, hue=0)],
                p=0.8,
            ),
            v2.RandomApply([v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))], p=0.5),
            v2.ToDtype(dytpe, scale=True),
        ]
    )


def oct_aug_mild(img_size=128, scale=[0.9, 1.0], dytpe=torch.float32):
    return v2.Compose(
        [
            v2.RandomResizedCrop(size=img_size, scale=scale, ratio=[1.0, 1.0], antialias=True),
            v2.RandomHorizontalFlip(p=0.3),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
            v2.ToDtype(dytpe, scale=True),
        ]
    )


def oct_aug_mae(img_size=128, horizontal_flip=True, dytpe=torch.float32):
    return v2.Compose(
        [
            v2.CenterCrop(size=img_size),
            v2.RandomHorizontalFlip(p=0.5) if horizontal_flip else None,
            v2.ToDtype(dytpe, scale=True),
        ]
    )
