import os
import torch
import numpy as np
import pandas as pd
import ast
from datamodule.tasks import oct_task_list
from torchvision.transforms import v2
from datamodule.oct_datasets import OCTDataset
from datamodule.cardiac_datasets import CardiacDataset
from models.spatiotemporal import SpatioTemporalEncoder

import hydra
from omegaconf import DictConfig
import random


def build_model(cfg):
    encoder = SpatioTemporalEncoder(cfg)
    pretrained_weights = cfg.get("pretrained_weights", None)
    if pretrained_weights is not None and os.path.exists(pretrained_weights):
        print("[*] Loading pretrained weights from: ", pretrained_weights)
        state_dict = torch.load(pretrained_weights, map_location="cpu")["state_dict"]

        prefix = "encoder."
        prefix_st = "st."

        # common prefix for the encoder
        if any(k.startswith(prefix) for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    k = k[len(prefix) :]
                    new_state_dict[k] = v

        elif any(k.startswith(prefix_st) for k in state_dict.keys()):
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith(prefix_st):
                    k = k[len(prefix_st) :]
                    new_state_dict[k] = v

        encoder.load_state_dict(new_state_dict, strict=False)
        print("[*] Loaded pretrained weights successfully")
    else:
        print("[*] No pretrained weights found")
    return encoder


def build_oct_dataset(df_path, label):
    img_dir = ""
    basic_t = v2.Compose(
        [
            v2.CenterCrop(size=128),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )

    df = pd.read_csv(df_path)
    df["Scan_history"] = df["Scan_history"].apply(lambda x: ast.literal_eval(x))
    df["Scan_lenght"] = df["Scan_history"].apply(len)
    df["Video"] = df["Video"].apply(lambda x: ast.literal_eval(x))
    _, filter_condition = oct_task_list[label]
    df = df[df.apply(filter_condition, axis=1)].reset_index() if filter_condition else df

    # some preprocess to run test on specific longtiudinal sequences
    df = df[df["Scan_lenght"] > 8]
    df = df[df[label] == 1]

    return OCTDataset(
        df,
        img_dir=img_dir,
        clip_frames=8,
        stride=1,
        target=label,
        transforms=basic_t,
        stage="test",
        pad_clip=False,
        cache=None,
        relative_time_embed=True,
    )


def extract_oct_latents(model, num_samples=4):
    out_dict = {}
    for label in [
        "Late",
        "Converts_to_cRORA of 1000 um_within_3_years",
        "Converts_to_CNV_within_3_years",
        # "Converts_to_Scar_within_3_years",
    ]:

        oct_ds = build_oct_dataset("", label)
        print(f"dataset has {len(oct_ds)} samples ...")

        images = []
        latents = []
        for idx in range(num_samples):
            idx += random.randint(0, len(oct_ds) - num_samples - 1)
            # idx += 12
            print("idx:", idx)
            clip, _, attn_mask, time_step = oct_ds[idx]
            # print(clip.shape, attn_mask, time_step)
            images.append(clip)

            clip = clip.unsqueeze(0).cuda()
            attn_mask = attn_mask.unsqueeze(0).cuda()
            time_step = time_step.unsqueeze(0).cuda()

            with torch.no_grad():
                x_spatial = model.forward_spatio_encoder(clip)
                out = model.forward_temporal_encoder(x_spatial, attn_mask, time_step)[:, 1:, :]
                latents.append(out.cpu())
            # print("output tensor shape:", out.shape)

        out_dict["img_" + label] = torch.stack(images)
        out_dict[label] = torch.stack(latents)
    torch.save(out_dict, "oct_latents.pth")


def build_cardiac_dataset(df_path, label):
    img_dir = ""
    basic_t = v2.Compose([v2.Resize(128), v2.ToDtype(torch.float32, scale=True)])

    df = pd.read_csv(os.path.join(df_path), dtype={"eid": str})

    if label in ["CAD_broad", "Fibrillation"]:
        df = df[df[label] == 1]
    elif (
        label == "LVEF"
    ):  # LVEF acts like a control task. THis is dirty, but otherwise needs to modify the cardiac task list
        # filter out the rows where both CAD_broad and Fibrillation are 0
        condition = (
            (df["CAD_broad"] == 0) & (df["Fibrillation"] == 0) & (df["Infarction"] == 0) & (df["Hypertension"] == 0)
        )
        df = df[condition]
        print(f"Control dataset has {len(df)} samples ...")

    return CardiacDataset(
        df,
        img_dir=img_dir,
        target=label,
        clip_frames=8,
        stride=1,
        n_clips=1,
        transforms=basic_t,
        dense_sampling=True,
        stage="test",
        cache=None,
        relative_time_embed=False,
        return_original_video=True,
    )


def extract_cardiac_latents(model, num_samples=4):
    cardiac_dict = {}
    num_frames = 50

    for label in ["CAD_broad", "Fibrillation", "LVEF"]:

        cardiac_ds = build_cardiac_dataset(
            "test.csv", label
        )
        print(f"dataset has {len(cardiac_ds)} samples ...")

        videos = []
        latents = []
        embeddings = torch.zeros((num_frames, 384))
        counts = torch.zeros(num_frames)
        # video = torch.zeros((num_frames, 128, 128))

        for idx in range(num_samples):
            idx += random.randint(0, 10)
            # use dense sampling with sliding windows
            clips_dense, _, attn_masks_dense, time_steps_dense, original_video = cardiac_ds[idx]

            for clip, attn_mask, abs_frame_index in zip(clips_dense, attn_masks_dense, time_steps_dense):
                # relative time embedding
                time_step = abs_frame_index - abs_frame_index[0]

                clip = clip.unsqueeze(0).cuda()
                attn_mask = attn_mask.unsqueeze(0).cuda()
                time_step = time_step.unsqueeze(0).cuda()

                with torch.no_grad():
                    x_spatial = model.forward_spatio_encoder(clip)
                    out = model.forward_temporal_encoder(x_spatial, attn_mask, time_step)[:, 1:, :].cpu().squeeze(0)

                # accumulate embeddings for each frame
                abs_frame_index = abs_frame_index.numpy().astype(int)
                embeddings[abs_frame_index] += out
                counts[abs_frame_index] += 1

            # average embeddings for each frame
            averaged_embeddings = embeddings / counts[:, None]
            latents.append(averaged_embeddings)
            videos.append(original_video)
            print("embed:", averaged_embeddings.shape)
            print("original video shape:", original_video.shape)

        cardiac_dict[label] = torch.stack(latents)
        cardiac_dict["img_" + label] = torch.stack(videos)

    torch.save(cardiac_dict, "cardiac_latents.pth")


@hydra.main(config_path="config", config_name="finetune")
def main(cfg: DictConfig):
    # remeber to load different checkpoints pretrained for cardiac and oct
    model = build_model(cfg)
    model = model.eval().cuda()

    extract_oct_latents(model)
    # extract_cardiac_latents(model)


if __name__ == "__main__":
    main()
