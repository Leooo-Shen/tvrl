import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import math
from timm.models.layers import trunc_normal_, to_2tuple

try:
    from .vit_helper import Block
    from .weight_init_helper import init_weights_conv3d
except:
    from vit_helper import Block
    from weight_init_helper import init_weights_conv3d


def return_vit_config(vit_config):
    # return  embed_dim, depth, num_heads
    if vit_config == "tiny":
        return 192, 12, 3
    elif vit_config == "small":
        return 384, 12, 6
    elif vit_config == "small_st":
        return 384, 15, 6
    elif vit_config == "base":
        return 768, 12, 12
    elif vit_config == "large":
        return 1024, 24, 16
    else:
        raise NotImplementedError(f"vit_config: {vit_config} is not available")


class Cuboids(nn.Module):
    """
    the frame sequences are transformed into smaller 3d cuboids;
    3d cuboids are directly projected to linear embeddings;
    op dimension: batch size x number of cuboids x embed dimension
    """

    def __init__(self, embedding_dim, tubelet_t, tubelet_h, tubelet_w, in_channels):
        super().__init__()
        tubelet_dim = in_channels * tubelet_h * tubelet_w * tubelet_t
        self.patches = Rearrange(
            "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)",
            pt=tubelet_t,
            ph=tubelet_h,
            pw=tubelet_w,
        )
        self.proj = nn.Linear(tubelet_dim, embedding_dim)

    def forward(self, x, return_patches=False):
        patches = self.patches(x)
        embeddings = self.proj(patches)
        if return_patches:
            return patches, embeddings
        return embeddings


# class PatchEmbed(nn.Module):
#     """Video to Patch Embedding with 3D CNN"""

#     def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=768, tubelet_size=2):
#         super().__init__()
#         img_size = to_2tuple(img_size) if isinstance(img_size, int) else img_size
#         patch_size = to_2tuple(patch_size) if isinstance(patch_size, int) else patch_size
#         # self.num_patches = (
#         #     (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (num_frames // tubelet_size)
#         # )
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.proj = nn.Conv3d(
#             in_channels=in_chans,
#             out_channels=embed_dim,
#             kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
#             stride=(tubelet_size, patch_size[0], patch_size[1]),
#         )

#     def forward(self, x):
#         B, C, T, H, W = x.shape
#         # FIXME look at relaxing size constraints
#         assert (
#             H == self.img_size[0] and W == self.img_size[1]
#         ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
#         x = self.proj(x).flatten(2).transpose(1, 2)
#         return x


class ConvStem(nn.Module):
    """
    3D ConvStem, similar to Early Convolutions Help Transformers See Better, Tete et al. https://arxiv.org/abs/2106.14881
    """

    def __init__(self, img_size=128, patch_size=16, in_chans=1, embed_dim=768, norm_layer=None, tubelet_size=2):
        super().__init__()

        assert patch_size == 16, "ConvStem only supports patch size of 16"
        assert embed_dim % 8 == 0, "Embed dimension must be divisible by 8 for ConvStem"

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # build stem, similar to the design in https://arxiv.org/abs/2106.14881
        stem = []
        input_dim, output_dim = in_chans, embed_dim // 8
        for l in range(4):
            stem.append(
                nn.Conv3d(
                    input_dim, output_dim, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False
                )
            )
            stem.append(nn.BatchNorm3d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv3d(input_dim, embed_dim, kernel_size=(tubelet_size, 1, 1), stride=(tubelet_size, 1, 1)))
        self.proj = nn.Sequential(*stem)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# class PosEmbedding(nn.Module):
#     # copied from https://github.com/SforAiDl/vformer/blob/main/vformer/encoder/embedding/pos_embedding.py#L77

#     def __init__(self, shape, dim, drop=None, sinusoidal=False, std=0.02):
#         super(PosEmbedding, self).__init__()

#         if not sinusoidal:
#             if isinstance(shape, int):
#                 shape = [1, shape, dim]
#             else:
#                 shape = [1] + list(shape) + [dim]
#             self.pos_embed = nn.Parameter(torch.zeros(shape))

#         else:
#             pe = torch.FloatTensor([[p / (10000 ** (2 * (i // 2) / dim)) for i in range(dim)] for p in range(shape)])
#             pe[:, 0::2] = torch.sin(pe[:, 0::2])
#             pe[:, 1::2] = torch.cos(pe[:, 1::2])
#             self.pos_embed = pe
#             self.pos_embed.requires_grad = False
#         trunc_normal_(self.pos_embed, std=std)
#         self.pos_drop = nn.Dropout(drop) if drop is not None else nn.Identity()

#     def forward(self, x, cls_token=False):
#         # NOTE: this is never used
#         print("pos embedding forward")
#         if cls_token:
#             x = x + self.pos_embed[:, 1:, :]
#         else:
#             x = x + self.pos_embed
#         return self.pos_drop(x)


class TimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=100, learnable=True):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # use sinusoidal embedding
        if not learnable:
            self.time_embed = nn.Identity()

        # create sinusoidal embedding then pass through MLP
        # following the design in stable diffusion models
        else:
            self.time_embed = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )

    @staticmethod
    def time_step_embedding(time_steps, dim, max_period=10):
        """
        ## Create sinusoidal time step embeddings

        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        assert len(time_steps.shape) == 1, "Timesteps should be a 1d-array"
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        args = time_steps[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, time_steps: torch.Tensor):
        b, l = time_steps.shape
        time_steps = time_steps.view(-1)
        time_embedding = self.time_step_embedding(time_steps, self.dim, self.max_period).view(b, l, -1)
        time_embedding = self.time_embed(time_embedding)
        return time_embedding


class VideoViT(nn.Module):
    def __init__(
        self,
        frame_size=128,
        channels=1,
        num_frames=8,
        patch_spatial=16,
        patch_temporal=2,
        vit_config="base",
        mlp_ratio=4,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        use_convstem=True,
    ):
        super().__init__()

        embed_dim, depth, num_heads = return_vit_config(vit_config)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        frame_size = to_2tuple(frame_size)
        patch_spatial = to_2tuple(patch_spatial)
        self.frame_size = frame_size
        self.patch_spatial = patch_spatial
        self.num_frames = num_frames
        self.patch_temporal = patch_temporal
        self.embed_dim = embed_dim
        self.num_patches = (
            (frame_size[0] // patch_spatial[0]) * (frame_size[1] // patch_spatial[1]) * (num_frames // patch_temporal)
        )
        self.n_repeat_interleave = self.num_patches // num_frames

        # # TODO: remove this. only for 1c1f with temporal MLP
        # self.num_cuboids = (
        #     (frame_size[1] // patch_spatial[0]) * (frame_size[2] // patch_spatial[1]) // patch_temporal
        # )  # number of smaller cuboids

        if use_convstem:
            self.patch_embed = ConvStem(
                img_size=frame_size[0],
                patch_size=patch_spatial[0],
                in_chans=channels,
                embed_dim=embed_dim,
                tubelet_size=patch_temporal,
                norm_layer=None,
            )
        else:
            self.patch_embed = Cuboids(
                embedding_dim=embed_dim,
                tubelet_t=patch_temporal,
                tubelet_h=patch_spatial[1],
                tubelet_w=patch_spatial[0],
                in_channels=channels,
            )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))

        self.norm = norm_layer(embed_dim)
        self.initialize_weights()
        self.first_pass = None  # for debug to interpolate pos_encoding

    def initialize_weights(self):
        init_weights_conv3d(self.patch_embed)
        torch.nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def prepare_token(self, x, attn_mask=None, time_step=None):
        B, nc, t, w, h = x.shape

        x = self.patch_embed(x)
        pos_embed = self.interpolate_pos_encoding(x, t, w, h)

        # expand the attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.n_repeat_interleave, dim=1)

            # do not mask the cls token
            mask_token = torch.ones((attn_mask.shape[0], 1), dtype=torch.bool, device=attn_mask.device)
            attn_mask = torch.cat((mask_token, attn_mask), dim=1)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += pos_embed

        return x, attn_mask

    def interpolate_pos_encoding(self, x, t, w, h):
        npatch = x.shape[1]
        N = self.pos_embed.shape[1] - 1

        if npatch == N and w == h:
            return self.pos_embed

        if self.first_pass is None:  # debug log at first pass
            print(f"Number of pos embed changed from {N} to {npatch}")
            self.first_pass = 1

        # Ns = N//self.patch_temporal
        # Nt = self.patch_temporal
        # the above lines are wrong;
        Nt = self.num_frames // self.patch_temporal
        Ns = N // Nt

        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]

        dim = x.shape[-1]
        w0 = w // self.patch_spatial[0]
        h0 = h // self.patch_spatial[1]
        t0 = t // self.patch_temporal
        w0, h0, t0 = w0 + 0.1, h0 + 0.1, t0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(Nt), int(math.sqrt(Ns)), int(math.sqrt(Ns)), dim).permute(0, 4, 1, 2, 3),
            scale_factor=(t0 / Nt, w0 / math.sqrt(Ns), h0 / math.sqrt(Ns)),
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )  # for videos
        assert (
            int(t0) == patch_pos_embed.shape[-3]
            and int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward_features(self, x, attn_mask=None, time_step=None):
        """pass features w/o head"""
        # simply extract feature for downstream task
        x, attn_mask = self.prepare_token(x, attn_mask, time_step)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward(self, x, attn_mask=None, time_step=None, feat_op="cls"):
        """fwd pass through head. ability to choose which feature to pass"""
        assert feat_op in ["pool", "cls"]
        # simply extract feature for downstream task
        x, attn_mask = self.prepare_token(x, attn_mask, time_step)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        x = self.norm(x)

        if feat_op == "pool":
            x = x[:, 1:, :].mean(dim=1)
        elif feat_op == "cls":
            x = x[:, 0]
        else:
            raise ValueError(f"feat_op should be either pool or cls; given {feat_op}")

        return x

    def get_last_selfattention(self, x, attn_mask=None, time_step=None):
        x, attn_mask = self.prepare_token(x, attn_mask, time_step)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, attn_mask=attn_mask)
            else:
                # return attention of the last block
                return blk(x, return_attention=True, attn_mask=attn_mask)

    def get_intermediate_layers(self, x, attn_mask=None, time_step=None, n=1):
        # we return the output tokens from the `n` last blocks
        output = []
        x, attn_mask = self.prepare_token(x, attn_mask, time_step)
        for i, blk in enumerate(self.blocks):
            x = blk(x, attn_mask=attn_mask)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def build_vit(
    frame_size,
    channels,
    num_frames,
    patch_spatial,
    patch_temporal,
    vit_config="small",
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_convstem=True,
):
    vit = VideoViT(
        frame_size=frame_size,
        channels=channels,
        num_frames=num_frames,
        patch_spatial=patch_spatial,
        patch_temporal=patch_temporal,
        vit_config=vit_config,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        use_convstem=use_convstem,
    )

    return vit


def build_vit_with_cfg(cfg):
    channels = cfg.data.get("channels", 1)
    use_convstem = cfg.model.vit.get("use_convstem", True)
    vit = build_vit(
        frame_size=cfg.data.img_size,
        channels=channels,
        num_frames=cfg.data.clip_frames,
        patch_spatial=cfg.model.vit.patch_spatial,
        patch_temporal=cfg.model.vit.patch_temporal,
        vit_config=cfg.model.vit.arch,
        drop_rate=cfg.model.vit.drop_rate,
        drop_path_rate=cfg.model.vit.drop_path_rate,
        use_convstem=use_convstem,
    )
    return vit
