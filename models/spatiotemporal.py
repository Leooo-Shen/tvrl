import torch
from torch import nn
from models.vit_helper import ViT_Backbone
from models.videovit import TimeEmbedding
from models.videovit import VideoViT


class SpatioTemporalEncoder(nn.Module):
    def __init__(self, cfg):
        super(SpatioTemporalEncoder, self).__init__()
        self.cfg = cfg
        self.lr = float(cfg.solver.lr) * float(cfg.data.train_batch_size) / 256
        self.epochs = cfg.trainer.epochs
        self.weight_decay = float(cfg.solver.weight_decay)
        self.optimizer_type = cfg.solver.optimizer
        self.lr_scheduler_type = cfg.solver.lr_scheduler
        self.warmup_epochs = cfg.solver.warmup_epochs

        self.spatio_encoder = VideoViT(
            frame_size=cfg.data.img_size,
            channels=1,
            num_frames=1,
            patch_spatial=cfg.model.vit.patch_spatial,
            patch_temporal=1,
            vit_config=cfg.model.vit.arch,
            drop_rate=cfg.model.vit.drop_rate,
            drop_path_rate=cfg.model.vit.drop_path_rate,
        )

        self.embed_dim = self.spatio_encoder.embed_dim

        self.temporal_dim = cfg.get("temporal_dim", 384)
        self.temporal_depth = cfg.get("temporal_depth", 3)
        self.mid_project = (
            nn.Linear(self.embed_dim, self.temporal_dim) if self.embed_dim != self.temporal_dim else nn.Identity()
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.temporal_dim))
        self.downstream_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.use_time_embed = cfg.use_time_embed
        print(f"[*] Using time embedding: {self.use_time_embed}")

        # use learnable time embedding
        if self.use_time_embed:
            learnable_time_embed = cfg.learnable_time_embed
            print(f"[*] Using learnable time embedding: {learnable_time_embed}")
            self.time_embed = TimeEmbedding(self.temporal_dim, learnable=learnable_time_embed)
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.data.clip_frames, self.embed_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, cfg.data.clip_frames, self.embed_dim))

        self.temporal_encoder = ViT_Backbone(
            embed_dim=self.temporal_dim, depth=self.temporal_depth, num_heads=6, mlp_ratio=4
        )

        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.downstream_cls_token, std=0.02)
        self.apply(self._init_weights)

        spatial_pretrained_weights = cfg.get("spatial_pretrained_weights", None)
        if spatial_pretrained_weights is not None:
            prefix = "encoder."
            state_dict = torch.load(spatial_pretrained_weights, map_location="cpu")["state_dict"]
            new_state_dict = {}
            if any(k.startswith(prefix) for k in state_dict.keys()):
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        k = k[len(prefix) :]
                        new_state_dict[k] = v

            self.spatio_encoder.load_state_dict(new_state_dict)
            print(f"[*] Loaded spatial pretrained weights from {spatial_pretrained_weights}")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"downstream_cls_token", "cls_token"}

    def _init_weights(self, m):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.downstream_cls_token, std=0.02)
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_spatio_encoder(self, x):
        # spatio encoder extract features per frame, thus do not need attention mask
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, 1, h, w)
        x = self.spatio_encoder(x)
        x = self.mid_project(x)
        return x.view(b, t, -1)

    def prepare_temporal_token(self, x, timestep):
        # add the time/pos embedding
        if self.use_time_embed:
            x += self.pos_embed
            x += self.time_embed(timestep)
        else:
            x += self.pos_embed
        return x

    def forward_temporal_encoder(self, x, attn_mask, time_step, prepare_token=True):
        # add the time/pos embedding
        if prepare_token:
            x = self.prepare_temporal_token(x, time_step)

        # do not mask the cls token and the downstream cls token
        cls_mask = torch.ones((attn_mask.shape[0], 1), dtype=torch.bool, device=attn_mask.device)
        attn_mask = torch.cat((cls_mask, attn_mask), dim=1)

        # concat the cls token and the downstream cls token to the beginning of the sequence
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.temporal_encoder(x, attn_mask=attn_mask)
        return x

    def forward(self, x, attn_mask, time_step, feat_op="cls", return_spatial=False):
        x_spatial = self.forward_spatio_encoder(x)
        x = self.forward_temporal_encoder(x_spatial, attn_mask, time_step)

        if feat_op == "cls":
            x = x[:, 0, :]
        elif feat_op == "pool":
            # remove the cls token
            x = x[:, 1:, :]
            # average pool the embeddings where attention mask is 1
            x *= attn_mask.unsqueeze(-1)
            x = x.sum(dim=1) / attn_mask.sum(dim=1, keepdim=True)
        if return_spatial:
            return x_spatial, x
        return x
