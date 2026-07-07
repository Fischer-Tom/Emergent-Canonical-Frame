"""Minimal forward-pass model: image -> per-object mesh vertex correspondences.

Backbone (DINO + LoRA/adapter) -> MeshDecoder (mask-conditioned cross-attention,
one query set per object instance) -> MeshCorrespondenceHead (per-vertex pixel
similarity + mask logits). No criterion, rendering, or training-time losses —
see models/model.py for the full training model.

Every branch here is on static (trace-time) shapes, never on tensor values, so
the forward pass compiles cleanly with `torch.compile`.
"""

import copy
import math
import os.path
from typing import Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor, nn


# ---------------------------------------------------------------------------
# Backbone: DINO ViT + LoRA / bottleneck adapters for parameter-efficient tuning
# ---------------------------------------------------------------------------
class LoRAQKV(nn.Module):
    def __init__(self, qkv: nn.Linear, r=16, alpha=32, dropout=0.05, target=("q", "v")):
        super().__init__()
        self.qkv = qkv
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        assert self.out_features == 3 * self.in_features, "Expect fused qkv with out=3*in"
        self.dim = self.in_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        for p in self.qkv.parameters():
            p.requires_grad = False

        def make_pair():
            A = nn.Linear(self.dim, r, bias=False)
            B = nn.Linear(r, self.dim, bias=False)
            nn.init.kaiming_uniform_(A.weight, a=5**0.5)
            nn.init.zeros_(B.weight)
            return A, B

        self.target = set(target)
        self.A_q, self.B_q = make_pair() if "q" in self.target else (None, None)
        self.A_k, self.B_k = make_pair() if "k" in self.target else (None, None)
        self.A_v, self.B_v = make_pair() if "v" in self.target else (None, None)

    def forward(self, x):
        base = self.qkv(x)
        C = self.dim
        q, k, v = base[..., :C], base[..., C : 2 * C], base[..., 2 * C : 3 * C]

        if self.A_q is not None:
            q = q + self.B_q(self.lora_dropout(self.A_q(x))).to(q.dtype) * self.scaling
        if self.A_k is not None:
            k = k + self.B_k(self.lora_dropout(self.A_k(x))).to(k.dtype) * self.scaling
        if self.A_v is not None:
            v = v + self.B_v(self.lora_dropout(self.A_v(x))).to(v.dtype) * self.scaling

        return torch.cat([q, k, v], dim=-1)


class FFNAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck_dim: int = 64, s: float = 0.1, p: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(dim, eps=1e-6)
        self.down = nn.Linear(dim, bottleneck_dim, bias=True)
        self.up = nn.Linear(bottleneck_dim, dim, bias=True)
        self.drop = nn.Dropout(p)
        self.s = s

        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        z = self.down(self.ln(x))
        z = self.drop(F.gelu(z))
        return self.s * self.up(z)


class MLPWithAdapter(nn.Module):
    def __init__(self, mlp: nn.Module, dim: int, bottleneck_dim=64, s=0.1, p=0.0):
        super().__init__()
        self.mlp = mlp
        self.adapter = FFNAdapter(dim, bottleneck_dim=bottleneck_dim, s=s, p=p)
        for p0 in self.mlp.parameters():
            p0.requires_grad = False

    def forward(self, x):
        return self.mlp(x) + self.adapter(x)


def inject_adapters_and_lora_into_vit(
    vit,
    use_lora: bool,
    use_adapter: bool,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_targets=("q", "v"),
    adapter_bottleneck=64,
    adapter_scale=0.1,
    adapter_dropout=0.0,
    target_blocks=None,
):
    vit.requires_grad_(False)
    blocks = vit.blocks
    if target_blocks is None:
        target_blocks = range(len(blocks))

    for i in target_blocks:
        blk = blocks[i]
        if use_lora:
            attn = blk.attn
            if not (hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear)):
                raise ValueError(f"Block {i}: attn.qkv is not a fused nn.Linear; need custom mapping.")
            attn.qkv = LoRAQKV(attn.qkv, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target=lora_targets)
        if use_adapter:
            dim = getattr(blk.norm1, "normalized_shape", (blk.norm1.weight.shape[0],))[0]
            blk.mlp = MLPWithAdapter(
                blk.mlp, dim=dim, bottleneck_dim=adapter_bottleneck, s=adapter_scale, p=adapter_dropout,
            )
    return vit


def _set_train_mode_for_trainable_modules(module: nn.Module, mode: bool = True):
    module.eval()
    for m in module.modules():
        if any(p.requires_grad for p in m.parameters(recurse=False)):
            m.train(mode)


class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, pool_scales=(1, 2, 3, 6), dropout=0.1):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True),
            )
            for scale in pool_scales
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_scales) * out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x):
        outs = [x] + [
            F.interpolate(stage(x), size=x.shape[2:], mode="bilinear", align_corners=False)
            for stage in self.stages
        ]
        return self.bottleneck(torch.cat(outs, dim=1))


class UPerNetDecoderWithAux(nn.Module):
    def __init__(self, in_channels, ppm_channels=512, fpn_channels=512, dropout=0.1):
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, fpn_channels, kernel_size=1, bias=False),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=False),
            )
            for ch in in_channels[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, fpn_channels),
                nn.ReLU(inplace=False),
            )
            for _ in in_channels[:-1]
        ])
        self.ppm = PPM(in_channels[-1], fpn_channels, pool_scales=(1, 2, 3, 6), dropout=dropout)
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(len(in_channels) * fpn_channels, fpn_channels, 3, padding=1, bias=False),
        )

    def forward(self, feats):
        assert len(feats) == 4, "Expecting [P2, P3, P4, P5] features"
        laterals = [l_conv(feats[i]) for i, l_conv in enumerate(self.lateral_convs)]
        laterals.append(self.ppm(feats[-1]))

        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + up

        fpn_outs = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        fpn_outs.append(laterals[-1])
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=False)

        return self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))


class Feature2Pyramid(nn.Module):
    def __init__(self, embed_dim, rescales=(4, 2, 1, 0.5)):
        super().__init__()
        self.rescales = rescales
        self.ops = nn.ModuleList()
        for r in rescales:
            if r == 4:
                self.ops.append(nn.Sequential(
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2, bias=False),
                    nn.GroupNorm(32, embed_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                ))
            elif r == 2:
                self.ops.append(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            elif r == 1:
                self.ops.append(nn.Identity())
            elif r == 0.5:
                self.ops.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                raise KeyError(f"Invalid rescale factor: {r}")

    def forward(self, inputs):
        assert len(inputs) == len(self.rescales)
        return tuple(self.ops[i](f) for i, f in enumerate(inputs))


class DINO(nn.Module):
    def __init__(self, out_ch: int, cfg: DictConfig, out_indices=(6, 14, 18, 23), adapt: bool = False):
        super().__init__()
        weights_path = cfg.remote_weights if os.path.exists(cfg.remote_weights) else cfg.local_weights
        repo_path = cfg.remote_repo_dir if os.path.exists(cfg.remote_repo_dir) else cfg.local_repo_dir
        self.backbone = torch.hub.load(repo_path, cfg.model, source="local", pretrained=False)
        self.backbone.load_state_dict(torch.load(weights_path))

        self.finetune = cfg.get("finetune_backbone", False)
        if not self.finetune:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.neck = Feature2Pyramid(embed_dim=self.backbone.embed_dim)
        self.decoder = UPerNetDecoderWithAux(
            in_channels=[self.backbone.embed_dim] * 4, ppm_channels=out_ch, fpn_channels=out_ch,
        )
        depth = len(self.backbone.blocks)
        req = list(out_indices)
        self.out_indices = req if all(i < depth for i in req) else [round((k + 1) * depth / 4) - 1 for k in range(4)]

        self.adapt = adapt
        self.use_lora = cfg.get("use_lora", False)
        self.lora_frozen = False
        if (self.use_lora or self.adapt) and not self.finetune:
            inject_adapters_and_lora_into_vit(
                self.backbone,
                use_lora=self.use_lora,
                use_adapter=self.adapt,
                lora_r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                lora_targets=("q", "v"),
                adapter_bottleneck=cfg.bottleneck_dim,
                adapter_scale=cfg.get("adapter_scale", 0.1),
                adapter_dropout=cfg.get("adapter_dropout", 0.0),
                target_blocks=cfg.get("adapt_target_blocks", None),
            )

    def forward(self, x):
        lora_active = self.use_lora and not self.lora_frozen
        ctx = torch.enable_grad() if (self.finetune or self.adapt or lora_active) else torch.no_grad()
        with ctx:
            feat_maps = self.backbone.get_intermediate_layers(
                x, n=self.out_indices, reshape=True, norm=True, return_class_token=False
            )
        pyramid_feats = self.neck(tuple(feat_maps))
        return self.decoder(pyramid_feats)

    def train(self, mode=True):
        super().train(mode)
        if not self.finetune:
            if self.lora_frozen:
                self.backbone.eval()
            else:
                _set_train_mode_for_trainable_modules(self.backbone, mode=mode)


# ---------------------------------------------------------------------------
# Mesh decoder: cross-attends mesh-vertex queries to image features, one query
# set per object instance, restricted to that instance's pixels via an
# attention mask (no pixel gathering/sampling — shapes stay static).
# ---------------------------------------------------------------------------
class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale if scale is not None else 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        self.register_buffer("dim_t", dim_t, persistent=False)
        self._cache: dict = {}  # keyed on (H, W, device, dtype); grid has no learnable state

    def _build_grid(self, H, W, device, dtype):
        y_embed = torch.arange(1, H + 1, device=device, dtype=torch.float32).view(1, H, 1).expand(1, H, W)
        x_embed = torch.arange(1, W + 1, device=device, dtype=torch.float32).view(1, 1, W).expand(1, H, W)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.dim_t.to(device=device)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).to(dtype).contiguous()

    def forward(self, x):
        B, _, H, W = x.shape
        key = (H, W, x.device, x.dtype)
        if key not in self._cache:
            self._cache[key] = self._build_grid(H, W, x.device, x.dtype)
        return self._cache[key].expand(B, -1, -1, -1)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    @staticmethod
    def _with_pos(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
        q = k = self._with_pos(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt)[0]
        tgt = self.norm1(tgt + self.dropout1(tgt2))

        tgt2 = self.multihead_attn(
            query=self._with_pos(tgt, query_pos),
            key=self._with_pos(memory, pos),
            value=memory,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(tgt2))

        tgt2 = self.linear2(self.dropout(F.gelu(self.linear1(tgt))))
        return self.norm3(tgt + self.dropout3(tgt2))


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_layers)])
        self.norm = norm

    def forward(self, tgt, memory, memory_key_padding_mask=None, pos=None, query_pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, memory_key_padding_mask=memory_key_padding_mask, pos=pos, query_pos=query_pos)
        return self.norm(output) if self.norm is not None else output


class MeshDecoder(nn.Module):
    V: Tensor

    def __init__(self, V, n_blocks=6, n_heads=8, d_model=512, dim_feedforward=2048):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward)
        self.decoder = TransformerDecoder(decoder_layer, num_layers=n_blocks, norm=nn.LayerNorm(d_model))

        self.register_buffer("V", V)
        self.querys_embed = nn.Embedding(V.shape[0], d_model)
        self.positional_embedding_verts = nn.Sequential(
            nn.Linear(3, d_model), nn.ReLU(inplace=True), nn.Linear(d_model, d_model),
        )
        self.pos_embed_2d = PositionEmbeddingSine(d_model // 2, normalize=True)
        self.obj_cond_proj = nn.Linear(d_model, d_model)

    def forward(self, feats: Tensor, obj_masks: Optional[Tensor] = None):
        """One mesh-vertex query set per object instance.

        obj_masks: (B, N, H, W) or (B, H, W) per-instance silhouettes, used to
        (a) condition each object's queries with a masked-pooled feature and
        (b) restrict cross-attention to that instance's pixels. None means a
        single (N=1) whole-image object.

        Returns a list of N tensors, each (num_vertices, B, d_model).
        """
        if feats.dim() != 4:
            raise ValueError(f"feats must be (B,C,H,W), got {feats.shape}")
        B, C, H, W = feats.shape

        if obj_masks is None:
            obj_masks = feats.new_ones((B, 1, H, W))
        else:
            if obj_masks.dim() == 3:
                obj_masks = obj_masks.unsqueeze(1)
            if tuple(obj_masks.shape[-2:]) != (H, W):
                obj_masks = F.interpolate(obj_masks.float(), size=(H, W), mode="nearest")
            obj_masks = obj_masks.to(device=feats.device, dtype=feats.dtype)
        N = obj_masks.shape[1]

        memory = feats.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        pos = rearrange(self.pos_embed_2d(feats), "b c h w -> (h w) b c")
        if N > 1:
            memory = memory.repeat_interleave(N, dim=1)  # (HW, B*N, C)
            pos = pos.repeat_interleave(N, dim=1)

        mask_flat = obj_masks.flatten(2)  # (B, N, HW)
        feats_flat = feats.flatten(2).permute(0, 2, 1)  # (B, HW, C)
        obj_cond = torch.bmm(mask_flat, feats_flat) / mask_flat.sum(dim=-1, keepdim=True).clamp_min(1.0)
        obj_cond = self.obj_cond_proj(obj_cond).reshape(B * N, C)  # order: b*N + n

        query_content = self.querys_embed.weight.unsqueeze(1).repeat(1, B * N, 1) + obj_cond.unsqueeze(0)
        vertex_pos = self.positional_embedding_verts(self.V).unsqueeze(1).repeat(1, B * N, 1)

        # A fully-masked instance (empty silhouette) would leave a query with
        # no valid key to attend to, which MultiheadAttention turns into NaN.
        # Un-mask those rows unconditionally instead of branching on `.any()`
        # (a data-dependent branch would force a host sync / graph break).
        key_padding_mask = mask_flat.reshape(B * N, H * W) <= 0
        all_masked = key_padding_mask.all(dim=1, keepdim=True)
        key_padding_mask = key_padding_mask & ~all_masked

        hs = self.decoder(
            tgt=query_content, memory=memory, memory_key_padding_mask=key_padding_mask, pos=pos, query_pos=vertex_pos,
        )  # (Q, B*N, D)

        Q = hs.shape[0]
        hs = hs.view(Q, B, N, -1)
        return [hs[:, :, i, :] for i in range(N)]


# ---------------------------------------------------------------------------
# Correspondence head: per-vertex pixel similarity + a mask logit derived from
# the same similarity (no separate dense mask conv head).
# ---------------------------------------------------------------------------
class MeshCorrespondenceHead(nn.Module):
    def __init__(self, d_model: int = 512, d_desc: int = 512, mask_topk: int = 8, mask_hidden: int = 32):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_desc)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(d_model, d_desc, 1),
            nn.GELU(),
            nn.Conv2d(d_desc, d_desc, 1),
            nn.GELU(),
            nn.Conv2d(d_desc, d_desc, 1),
        )
        self.mask_topk = mask_topk
        self.mask_head = nn.Sequential(nn.Linear(2, mask_hidden), nn.GELU(), nn.Linear(mask_hidden, 1))

    def forward(self, hs: Tensor, feats: Tensor):
        # hs: (num_queries, B, d_model)  feats: (B, d_model, H, W)
        # returns sim: (B, Q, H*W), mask_logits: (B, 1, H, W)
        q_desc = F.normalize(self.query_proj(hs.permute(1, 0, 2)), dim=-1)

        k_desc = self.feat_proj(feats)
        B, D, H, W = k_desc.shape
        k_desc = F.normalize(k_desc.flatten(2).transpose(1, 2), dim=-1)

        sim = torch.matmul(q_desc, k_desc.transpose(1, 2))  # (B, Q, H*W)

        # Pixel is foreground if it matches some vertex well. Pooled top-k
        # stats keep the mask head's param count independent of vertex count.
        k = min(self.mask_topk, sim.shape[1])
        topk_sim = sim.topk(k, dim=1).values  # (B, k, H*W)
        stats = torch.stack([topk_sim[:, 0, :], topk_sim.mean(dim=1)], dim=-1)  # (B, H*W, 2)
        mask_logits = self.mask_head(stats).squeeze(-1).view(B, 1, H, W)

        return sim, mask_logits

class Model(nn.Module):
    def __init__(
        self,
        cfg,
        V: Tensor,
        d_model: int = 512,
        d_desc: int = 512,
        n_blocks: int = 6,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.backbone = DINO(d_model, cfg.model, adapt=cfg.model.get("adapt", True))
        self.mesh_decoder = MeshDecoder(V, n_blocks=n_blocks, n_heads=n_heads, d_model=d_model, dim_feedforward=dim_feedforward)
        self.correspondence_head = MeshCorrespondenceHead(d_model=d_model, d_desc=d_desc)
        self.register_buffer("V", V)

    def forward(self, img: Tensor, obj_masks: Optional[Tensor] = None):
        """
        img: (B, 3, H, W)
        obj_masks: (B, N, H, W) or (B, H, W) per-instance silhouettes, or None for N=1.
        Returns: feats, mesh_descriptors_list, logits_list, mask_logits_list — one
        entry per object instance N.
        """
        feats = self.backbone(img)
        mesh_descriptors_list = self.mesh_decoder(feats, obj_masks)
        logits_list, mask_logits_list = [], []
        for md in mesh_descriptors_list:
            logits, mask_logits = self.correspondence_head(md, feats)
            logits_list.append(logits)
            mask_logits_list.append(mask_logits)
        return feats, mesh_descriptors_list, logits_list, mask_logits_list
