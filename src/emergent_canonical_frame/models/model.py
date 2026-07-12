
import copy
import math
import os.path
from typing import Optional, Mapping, Any

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig
from torch import Tensor, nn

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


def inject_lora_into_vit(
    vit,
    use_lora: bool,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    lora_targets=("q", "v"),
):
    vit.requires_grad_(False)
    blocks = vit.blocks
    target_blocks = range(len(blocks))

    for i in target_blocks:
        blk = blocks[i]
        if use_lora:
            attn = blk.attn
            if not (hasattr(attn, "qkv") and isinstance(attn.qkv, nn.Linear)):
                raise ValueError(f"Block {i}: attn.qkv is not a fused nn.Linear; need custom mapping.")
            attn.qkv = LoRAQKV(attn.qkv, r=lora_r, alpha=lora_alpha, dropout=lora_dropout, target=lora_targets)
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
    def __init__(
        self,
        out_ch: int,
        backbone_name: str = "dinov3_vitl16",
        repo_path: str | None = None,
        weights_path: str | None = None,
        out_indices: tuple[int, int, int, int] = (6, 14, 18, 23),
        lora: Mapping[str, Any] | None = None,
    ):
        super().__init__()

        lora = dict(lora or {})

        self.backbone = torch.hub.load(
            repo_path,
            backbone_name,
            source="local",
            pretrained=False,
        )
        if os.path.exists(weights_path):
            self.backbone.load_state_dict(torch.load(weights_path))
            print(f"Loaded weights from {weights_path}")

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.neck = Feature2Pyramid(embed_dim=self.backbone.embed_dim)
        self.decoder = UPerNetDecoderWithAux(
            in_channels=[self.backbone.embed_dim] * 4,
            ppm_channels=out_ch,
            fpn_channels=out_ch,
        )

        depth = len(self.backbone.blocks)
        req = list(out_indices)
        self.out_indices = (
            req
            if all(i < depth for i in req)
            else [round((k + 1) * depth / 4) - 1 for k in range(4)]
        )

        self.finetune = False
        self.use_lora = lora.get("use", True)
        self.lora_frozen = False

        if self.use_lora:
            inject_lora_into_vit(
                self.backbone,
                use_lora=self.use_lora,
                lora_r=int(lora.get("r", 8)),
                lora_alpha=int(lora.get("alpha", 8)),
                lora_dropout=float(lora.get("dropout", 0.1)),
                lora_targets=tuple(lora.get("targets", ("q", "v"))),
            )

    def forward(self, x):
        lora_active = self.use_lora and not self.lora_frozen
        requires_backbone_grad = self.training and lora_active
        context = (
            torch.enable_grad() if requires_backbone_grad else torch.no_grad()
        )
        with context:
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

        dim_t = self.dim_t.to(device=device, dtype=torch.float32)     
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2).to(dtype).contiguous()

    def forward(self, x):
        B, _, H, W = x.shape
    
        if x.is_cuda and torch.is_autocast_enabled("cuda"):
            target_dtype = torch.get_autocast_dtype("cuda")
        else:
            target_dtype = x.dtype
    
        key = (H, W, x.device, target_dtype)
    
        if key not in self._cache:
            self._cache[key] = self._build_grid(H, W, x.device, target_dtype)
    
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

        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
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

    def __init__(self, V, n_blocks=4, n_heads=8, d_model=256, dim_feedforward=512):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward
        )
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers=n_blocks, norm=nn.LayerNorm(d_model)
        )

        self.register_buffer("V", V)
        self.querys_embed = nn.Embedding(V.shape[0], d_model)
        self.positional_embedding_verts = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.pos_embed_2d = PositionEmbeddingSine(d_model // 2, normalize=True)

    def forward(self, feats):
        if feats.dim() != 4:
            raise ValueError(f"feats must be (B,C,H,W), got {feats.shape}")

        memory = feats.flatten(2).permute(2, 0, 1)  # (HW, B, C)
        pos = rearrange(self.pos_embed_2d(feats), "b c h w -> (h w) b c")

        B = memory.shape[1]
        query_content = self.querys_embed.weight.unsqueeze(1).repeat(1, B, 1)
        vertex_pos = self.positional_embedding_verts(self.V).unsqueeze(1).repeat(1, B, 1)

        return self.decoder(tgt=query_content, memory=memory, pos=pos, query_pos=vertex_pos)


class MaskHead(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(d_model, d_model // 2, 3, padding=1),
            nn.GroupNorm(16, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // 2, d_model // 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.GroupNorm(16, d_model // 4),
            nn.Conv2d(d_model // 4, 1, 1),
        )

    def forward(self, feats):
        return self.head(feats)


class MeshCorrespondenceHead(nn.Module):
    def __init__(self, d_model: int = 256, d_desc: int = 128):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_desc)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(d_model, d_desc, 1),
            nn.GELU(),
            nn.Conv2d(d_desc, d_desc, 1),
            nn.GELU(),
            nn.Conv2d(d_desc, d_desc, 1),
        )

    def forward(self, hs: Tensor, feats: Tensor):
        # hs: (num_queries, B, d_model)  feats: (B, d_model, H, W)
        q = hs.permute(1, 0, 2)
        q_desc = self.query_proj(q)

        k_desc = self.feat_proj(feats)
        B, D, H, W = k_desc.shape
        k_desc = k_desc.flatten(2).transpose(1, 2)

        q_desc = F.normalize(q_desc, dim=-1)
        k_desc = F.normalize(k_desc, dim=-1)

        return torch.matmul(q_desc, k_desc.transpose(1, 2))  # (B, Q, H*W)

class Model(nn.Module):
    def __init__(
        self,
        repo_path: str | None = None,
        backbone_name: str = "dinov3_vitl16",
        weights_path: str | None = None,
        output_size: tuple[int, int] = (64, 64),
        lora: Mapping[str, Any] | None = None,
        mesh_subdivisions: int = 8,
        d_model: int = 512,
        d_desc: int = 512,
        n_blocks: int = 6,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        downsample_factor: int = 4,
    ):
        super().__init__()
        V, F = healpix_cube_mesh(subdivisions=mesh_subdivisions)
        self.register_buffer("V", V)
        self.register_buffer("F", F)

        self.backbone_name = backbone_name
        self.output_size = tuple(output_size)
        self.downsample_factor = downsample_factor
        self.lora_cfg = dict(lora) if lora is not None else {}

        self.backbone = DINO(
            out_ch=d_model,
            backbone_name=backbone_name,
            repo_path=repo_path,
            weights_path=weights_path,
            lora=self.lora_cfg,
        )

        self.mesh_decoder = MeshDecoder(
            V,
            n_blocks=n_blocks,
            n_heads=n_heads,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
        )

        self.correspondence_head = MeshCorrespondenceHead(
            d_model=d_model,
            d_desc=d_desc,
        )
        self.mask_head = MaskHead(d_model=d_model)


    def forward(self, img: Tensor):
        feats = self.backbone(img)
        mesh_descriptors = self.mesh_decoder(feats)
        logits = self.correspondence_head(mesh_descriptors, feats)
        mask_logits = self.mask_head(feats)
        return logits, mask_logits

    @torch.no_grad()
    def infer(
        self,
        batch,
        annotations=None,
        *,
        use_gt_mask: bool = False,
    ) -> dict:
        """Return sparse pixel-to-canonical-vertex matches for pose fitting."""
        annotations = annotations if annotations is not None else batch.annotations
        if annotations is None:
            raise ValueError("Model.infer requires rendered annotations")

        logits, mask_logits = self(batch.image_rgb)
        B, _M, HW = logits.shape
        H, W = mask_logits.shape[-2:]
        if H * W != HW:
            raise ValueError(
                f"Correspondence logits have {HW} pixels, mask has {H}x{W}"
            )

        vertex_idx = logits.argmax(dim=1)
        matched_vertices = self.V[vertex_idx].view(B, H, W, 3)

        if batch.valid_masks is None:
            valid_region = torch.ones(
                (B, H, W), device=logits.device, dtype=torch.bool
            )
        else:
            valid_region = F.interpolate(
                batch.valid_masks.float(),
                size=(H, W),
                mode="nearest",
            ).squeeze(1).bool()

        if use_gt_mask:
            gt_mask = annotations.proxy_mask
            if tuple(gt_mask.shape[-2:]) != (H, W):
                gt_mask = F.interpolate(
                    gt_mask.float().unsqueeze(1),
                    size=(H, W),
                    mode="nearest",
                ).squeeze(1).bool()
            pred_mask = gt_mask.bool() & valid_region
        else:
            pred_mask = (
                torch.sigmoid(mask_logits).squeeze(1) > 0.5
            ) & valid_region

        selected = pred_mask.nonzero(as_tuple=False)
        if selected.numel() == 0:
            b_sel = torch.empty(0, device=logits.device, dtype=torch.long)
            yx_sel = torch.empty(
                (0, 2), device=logits.device, dtype=torch.long
            )
            m_v3d = torch.empty(
                (0, 3), device=logits.device, dtype=self.V.dtype
            )
        else:
            b_sel = selected[:, 0]
            yx_sel = selected[:, 1:3]
            m_v3d = matched_vertices[
                selected[:, 0], selected[:, 1], selected[:, 2]
            ]

        return {
            "m_v3d": m_v3d,
            "match_dict": {
                "matches": m_v3d,
                "b_sel": b_sel,
                "yx_sel": yx_sel,
            },
            "mask": pred_mask.float(),
            "pred_mask": pred_mask.float(),
            "b_sel": b_sel,
            "yx_sel": yx_sel,
            "logits": logits,
            "mask_logits": mask_logits,
            "feat_hw": (H, W),
        }

def healpix_cube_mesh(subdivisions: int, round_decimals: int = 14):
    n = subdivisions
    u = np.linspace(0, 1, n)
    v = np.linspace(0, 1, n)
    uu, vv = np.meshgrid(u, v)
    uu = uu - 0.5
    vv = vv - 0.5

    verts = []
    vindex = {}

    def vid_of(vertex):
        key = tuple(np.round(vertex, round_decimals))
        if key in vindex:
            return vindex[key]
        k = len(verts)
        vindex[key] = k
        verts.append(vertex)
        return k

    faces = []
    face_defs = [
        (2, +0.5, 0, 1),
        (2, -0.5, 0, 1),
        (1, +0.5, 0, 2),
        (1, -0.5, 0, 2),
        (0, +0.5, 1, 2),
        (0, -0.5, 1, 2),
    ]
    for const_axis, const_val, u_axis, v_axis in face_defs:
        face_vids = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(n):
                vertex = np.zeros(3)
                vertex[const_axis] = const_val
                vertex[u_axis] = uu[i, j]
                vertex[v_axis] = vv[i, j]
                face_vids[i, j] = vid_of(vertex)

        for i in range(n - 1):
            for j in range(n - 1):
                v00 = face_vids[i, j]
                v01 = face_vids[i, j + 1]
                v10 = face_vids[i + 1, j]
                v11 = face_vids[i + 1, j + 1]
                faces.append([v00, v10, v11])
                faces.append([v00, v11, v01])

    verts = np.asarray(verts, dtype=float)
    faces = np.asarray(faces, dtype=int)

    v0 = verts[faces[:, 1]] - verts[faces[:, 0]]
    v1 = verts[faces[:, 2]] - verts[faces[:, 0]]
    n = np.cross(v0, v1)
    centroids = verts[faces].mean(axis=1)
    inward = np.einsum("ij,ij->i", n, centroids) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]

    bbox = np.max(verts, axis=0) - np.min(verts, axis=0)
    verts *= 1 / np.linalg.norm(bbox)

    return torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.float32))