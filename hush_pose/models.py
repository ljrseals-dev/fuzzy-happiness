"""Models: DepthViT encoder, MAE, TeacherPose (ViT), StudentPose (separable-conv FPN).

All models accept single-channel depth [B,1,256,256] normalized to [0,1]
(near=1, far=0, invalid=0) and Teacher/Student emit heatmaps [B,17,64,64].
"""
from __future__ import annotations
import torch
import torch.nn as nn
from common import NUM_JOINTS


def _gather(t: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    """Gather tokens along sequence dimension."""
    return torch.gather(t, 1, ids[..., None].expand(-1, -1, t.shape[-1]))


class DepthViT(nn.Module):
    """ViT-S/16 for 1-channel depth. 256x256 -> 16x16 = 256 tokens."""

    def __init__(self, img: int = 256, patch: int = 16, dim: int = 384,
                 depth: int = 12, heads: int = 6):
        super().__init__()
        self.patch = patch
        self.g = img // patch
        self.dim = dim
        self.proj = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.g ** 2, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=heads, dim_feedforward=dim * 4,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2) + self.pos

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            tokens = blk(tokens)
        return self.norm(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(self.embed(x))


class MAE(nn.Module):
    """Masked Autoencoder for depth. Returns scalar reconstruction loss."""

    def __init__(self, enc_dim: int = 384, dec_dim: int = 192,
                 dec_depth: int = 4, dec_heads: int = 6):
        super().__init__()
        self.enc = DepthViT(dim=enc_dim)
        N = self.enc.g ** 2
        p = self.enc.patch
        self.dec_in = nn.Linear(enc_dim, dec_dim)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, N, dec_dim))
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        self.dec = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dec_dim, nhead=dec_heads, dim_feedforward=dec_dim * 4,
                dropout=0.0, activation="gelu", batch_first=True, norm_first=True
            ) for _ in range(dec_depth)
        ])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, p * p)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        B, g, p = x.shape[0], self.enc.g, self.enc.patch
        return (x.reshape(B, 1, g, p, g, p)
                .permute(0, 2, 4, 3, 5, 1)
                .reshape(B, g * g, p * p))

    def forward(self, x: torch.Tensor, mask_ratio: float = 0.75) -> torch.Tensor:
        tok = self.enc.embed(x)
        B, N, _ = tok.shape
        keep = int(N * (1.0 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        ids_keep = ids_shuffle[:, :keep]
        z = self.enc.encode(_gather(tok, ids_keep))
        z = self.dec_in(z)
        mask_tokens = self.mask_tok.expand(B, N - keep, -1)
        z_ = torch.cat([z, mask_tokens], dim=1)
        z_ = _gather(z_, ids_restore) + self.dec_pos
        for blk in self.dec:
            z_ = blk(z_)
        pred = self.pred(self.dec_norm(z_))
        tgt = self.patchify(x)
        w = 1.0 + 4.0 * (tgt.abs().amax(dim=-1) > 0).float()
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0.0
        mask = _gather(mask.unsqueeze(-1), ids_restore).squeeze(-1)
        loss = ((pred - tgt) ** 2).mean(dim=-1) * w * mask
        return loss.sum() / mask.sum().clamp(min=1.0)


class TeacherPose(nn.Module):
    """Pretrained ViT encoder + deconvolutional heatmap head."""

    def __init__(self, encoder: DepthViT | None = None):
        super().__init__()
        self.enc = encoder or DepthViT()
        D = self.enc.dim
        self.head = nn.Sequential(
            nn.ConvTranspose2d(D, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, NUM_JOINTS, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.enc(x)
        B, N, D = t.shape
        g = self.enc.g
        feat = t.transpose(1, 2).reshape(B, D, g, g)
        return self.head(feat)


def _dw(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch), nn.ReLU6(True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU6(True),
    )


class StudentPose(nn.Module):
    """Lightweight separable-conv encoder + 2-level FPN. ~0.5 GMACs @ 256x256."""

    def __init__(self, width_mult: float = 1.0):
        super().__init__()
        c = [int(v * width_mult) for v in (32, 64, 128, 256)]
        self.stem = nn.Sequential(
            nn.Conv2d(1, c[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(c[0]), nn.ReLU6(True),
        )
        self.s1 = nn.Sequential(_dw(c[0], c[1], 2), _dw(c[1], c[1]))
        self.s2 = nn.Sequential(_dw(c[1], c[2], 2), _dw(c[2], c[2]), _dw(c[2], c[2]))
        self.s3 = nn.Sequential(
            _dw(c[2], c[3], 2),
            *[_dw(c[3], c[3]) for _ in range(4)],
        )
        self.l1 = nn.Conv2d(c[1], 128, 1)
        self.l2 = nn.Conv2d(c[2], 128, 1)
        self.l3 = nn.Conv2d(c[3], 128, 1)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.smooth = nn.Sequential(_dw(128, 128), _dw(128, 128))
        self.out = nn.Conv2d(128, NUM_JOINTS, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1 = self.s1(self.stem(x))
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        p = self.up(self.l3(f3)) + self.l2(f2)
        p = self.up(p) + self.l1(f1)
        return self.out(self.smooth(p))
