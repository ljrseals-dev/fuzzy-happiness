"""Models: DepthViT encoder, MAE, TeacherPose (ViT), StudentPose (separable-conv FPN)."""
import torch
import torch.nn as nn
from common import NUM_JOINTS


def _gather(t, ids):
    return torch.gather(t, 1, ids[..., None].expand(-1, -1, t.shape[-1]))


class DepthViT(nn.Module):
    """ViT-S/16 for 1-channel depth. 256x256 -> 16x16=256 tokens."""
    def __init__(self, img=256, patch=16, dim=384, depth=12, heads=6):
        super().__init__()
        self.patch, self.g, self.dim = patch, img // patch, dim
        self.proj = nn.Conv2d(1, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, self.g ** 2, dim))
        nn.init.trunc_normal_(self.pos, std=.02)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, dim * 4, 0.0, "gelu",
                                     batch_first=True, norm_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def embed(self, x):
        return self.proj(x).flatten(2).transpose(1, 2) + self.pos

    def encode(self, t):
        for b in self.blocks:
            t = b(t)
        return self.norm(t)

    def forward(self, x):
        return self.encode(self.embed(x))


class MAE(nn.Module):
    def __init__(self, enc_dim=384, dec_dim=192, dec_depth=4, dec_heads=6):
        super().__init__()
        self.enc = DepthViT(dim=enc_dim)
        N, p = self.enc.g ** 2, self.enc.patch
        self.dec_in = nn.Linear(enc_dim, dec_dim)
        self.mask_tok = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_pos = nn.Parameter(torch.zeros(1, N, dec_dim))
        nn.init.trunc_normal_(self.dec_pos, std=.02)
        self.dec = nn.ModuleList([nn.TransformerEncoderLayer(dec_dim, dec_heads, dec_dim * 4, 0.0, "gelu",
                                  batch_first=True, norm_first=True) for _ in range(dec_depth)])
        self.dec_norm, self.pred = nn.LayerNorm(dec_dim), nn.Linear(dec_dim, p * p)

    def patchify(self, x):
        B, g, p = x.shape[0], self.enc.g, self.enc.patch
        return x.reshape(B, 1, g, p, g, p).permute(0, 2, 4, 3, 5, 1).reshape(B, g * g, p * p)

    def forward(self, x, mask_ratio=0.75):
        tok = self.enc.embed(x)
        B, N, _ = tok.shape
        keep = int(N * (1 - mask_ratio))
        shuf = torch.rand(B, N, device=x.device).argsort(1)
        restore = shuf.argsort(1)
        z = self.enc.encode(_gather(tok, shuf[:, :keep]))                  # visible tokens only
        y = torch.cat([self.dec_in(z), self.mask_tok.expand(B, N - keep, -1)], 1)
        y = _gather(y, restore) + self.dec_pos
        for b in self.dec:
            y = b(y)
        pred = self.pred(self.dec_norm(y))
        tgt = self.patchify(x)
        w = 1 + 4 * (tgt.abs().amax(-1) > 0).float()                       # upweight foreground patches
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep] = 0
        mask = _gather(mask[..., None], restore)[..., 0]
        return (((pred - tgt) ** 2).mean(-1) * w * mask).sum() / mask.sum()


class TeacherPose(nn.Module):
    """Pretrained ViT encoder (generative top discarded) + deconv heatmap head."""
    def __init__(self, encoder=None):
        super().__init__()
        self.enc = encoder or DepthViT()
        D = self.enc.dim
        self.head = nn.Sequential(
            nn.ConvTranspose2d(D, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, NUM_JOINTS, 1))                                  # 16x16 -> 64x64

    def forward(self, x):
        t = self.enc(x)
        B, N, D = t.shape
        return self.head(t.transpose(1, 2).reshape(B, D, self.enc.g, self.enc.g))


def _dw(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, i, 3, s, 1, groups=i, bias=False), nn.BatchNorm2d(i), nn.ReLU6(True),
                         nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU6(True))


class StudentPose(nn.Module):
    """Separable-conv encoder + 2-level FPN to stride 4. ~0.5 GMACs at 256x256."""
    def __init__(self, w=1.0):
        super().__init__()
        c = [int(v * w) for v in (32, 64, 128, 256)]
        self.stem = nn.Sequential(nn.Conv2d(1, c[0], 3, 2, 1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU6(True))
        self.s1 = nn.Sequential(_dw(c[0], c[1], 2), _dw(c[1], c[1]))                       # 64x64
        self.s2 = nn.Sequential(_dw(c[1], c[2], 2), _dw(c[2], c[2]), _dw(c[2], c[2]))      # 32x32
        self.s3 = nn.Sequential(_dw(c[2], c[3], 2), *[_dw(c[3], c[3]) for _ in range(4)])  # 16x16
        self.l1, self.l2, self.l3 = (nn.Conv2d(c[1], 128, 1), nn.Conv2d(c[2], 128, 1), nn.Conv2d(c[3], 128, 1))
        self.up = nn.Upsample(scale_factor=2, mode="nearest")                              # TRT-friendly
        self.smooth = nn.Sequential(_dw(128, 128), _dw(128, 128))
        self.out = nn.Conv2d(128, NUM_JOINTS, 1)

    def forward(self, x):
        f1 = self.s1(self.stem(x))
        f2 = self.s2(f1)
        f3 = self.s3(f2)
        p = self.up(self.l3(f3)) + self.l2(f2)
        p = self.up(p) + self.l1(f1)
        return self.out(self.smooth(p))                                                    # [B,17,64,64]
