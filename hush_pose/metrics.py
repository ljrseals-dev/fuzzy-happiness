"""PCK and related metrics."""
from __future__ import annotations
import torch
from common import IMG, HM


@torch.no_grad()
def decode_argmax(hm: torch.Tensor) -> torch.Tensor:
    """hm [B,J,H,W] -> xy in 256-input coords [B,J,2]."""
    B, J, H, W = hm.shape
    idx = hm.flatten(2).argmax(-1)
    xy = torch.stack([idx % W, idx // W], -1).float()
    return (xy + 0.5) * IMG / HM


@torch.no_grad()
def pck(pred_xy: torch.Tensor, gt_xy: torch.Tensor, frac: float = 0.1):
    """PCK with threshold = frac * torso length (neck-pelvis). Only GT joints inside crop count."""
    torso = (gt_xy[:, 2] - gt_xy[:, 4]).norm(dim=-1).clamp(min=8.0)[:, None]
    inside = ((gt_xy >= 0) & (gt_xy < IMG)).all(-1)
    ok = ((pred_xy - gt_xy).norm(dim=-1) < frac * torso) & inside
    return ok.sum().item(), inside.sum().item()


@torch.no_grad()
def eval_pck(model, loader, device=None) -> float:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    hits = total = 0
    for x, _, j in loader:
        x = x.to(device)
        use_amp = device.type == "cuda"
        with torch.autocast(device.type, dtype=torch.bfloat16 if use_amp else torch.float32, enabled=use_amp):
            p = model(x)
        a, b = pck(decode_argmax(p.float()).cpu(), j)
        hits += a
        total += b
    return hits / max(1, total)
