import torch
from common import IMG, HM


@torch.no_grad()
def decode_argmax(hm):
    """hm [B,J,H,W] -> xy in 256-input coords [B,J,2]."""
    B, J, H, W = hm.shape
    idx = hm.flatten(2).argmax(-1)
    xy = torch.stack([idx % W, idx // W], -1).float()
    return (xy + 0.5) * IMG / HM


@torch.no_grad()
def pck(pred_xy, gt_xy, frac=0.1):
    """PCK with threshold = frac * torso length (neck(2)-pelvis(4)). Only GT joints inside the crop count."""
    torso = (gt_xy[:, 2] - gt_xy[:, 4]).norm(dim=-1).clamp(min=8)[:, None]
    inside = ((gt_xy >= 0) & (gt_xy < IMG)).all(-1)
    ok = ((pred_xy - gt_xy).norm(dim=-1) < frac * torso) & inside
    return ok.sum().item(), inside.sum().item()


@torch.no_grad()
def eval_pck(model, loader):
    model.eval()
    h = n = 0
    for x, _, j in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(x.cuda())
        a, b = pck(decode_argmax(p.float()).cpu(), j)
        h += a
        n += b
    return h / max(1, n)
