"""Fine-tune / distill for 2D heatmap regression.

Modes:
  baseline : StudentPose from scratch
  teacher  : ViT encoder from MAE + deconv head
  student  : StudentPose distilled from teacher

Examples:
  python finetune_pose.py --mode teacher  --pretrained ckpt/mae/encoder_last.pt --out ckpt/teacher.pt
  python finetune_pose.py --mode student  --teacher ckpt/teacher.pt --out ckpt/student.pt
  python finetune_pose.py --mode baseline --out ckpt/baseline.pt
"""
from __future__ import annotations
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import SyntheticPoseDataset
from models import StudentPose, TeacherPose, DepthViT
from metrics import eval_pck


def kd_loss(student_hm: torch.Tensor, teacher_hm: torch.Tensor,
            T: float = 2.0, beta: float = 20.0) -> torch.Tensor:
    """Temperature-scaled spatial KL + MSE between heatmaps."""
    B, J, H, W = student_hm.shape
    s = student_hm.flatten(2) * beta / T
    t = teacher_hm.flatten(2) * beta / T
    ls = F.log_softmax(s, dim=-1)
    pt = F.softmax(t, dim=-1)
    kl = F.kl_div(ls, pt, reduction="batchmean") * (T * T) / J
    mse = F.mse_loss(student_hm, teacher_hm)
    return kl + 0.5 * mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "teacher", "student"], required=True)
    ap.add_argument("--pretrained", default=None, help="MAE encoder weights for teacher")
    ap.add_argument("--teacher", default=None, help="Teacher checkpoint for distillation")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Weight on hard (GT) loss; (1-alpha) on soft (teacher) loss")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true", help="Tiny run for CI / syntax check")
    a = ap.parse_args()

    if a.mode == "student" and not a.teacher:
        ap.error("--teacher is required for --mode student")
    if a.smoke:
        a.n_train = 128
        a.epochs = 1
        a.bs = 8
        a.workers = 0

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    device = torch.device(a.device)

    tr = DataLoader(
        SyntheticPoseDataset(a.n_train, seed=0, augment=not a.smoke),
        a.bs, shuffle=True, num_workers=a.workers,
        drop_last=True, pin_memory=(device.type == "cuda"),
        persistent_workers=(a.workers > 0),
    )
    va = DataLoader(SyntheticPoseDataset(512 if a.smoke else 2000, seed=1, augment=False),
                    min(64, a.bs), num_workers=min(2, a.workers))

    if a.mode == "teacher":
        enc = DepthViT()
        if a.pretrained:
            enc.load_state_dict(torch.load(a.pretrained, map_location="cpu"))
        model = TeacherPose(enc).to(device)
        groups = [
            {"params": model.enc.parameters(), "lr": a.lr * (0.1 if a.pretrained else 1.0)},
            {"params": model.head.parameters(), "lr": a.lr},
        ]
    else:
        model = StudentPose().to(device)
        groups = [{"params": model.parameters(), "lr": a.lr}]

    teacher = None
    if a.mode == "student":
        teacher = TeacherPose().to(device)
        ck = torch.load(a.teacher, map_location="cpu")
        teacher.load_state_dict(ck["model"] if "model" in ck else ck)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.epochs * len(tr)))

    for ep in range(a.epochs):
        model.train()
        running = 0.0
        for i, (x, hm, _) in enumerate(tr):
            x = x.to(device, non_blocking=True)
            hm = hm.to(device, non_blocking=True)
            use_amp = device.type == "cuda"
            with torch.autocast(device.type, dtype=torch.bfloat16 if use_amp else torch.float32, enabled=use_amp):
                pred = model(x).float()
                hard = ((pred - hm) ** 2 * (1.0 + 10.0 * hm)).mean()
                if teacher is not None:
                    with torch.no_grad():
                        soft = teacher(x).float()
                    soft_l = kd_loss(pred, soft, a.temp)
                    # alpha * hard + (1-alpha) * soft  (master-prompt convention)
                    loss = a.alpha * hard + (1.0 - a.alpha) * soft_l
                else:
                    loss = hard
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            if i % 50 == 0:
                print(f"ep {ep} it {i}/{len(tr)} loss {loss.item():.5f}", flush=True)
        pck = eval_pck(model, va, device=device)
        print(f"== epoch {ep} mean_loss {running/max(1,len(tr)):.5f}  PCK@0.1torso {pck:.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep, "mode": a.mode}, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
