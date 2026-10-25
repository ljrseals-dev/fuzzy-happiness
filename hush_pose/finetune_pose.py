"""Fine-tune / distill for 2D heatmap regression.
  baseline : StudentPose from scratch, no pretraining, no teacher   (benchmark config a)
  teacher  : ViT encoder from MAE ckpt + deconv head                 (config b)
  student  : StudentPose distilled from teacher                      (config c)
Examples:
  python finetune_pose.py --mode teacher  --pretrained ckpt/mae/encoder_last.pt --out ckpt/teacher.pt
  python finetune_pose.py --mode student  --teacher ckpt/teacher.pt --out ckpt/student.pt
  python finetune_pose.py --mode baseline --out ckpt/baseline.pt"""
import argparse
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data import SyntheticPoseDataset
from models import StudentPose, TeacherPose, DepthViT
from metrics import eval_pck


def kd_loss(s, t, T, beta=20.0):
    """KL between spatial softmaxes of student/teacher heatmaps (temperature T), per joint."""
    ls = F.log_softmax(s.flatten(2) * beta / T, -1)
    pt = F.softmax(t.flatten(2) * beta / T, -1)
    return F.kl_div(ls, pt, reduction="batchmean") * T * T / s.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "teacher", "student"], required=True)
    ap.add_argument("--pretrained", default=None)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-train", type=int, default=50000)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.mode == "student" and not a.teacher:
        ap.error("--teacher is required for --mode student")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    tr = DataLoader(SyntheticPoseDataset(a.n_train, seed=0), a.bs, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    va = DataLoader(SyntheticPoseDataset(2000, seed=1), 64, num_workers=4)   # held-out seed

    if a.mode == "teacher":
        enc = DepthViT()
        if a.pretrained:
            enc.load_state_dict(torch.load(a.pretrained))
        model = TeacherPose(enc).cuda()
        groups = [{"params": model.enc.parameters(), "lr": a.lr * (0.1 if a.pretrained else 1)},
                  {"params": model.head.parameters(), "lr": a.lr}]
    else:
        model = StudentPose().cuda()
        groups = [{"params": model.parameters(), "lr": a.lr}]
    teacher = None
    if a.mode == "student":
        teacher = TeacherPose().cuda()
        teacher.load_state_dict(torch.load(a.teacher)["model"])
        teacher.eval()

    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs * len(tr))
    for ep in range(a.epochs):
        model.train()
        for i, (x, hm, _) in enumerate(tr):
            x, hm = x.cuda(non_blocking=True), hm.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model(x).float()
                loss = ((pred - hm) ** 2 * (1 + 10 * hm)).mean()          # foreground-weighted MSE
                if teacher is not None:
                    with torch.no_grad():
                        t = teacher(x).float()
                    loss_kd = kd_loss(pred, t, a.temp) + F.mse_loss(pred, t)
                    loss = (1 - a.alpha) * loss + a.alpha * loss_kd
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if i % 100 == 0:
                print(f"ep {ep} it {i} loss {loss.item():.5f}", flush=True)
        print(f"== epoch {ep} PCK@0.1torso (synthetic val): {eval_pck(model, va):.4f}", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep}, a.out)


if __name__ == "__main__":
    main()
