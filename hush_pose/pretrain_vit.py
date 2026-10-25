"""MAE pre-training on unlabeled depth. Example:
  python pretrain_vit.py --data /data/kinect_frames --synth 50000 --epochs 100 --bs 96 --out ckpt/mae"""
import argparse
import csv
import math
import os
import time
import torch
from torch.utils.data import DataLoader
from data import DepthPretrainDataset
from models import MAE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--synth", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1.5e-4, help="base LR per 256 samples")
    ap.add_argument("--mask", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="ckpt/mae")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dl = DataLoader(DepthPretrainDataset(a.data, a.synth), a.bs, shuffle=True, num_workers=a.workers,
                    drop_last=True, pin_memory=True, persistent_workers=True)
    model = MAE().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr * a.bs / 256, betas=(.9, .95), weight_decay=.05)
    total, warm = a.epochs * len(dl), a.warmup * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm
        else .5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ep0 = 0
    if a.resume:
        ck = torch.load(a.resume)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        ep0 = ck["epoch"] + 1
    logf = open(os.path.join(a.out, "log.csv"), "a", newline="")
    log = csv.writer(logf)
    for ep in range(ep0, a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, x in enumerate(dl):
            x = x.cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, a.mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            if i % 50 == 0:
                print(f"ep {ep} it {i}/{len(dl)} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        log.writerow([ep, run / len(dl), time.time() - t0])
        logf.flush()
        ck = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep}
        torch.save(ck, os.path.join(a.out, "last.pt"))
        torch.save(model.enc.state_dict(), os.path.join(a.out, "encoder_last.pt"))   # used by fine-tuning
        if ep % 10 == 9:
            torch.save(ck, os.path.join(a.out, f"epoch{ep + 1}.pt"))


if __name__ == "__main__":
    main()
