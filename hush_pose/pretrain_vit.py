"""MAE pre-training on unlabeled depth.

Example:
  python pretrain_vit.py --data /data/frames --synth 50000 --epochs 100 --bs 96 --out ckpt/mae
  python pretrain_vit.py --synth 256 --bs 8 --epochs 1 --smoke --device cpu --out /tmp/mae_smoke
"""
from __future__ import annotations
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
    ap.add_argument("--data", default=None, help="Folder of real depth .npy/.png")
    ap.add_argument("--synth", type=int, default=50000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=96)
    ap.add_argument("--lr", type=float, default=1.5e-4, help="base LR per 256 samples")
    ap.add_argument("--mask", type=float, default=0.75)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--out", default="ckpt/mae")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.smoke:
        a.epochs = 1
        a.synth = 128
        a.bs = 8
        a.workers = 0
        a.warmup = 0

    os.makedirs(a.out, exist_ok=True)
    device = torch.device(a.device)

    dl = DataLoader(
        DepthPretrainDataset(a.data, a.synth),
        a.bs, shuffle=True, num_workers=a.workers,
        drop_last=True, pin_memory=(device.type == "cuda"),
        persistent_workers=(a.workers > 0),
    )
    model = MAE().to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=a.lr * a.bs / 256.0,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )
    total = max(1, a.epochs * len(dl))
    warm = a.warmup * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / max(1, warm) if s < warm
        else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))),
    )
    ep0 = 0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        ep0 = ck["epoch"] + 1

    log_path = os.path.join(a.out, "log.csv")
    logf = open(log_path, "a", newline="")
    log = csv.writer(logf)
    if ep0 == 0:
        log.writerow(["epoch", "loss", "seconds"])

    for ep in range(ep0, a.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for i, x in enumerate(dl):
            x = x.to(device, non_blocking=True)
            use_amp = device.type == "cuda"
            with torch.autocast(device.type, dtype=torch.bfloat16 if use_amp else torch.float32, enabled=use_amp):
                loss = model(x, a.mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run += loss.item()
            if i % 20 == 0 or a.smoke:
                print(f"ep {ep} it {i}/{len(dl)} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        mean_loss = run / max(1, len(dl))
        log.writerow([ep, mean_loss, time.time() - t0])
        logf.flush()
        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "sched": sched.state_dict(), "epoch": ep}
        torch.save(ck, os.path.join(a.out, "last.pt"))
        torch.save(model.enc.state_dict(), os.path.join(a.out, "encoder_last.pt"))
        if ep % 10 == 9:
            torch.save(ck, os.path.join(a.out, f"epoch{ep+1}.pt"))
        print(f"epoch {ep} done  mean_loss={mean_loss:.4f}", flush=True)
    logf.close()
    print("finished, checkpoints in", a.out)


if __name__ == "__main__":
    main()
