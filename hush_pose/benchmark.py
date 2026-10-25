"""Compare baseline / teacher / student on held-out synthetic data.

  python benchmark.py --baseline ckpt/baseline.pt --teacher ckpt/teacher.pt --student ckpt/student.pt
"""
from __future__ import annotations
import argparse
import time
import torch
from torch.utils.data import DataLoader
from data import SyntheticPoseDataset
from models import StudentPose, TeacherPose
from metrics import eval_pck


def latency(model, device, iters=100):
    x = torch.rand(1, 1, 256, 256, device=device)
    model.eval()
    use_amp = device.type == "cuda"
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    return ms, 1000.0 / ms


def main():
    ap = argparse.ArgumentParser()
    for k in ("baseline", "teacher", "student"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)

    val = DataLoader(SyntheticPoseDataset(512 if device.type == "cpu" else 2000, seed=1, augment=False),
                     32, num_workers=0)
    rows = []
    for name, cls, path in [
        ("(a) baseline CNN", StudentPose, a.baseline),
        ("(b) MAE-pretrained teacher", TeacherPose, a.teacher),
        ("(c) distilled student", StudentPose, a.student),
    ]:
        m = cls().to(device)
        ck = torch.load(path, map_location="cpu")
        m.load_state_dict(ck["model"] if "model" in ck else ck)
        ms, fps = latency(m, device)
        pck = eval_pck(m, val, device=device)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        rows.append((name, pck, ms, fps, params))

    print("| Config | PCK@0.1 torso | Latency (ms) | FPS | Params (M) |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.2f} | {r[3]:.0f} | {r[4]:.2f} |")
    print("\nFor TensorRT end-to-end numbers use pipeline.py or trtexec.")


if __name__ == "__main__":
    main()
