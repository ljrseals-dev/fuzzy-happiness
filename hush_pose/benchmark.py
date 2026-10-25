"""Compares (a) baseline CNN, (b) MAE-pretrained teacher, (c) distilled student on held-out synthetic data.
  python benchmark.py --baseline ckpt/baseline.pt --teacher ckpt/teacher.pt --student ckpt/student.pt"""
import argparse
import time
import torch
from torch.utils.data import DataLoader
from data import SyntheticPoseDataset
from models import StudentPose, TeacherPose
from metrics import eval_pck


def latency(model, iters=200):
    x = torch.rand(1, 1, 256, 256).cuda()
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for _ in range(30):
            model(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            model(x)
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / iters * 1e3
    return ms, 1e3 / ms


ap = argparse.ArgumentParser()
for k in ("baseline", "teacher", "student"):
    ap.add_argument(f"--{k}", required=True)
a = ap.parse_args()
val = DataLoader(SyntheticPoseDataset(2000, seed=1), 64, num_workers=4)       # seed 1 = held out
rows = []
for name, cls, path in [("(a) baseline CNN", StudentPose, a.baseline),
                        ("(b) MAE-pretrained teacher", TeacherPose, a.teacher),
                        ("(c) distilled student", StudentPose, a.student)]:
    m = cls().cuda()
    m.load_state_dict(torch.load(path)["model"])
    ms, fps = latency(m)
    rows.append((name, eval_pck(m, val), ms, fps, sum(p.numel() for p in m.parameters()) / 1e6))
print("| Config | PCK@0.1 torso | Latency (ms, PyTorch fp16) | FPS | Params (M) |\n|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]:.3f} | {r[2]:.2f} | {r[3]:.0f} | {r[4]:.2f} |")
# For TensorRT numbers use pipeline.py (end-to-end) or:
#   trtexec --loadEngine=student_fp16.plan --shapes=depth:1x1x256x256
