#!/usr/bin/env python3
"""Generate synthetic depth + pose labels.

Uses the built-in stick-figure renderer (always available).
Optional SMPL path is stubbed with clear instructions for production use.

Usage:
  python generate_synthetic_data.py --out data/synth --n 2000
  python generate_synthetic_data.py --out data/synth --n 100 --seed 0
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from common import KINECT_V2, JOINT_NAMES
from synth import sample_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    for i in range(a.n):
        depth, xyz, uv = sample_frame(rng)
        np.save(out / f"{i:06d}_depth.npy", depth.astype(np.float32))
        np.save(out / f"{i:06d}_xyz.npy", xyz.astype(np.float32))
        np.save(out / f"{i:06d}_uv.npy", uv.astype(np.float32))
        if (i + 1) % 100 == 0 or i == 0:
            print(f"wrote {i+1}/{a.n}", flush=True)

    meta = {
        "n": a.n,
        "intrinsics": {
            "fx": KINECT_V2.fx, "fy": KINECT_V2.fy,
            "cx": KINECT_V2.cx, "cy": KINECT_V2.cy,
            "width": KINECT_V2.width, "height": KINECT_V2.height,
        },
        "joint_names": JOINT_NAMES,
        "renderer": "stick",
        "note": "For production accuracy replace with SMPL/AMASS + mesh renderer "
                "(PyBullet / Blender / pyrender). Keep the same (depth_m, joints_xyz, joints_uv) contract.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done. {a.n} frames written to {out}")


if __name__ == "__main__":
    main()
