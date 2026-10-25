#!/usr/bin/env python3
"""Dump fixed heatmap/depth fixtures + Python decode_skeleton reference for C++ parity_test.

Usage:
  python cpp/tests/dump_reference.py --out /tmp/fix1_fixtures
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Allow running from repo root or cpp/tests/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "hush_pose"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1].parent / "hush_pose"))

from common import KINECT_V2, NUM_JOINTS, HM, IMG, make_heatmaps, to_crop
from postprocess import decode_skeleton
from synth import sample_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5, help="number of fixed cases")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)  # fixed seed so C++ side can reproduce if needed
    cases = []
    for i in range(a.n):
        depth, xyz, uv = sample_frame(rng)
        j256 = to_crop(uv, *depth.shape)
        hm = make_heatmaps(j256)  # [17,64,64]
        sk = decode_skeleton(hm, depth, KINECT_V2, frame_id=i)

        # Expected joints as [17, 4] = x,y,z,valid (valid as 0/1 float)
        expected = np.zeros((NUM_JOINTS, 4), np.float32)
        for j, jt in enumerate(sk["joints"]):
            if jt["xyz"] is not None:
                expected[j, 0] = jt["xyz"][0]
                expected[j, 1] = jt["xyz"][1]
                expected[j, 2] = jt["xyz"][2]
                expected[j, 3] = 1.0
            # conf is not asserted in parity_test (only xyz + valid)

        np.save(out / f"case{i:02d}_heatmaps.npy", hm.astype(np.float32))
        np.save(out / f"case{i:02d}_depth.npy", depth.astype(np.float32))
        np.save(out / f"case{i:02d}_expected.npy", expected)
        cases.append(i)

    meta = {
        "n": a.n,
        "intrinsics": {
            "fx": KINECT_V2.fx, "fy": KINECT_V2.fy,
            "cx": KINECT_V2.cx, "cy": KINECT_V2.cy,
            "width": KINECT_V2.width, "height": KINECT_V2.height,
        },
        "num_joints": NUM_JOINTS,
        "hm": HM,
        "img": IMG,
        "cases": cases,
    }
    (out / "K.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {a.n} cases to {out}")


if __name__ == "__main__":
    main()
