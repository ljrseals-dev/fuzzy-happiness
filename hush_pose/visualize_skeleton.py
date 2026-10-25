#!/usr/bin/env python3
"""Visualize skeleton overlaid on a depth image.

Examples:
  # From a recorded synthetic frame
  python visualize_skeleton.py --depth data/synth/000000_depth.npy --xyz data/synth/000000_xyz.npy --out vis.png

  # Run model on a depth frame (requires checkpoint)
  python visualize_skeleton.py --depth some_depth.npy --ckpt ckpt/student.pt --out vis.png
"""
from __future__ import annotations
import argparse
import cv2
import numpy as np
from common import JOINT_NAMES, PARENT, KINECT_V2, to_crop, make_heatmaps
from postprocess import decode_skeleton
from preprocess import preprocess_depth


def depth_to_color(depth_m: np.ndarray) -> np.ndarray:
    d = np.clip(depth_m / 5.0, 0, 1)
    gray = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def draw_skeleton(img: np.ndarray, joints_uv: list | dict, conf: list | None = None):
    """joints_uv: list of (u,v) or list of dicts with 'uv'."""
    h, w = img.shape[:2]
    pts = {}
    for i, name in enumerate(JOINT_NAMES):
        if isinstance(joints_uv[0], dict):
            j = joints_uv[i]
            if j.get("xyz") is None and "uv" not in j:
                continue
            if "uv" in j:
                u, v = j["uv"]
            else:
                # re-project from xyz
                x, y, z = j["xyz"]
                u = KINECT_V2.fx * x / z + KINECT_V2.cx
                v = KINECT_V2.fy * y / z + KINECT_V2.cy
            c = j.get("conf", 1.0)
        else:
            u, v = joints_uv[i]
            c = conf[i] if conf is not None else 1.0
        if 0 <= u < w and 0 <= v < h and c > 0.1:
            pts[i] = (int(round(u)), int(round(v)))
            color = (0, int(255 * min(1, c)), 0)
            cv2.circle(img, pts[i], 5, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(img, name[:4], (pts[i][0] + 4, pts[i][1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    for j, p in enumerate(PARENT):
        if p >= 0 and j in pts and p in pts:
            cv2.line(img, pts[j], pts[p], (0, 200, 255), 2, lineType=cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", required=True, help=".npy depth in meters")
    ap.add_argument("--xyz", default=None, help="optional ground-truth xyz")
    ap.add_argument("--ckpt", default=None, help="optional model checkpoint")
    ap.add_argument("--out", default="skeleton_vis.png")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    depth = np.load(a.depth).astype(np.float32)
    img = depth_to_color(depth)

    if a.xyz is not None:
        xyz = np.load(a.xyz)
        # project
        uv = np.stack([
            KINECT_V2.fx * xyz[:, 0] / xyz[:, 2] + KINECT_V2.cx,
            KINECT_V2.fy * xyz[:, 1] / xyz[:, 2] + KINECT_V2.cy,
        ], 1)
        img = draw_skeleton(img, uv)
        print("drew ground-truth skeleton")
    elif a.ckpt is not None:
        import torch
        from models import StudentPose
        m = StudentPose()
        ck = torch.load(a.ckpt, map_location="cpu")
        m.load_state_dict(ck["model"] if "model" in ck else ck)
        m.eval()
        x = torch.from_numpy(preprocess_depth(depth)).unsqueeze(0)
        with torch.no_grad():
            hm = m(x).numpy()[0]
        sk = decode_skeleton(hm, depth, KINECT_V2)
        img = draw_skeleton(img, sk["joints"])
        print("drew predicted skeleton")
    else:
        print("No --xyz or --ckpt given; saving colored depth only")

    cv2.imwrite(a.out, img)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
