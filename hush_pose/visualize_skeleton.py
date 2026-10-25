#!/usr/bin/env python3
"""Overlay a skeleton on a depth image.

  python visualize_skeleton.py \\
      --depth  /tmp/smpl_test/000000_depth.npy \\
      --skeleton skeleton.json \\
      --meta   /tmp/smpl_test/meta.json \\
      --out    overlay.png

Skeleton JSON may come from postprocess.decode_skeleton / pipeline output.
Prefer per-joint "uv" when present; otherwise project from "xyz" using meta intrinsics.
Also accepts --xyz / --uv numpy arrays for ground-truth visualisation.
"""
from __future__ import annotations
import argparse
import json
import sys
import cv2
import numpy as np
from common import JOINT_NAMES, PARENT, Intrinsics


def load_intrinsics(meta_path: str) -> Intrinsics:
    meta = json.loads(Path_read(meta_path))
    k = meta["intrinsics"]
    return Intrinsics(k["fx"], k["fy"], k["cx"], k["cy"], k["width"], k["height"])


def Path_read(p: str) -> str:
    from pathlib import Path
    return Path(p).read_text()


def depth_to_color(depth_m: np.ndarray) -> np.ndarray:
    d = np.clip(depth_m / 5.0, 0, 1)
    gray = (d * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)


def joints_from_skeleton_json(path: str, intr: Intrinsics):
    """Return list of (u, v, conf) preferring UV, else projecting XYZ."""
    sk = json.loads(Path_read(path))
    out = []
    for j in sk["joints"]:
        conf = float(j.get("conf", 1.0))
        if j.get("uv") is not None:
            u, v = j["uv"]
            out.append((float(u), float(v), conf))
        elif j.get("xyz") is not None:
            x, y, z = j["xyz"]
            if z is None or z <= 0:
                out.append((None, None, 0.0))
            else:
                u = intr.fx * x / z + intr.cx
                v = intr.fy * y / z + intr.cy
                out.append((float(u), float(v), conf))
        else:
            out.append((None, None, 0.0))
    return out


def draw_skeleton(img: np.ndarray, joints_uvc: list, scale: float = 1.0):
    """joints_uvc: list of (u, v, conf) in the coordinate system of img (already scaled)."""
    h, w = img.shape[:2]
    pts = {}
    for i, (u, v, c) in enumerate(joints_uvc):
        if u is None or v is None or c < 0.1:
            continue
        u_s, v_s = u * scale, v * scale
        if 0 <= u_s < w and 0 <= v_s < h:
            pts[i] = (int(round(u_s)), int(round(v_s)))
            color = (0, int(255 * min(1.0, c)), 0)
            cv2.circle(img, pts[i], 5, color, -1, lineType=cv2.LINE_AA)
            cv2.putText(img, JOINT_NAMES[i][:4], (pts[i][0] + 4, pts[i][1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    for j, p in enumerate(PARENT):
        if p >= 0 and j in pts and p in pts:
            cv2.line(img, pts[j], pts[p], (0, 200, 255), 2, lineType=cv2.LINE_AA)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", required=True, help=".npy depth in meters")
    ap.add_argument("--meta", required=True, help="meta.json with intrinsics (from generate_synthetic_data.py)")
    ap.add_argument("--skeleton", default=None, help="skeleton JSON from pipeline / decode_skeleton")
    ap.add_argument("--xyz", default=None, help="optional GT joints_xyz .npy [17,3]")
    ap.add_argument("--uv", default=None, help="optional GT joints_uv .npy [17,2]")
    ap.add_argument("--out", default="skeleton_vis.png")
    ap.add_argument("--scale", type=float, default=2.0,
                    help="Upscale factor for the output image (default 2x, INTER_NEAREST)")
    a = ap.parse_args()

    depth = np.load(a.depth).astype(np.float32)
    intr = load_intrinsics(a.meta)
    img = depth_to_color(depth)

    joints_uvc = None
    if a.skeleton:
        joints_uvc = joints_from_skeleton_json(a.skeleton, intr)
    elif a.uv is not None:
        uv = np.load(a.uv)
        joints_uvc = [(float(u), float(v), 1.0) for u, v in uv]
    elif a.xyz is not None:
        xyz = np.load(a.xyz)
        joints_uvc = []
        for x, y, z in xyz:
            if z <= 0:
                joints_uvc.append((None, None, 0.0))
            else:
                u = intr.fx * x / z + intr.cx
                v = intr.fy * y / z + intr.cy
                joints_uvc.append((float(u), float(v), 1.0))
    else:
        print("No --skeleton / --xyz / --uv given; saving colored depth only", file=sys.stderr)

    if joints_uvc is not None:
        # Upscale image and UV coordinates together for aspect-correct display
        scale = a.scale
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        img = draw_skeleton(img, joints_uvc, scale=scale)
        print(f"drew {sum(1 for u,v,c in joints_uvc if u is not None and c >= 0.1)} joints")
    else:
        if a.scale != 1.0:
            img = cv2.resize(img, None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(a.out, img)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
