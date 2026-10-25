"""postprocess.py — NMS peaks, back-projection, kinematic sanity check, JSON output."""
import json
import numpy as np
from scipy.ndimage import maximum_filter
from common import JOINT_NAMES, PARENT, IMG, HM, from_crop

MAX_BONE_M = 0.8      # reject joints implausibly far from their parent


def nms_peaks(hm, thr=0.15, topk=1):
    """3x3 non-max suppression per joint + quarter-pixel refinement. Returns [[(x,y,conf),...]*J]."""
    J, H, W = hm.shape
    keep = (hm == maximum_filter(hm, size=(1, 3, 3), mode="constant")) & (hm > thr)
    out = []
    for j in range(J):
        ys, xs = np.nonzero(keep[j])
        sc = hm[j, ys, xs]
        pk = []
        for i in np.argsort(-sc)[:topk]:
            x, y = xs[i], ys[i]
            fx = x + 0.25 * np.sign(hm[j, y, min(x + 1, W - 1)] - hm[j, y, max(x - 1, 0)])
            fy = y + 0.25 * np.sign(hm[j, min(y + 1, H - 1), x] - hm[j, max(y - 1, 0), x])
            pk.append((fx, fy, float(sc[i])))
        out.append(pk)
    return out


def _depth_at(raw, u, v, r=2):
    h, w = raw.shape
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    win = raw[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
    win = win[win > 0]
    return float(np.median(win)) if win.size else None


def decode_skeleton(hm, raw_depth_m, intr, frame_id=0, thr=0.15):
    """hm [17,64,64], raw_depth_m [H,W] meters, intr = Intrinsics of the raw image."""
    h, w = raw_depth_m.shape
    joints = []
    for j, pk in enumerate(nms_peaks(hm, thr)):
        rec = {"name": JOINT_NAMES[j], "xyz": None, "conf": 0.0}
        if pk:
            x, y, c = pk[0]
            u, v = from_crop(np.array([(x + .5) * IMG / HM - .5, (y + .5) * IMG / HM - .5]), h, w)
            z = _depth_at(raw_depth_m, u, v)
            if z:
                rec.update(xyz=[float((u - intr.cx) * z / intr.fx), float((v - intr.cy) * z / intr.fy), z], conf=c)
        joints.append(rec)
    for j, p in enumerate(PARENT):                      # kinematic plausibility check
        if p >= 0 and joints[j]["xyz"] and joints[p]["xyz"]:
            if np.linalg.norm(np.subtract(joints[j]["xyz"], joints[p]["xyz"])) > MAX_BONE_M:
                joints[j]["conf"] *= 0.3
    return {"frame": frame_id, "joints": joints, "bones": [[j, p] for j, p in enumerate(PARENT) if p >= 0]}


def to_json(sk):
    return json.dumps(sk)

# TODO: temporal smoothing (One-Euro filter, https://gery.casiez.net/1euro/) on xyz per joint.
# TODO: multi-person -> use nms_peaks(topk>1) + bottom-up grouping (Part Affinity Fields, OpenPose).
