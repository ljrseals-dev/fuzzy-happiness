"""Synthetic depth + labels from a randomized 17-joint stick body (cv2 primitives).
For production, replace with Blender/SMPL renders of mocap (BVH/AMASS) producing the
same (depth_m, joints_xyz, joints_uv) triple."""
import cv2
import numpy as np
from common import PARENT, KINECT_V2

REST = {3: (0, -.25, 0), 2: (0, -.25, 0), 1: (0, -.10, 0), 0: (0, -.12, 0),
        5: (-.18, 0, 0), 6: (-.03, .28, 0), 7: (-.03, .25, 0),
        8: (.18, 0, 0), 9: (.03, .28, 0), 10: (.03, .25, 0),
        11: (-.10, 0, 0), 12: (0, .42, 0), 13: (0, .42, 0),
        14: (.10, 0, 0), 15: (0, .42, 0), 16: (0, .42, 0)}   # offsets from parent (m)
FLEX = {3: .05, 2: .05, 1: .15, 0: .1, 5: 0, 6: .7, 7: .7, 8: 0, 9: .7, 10: .7,
        11: 0, 12: .5, 13: .4, 14: 0, 15: .5, 16: .4}         # direction noise per joint
ORDER = [4, 3, 2, 1, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


def sample_pose(rng, intr=KINECT_V2):
    J = np.zeros((17, 3))
    scale = rng.uniform(.85, 1.15)
    for j in ORDER[1:]:
        off = np.array(REST[j])
        L = np.linalg.norm(off)
        d = off / L + rng.normal(0, FLEX[j], 3)
        d /= np.linalg.norm(d)
        J[j] = J[PARENT[j]] + d * L * scale
    th = rng.uniform(-1.0, 1.0)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    J = J @ R.T
    z = rng.uniform(1.5, 3.5)
    u, v = rng.uniform(150, 360), rng.uniform(180, 280)
    return J + np.array([(u - intr.cx) * z / intr.fx, (v - intr.cy) * z / intr.fy, z]), z


def render(J, z_root, intr, rng):
    h, w = intr.height, intr.width
    img = np.zeros((h, w), np.float32)
    uv = np.stack([intr.fx * J[:, 0] / J[:, 2] + intr.cx,
                   intr.fy * J[:, 1] / J[:, 2] + intr.cy], 1)
    bones = sorted([(j, PARENT[j]) for j in range(17) if PARENT[j] >= 0],
                   key=lambda b: -(J[b[0], 2] + J[b[1], 2]))      # painter: far first
    for a, b in bones:
        z = float((J[a, 2] + J[b, 2]) / 2)
        wide = 3.0 if {a, b} in ({2, 3}, {3, 4}) else 1.0
        th = max(2, int(wide * intr.fx * 0.05 / z))
        cv2.line(img, tuple(int(c) for c in uv[a]), tuple(int(c) for c in uv[b]), z, th)
    zh = float(J[1, 2])
    cv2.circle(img, tuple(int(c) for c in uv[1]), max(3, int(intr.fx * .09 / zh)), zh, -1)
    if rng.random() < .6:                                          # background wall
        wall = z_root + rng.uniform(1.3, 2.0)
        if wall < 4.4:
            img[img == 0] = wall
    img += (img > 0) * rng.normal(0, 0.004, img.shape).astype(np.float32) * img
    img[rng.random(img.shape) < 0.02] = 0                          # dropouts
    return img, uv


def sample_frame(rng, intr=KINECT_V2):
    """Returns (depth_m [H,W], joints_xyz [17,3], joints_uv [17,2] in raw pixels)."""
    J, z = sample_pose(rng, intr)
    depth, uv = render(J, z, intr, rng)
    return depth, J, uv
