"""Shared constants and geometry helpers. License: MIT."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

JOINT_NAMES = [
    "head_top", "head", "neck", "spine", "pelvis",
    "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle",
]
NUM_JOINTS = 17
# Kinematic tree (parent index); root = pelvis (4)
PARENT = [1, 2, 3, 4, -1, 2, 5, 6, 2, 8, 9, 4, 11, 12, 4, 14, 15]
IMG, HM = 256, 64            # network input size, heatmap size (stride 4)
NEAR, FAR, BAND = 0.5, 4.5, 1.2   # valid range (m); subject band depth (m)


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


# Approximate Kinect v2 depth intrinsics. Prefer reading real values from the device.
KINECT_V2 = Intrinsics(366.0, 366.0, 256.0, 212.0, 512, 424)


def crop_box(h: int, w: int):
    """Centered square crop (x0, y0, size)."""
    s = min(h, w)
    return (w - s) // 2, (h - s) // 2, s


def to_crop(uv, h: int, w: int) -> np.ndarray:
    """Raw pixel coords -> 256x256 network-input coords."""
    x0, y0, s = crop_box(h, w)
    return (np.asarray(uv, np.float32) - [x0, y0]) * IMG / s


def from_crop(uv, h: int, w: int) -> np.ndarray:
    x0, y0, s = crop_box(h, w)
    return np.asarray(uv, np.float32) * s / IMG + [x0, y0]


def make_heatmaps(j256, sigma: float = 2.0) -> np.ndarray:
    """Gaussian heatmaps [J,HM,HM]. Heatmap pixel i centers on input coord (i+0.5)*IMG/HM."""
    hm = np.zeros((NUM_JOINTS, HM, HM), np.float32)
    xs = np.arange(HM)[None, :]
    ys = np.arange(HM)[:, None]
    for k, (x, y) in enumerate(np.asarray(j256) * HM / IMG - 0.5):
        if -0.5 <= x < HM - 0.5 and -0.5 <= y < HM - 0.5:
            hm[k] = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return hm

# Map SMPL 24-joint order -> our 17-joint order.
# Indices into SMPL joints (see SMPL paper / smplx docs). -1 = no correspondence (synthesized).
# Our order: head_top, head, neck, spine, pelvis,
#   r_shoulder, r_elbow, r_wrist, l_shoulder, l_elbow, l_wrist,
#   r_hip, r_knee, r_ankle, l_hip, l_knee, l_ankle
SMPL24_TO_17 = [
    15,  # head_top  <- SMPL head
    15,  # head
    12,  # neck
    9,   # spine (spine2)
    0,   # pelvis
    17,  # r_shoulder
    19,  # r_elbow
    21,  # r_wrist
    16,  # l_shoulder
    18,  # l_elbow
    20,  # l_wrist
    2,   # r_hip
    5,   # r_knee
    8,   # r_ankle
    1,   # l_hip
    4,   # l_knee
    7,   # l_ankle
]

