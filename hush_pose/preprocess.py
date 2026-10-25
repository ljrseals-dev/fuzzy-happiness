"""Depth preprocessing: range gating, subject-band background suppression,
morphology, center crop, resize, normalize to [0,1] (near=1, far=0, invalid=0)."""
import cv2
import numpy as np
from common import NEAR, FAR, BAND, IMG, crop_box


def preprocess_depth(depth_m: np.ndarray, out: int = IMG) -> np.ndarray:
    """depth_m: [H,W] float meters (0 = invalid). Returns [1,out,out] float32."""
    d = depth_m.astype(np.float32).copy()
    d[(d < NEAR) | (d > FAR)] = 0
    valid = d > 0
    if valid.sum() > 500:
        z0 = np.percentile(d[valid], 2)          # nearest surface ~ the subject
        valid &= d <= z0 + BAND                  # drop walls / far clutter
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        d *= m
    else:
        d[:] = 0
    x0, y0, s = crop_box(*d.shape)
    d = cv2.resize(d[y0:y0 + s, x0:x0 + s], (out, out), interpolation=cv2.INTER_NEAREST)
    x = np.where(d > 0, 1.0 - (d - NEAR) / (FAR - NEAR), 0.0).astype(np.float32)
    return x[None]


def hha_encode(depth_m, intr):
    """TODO (optional 3-channel input): horizontal disparity, height above ground,
    angle with gravity. Reference: Gupta et al., 'Learning Rich Features from RGB-D
    Images' (github.com/s-gupta/rcnn-depth). Needs ground-plane estimation; if you add
    it, change the first conv to 3 input channels in models.py."""
    raise NotImplementedError
