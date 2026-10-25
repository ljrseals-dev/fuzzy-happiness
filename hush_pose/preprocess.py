"""Depth preprocessing: range gating, subject-band background suppression,
morphology, center crop, resize, normalize to [0,1] (near=1, far=0, invalid=0)."""
import cv2
import numpy as np
from common import NEAR, FAR, BAND, IMG, crop_box


def _gate(depth_m: np.ndarray) -> np.ndarray:
    """Range gate + subject-band background suppression + morphology. Returns cleaned depth [H,W] (0 = invalid)."""
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
    return d


def _crop_resize(a: np.ndarray, out: int) -> np.ndarray:
    """Center-square crop + nearest-neighbour resize (works for [H,W] and [H,W,C])."""
    x0, y0, s = crop_box(*a.shape[:2])
    return cv2.resize(np.ascontiguousarray(a[y0:y0 + s, x0:x0 + s]), (out, out), interpolation=cv2.INTER_NEAREST)


def preprocess_depth(depth_m: np.ndarray, out: int = IMG) -> np.ndarray:
    """depth_m: [H,W] float meters (0 = invalid). Returns [1,out,out] float32."""
    d = _crop_resize(_gate(depth_m), out)
    x = np.where(d > 0, 1.0 - (d - NEAR) / (FAR - NEAR), 0.0).astype(np.float32)
    return x[None]


def hha_encode(depth_m: np.ndarray, intr, out: int = IMG, gravity=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Optional 3-channel HHA encoding (Gupta et al., 2014). Returns [3,out,out] float32 in [0,1], invalid = 0.

    ch0 horizontal disparity : 1/Z scaled between FAR and NEAR (near = 1)
    ch1 height above ground  : (lowest observed point along gravity) - point, clipped to [0, 2.5] m, /2.5
    ch2 angle with gravity   : angle between the (camera-facing) surface normal and 'up', /pi

    `gravity` is the 'down' direction in camera coordinates (default: camera upright, +y down). For a tilted
    sensor pass the IMU-derived vector. Ground height is the 98th percentile of the along-gravity coordinate of
    the subject points (a cheap stand-in for RANSAC ground-plane fitting).

    NOTE: the shipped models take 1 input channel. To use HHA, build the models with a 3-channel patch/stem
    conv, feed this tensor from data.py / pipeline.py, and re-export; the TensorRT runner and C++ estimator
    assume [1,1,256,256]."""
    d = _gate(depth_m)
    h, w = d.shape
    valid = d > 0
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    X = (u - intr.cx) * d / intr.fx
    Y = (v - intr.cy) * d / intr.fy
    P = np.stack([X, Y, d], -1)                                     # [H,W,3] camera-frame points
    g = np.asarray(gravity, np.float32); g /= np.linalg.norm(g)

    disp = np.zeros_like(d)
    disp[valid] = (1.0 / d[valid] - 1.0 / FAR) / (1.0 / NEAR - 1.0 / FAR)

    s = P @ g                                                       # coordinate along gravity (grows downward)
    height = np.zeros_like(d)
    if valid.any():
        ground = np.percentile(s[valid], 98)
        height[valid] = np.clip(ground - s[valid], 0.0, 2.5) / 2.5

    du = np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv)
    norm = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.maximum(norm, 1e-9)
    n[(n * P).sum(-1) > 0] *= -1                                    # orient toward the camera
    inner = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & (norm[..., 0] > 1e-9)
    ang = np.zeros_like(d)
    ang[inner] = np.arccos(np.clip((n[inner] * -g).sum(-1), -1.0, 1.0)) / np.pi

    hha = np.stack([disp, height, ang], 0).astype(np.float32)       # [3,H,W]
    return np.stack([_crop_resize(c, out) for c in hha], 0)
