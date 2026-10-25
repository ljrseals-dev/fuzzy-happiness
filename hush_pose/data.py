"""Datasets: unlabeled pretraining frames + labeled synthetic (with augmentation)."""
from __future__ import annotations
import glob
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from common import make_heatmaps, to_crop, IMG
from preprocess import preprocess_depth
from synth import sample_frame


def load_depth(path: str) -> np.ndarray:
    a = np.load(path) if path.endswith(".npy") else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    return a.astype(np.float32) / 1000.0 if a.dtype == np.uint16 else a.astype(np.float32)


class DepthPretrainDataset(Dataset):
    """Unlabeled depth: real files from `root` + on-the-fly synthetic frames."""

    def __init__(self, root: str | None = None, n_synth: int = 0, seed: int = 123):
        self.files = []
        if root:
            self.files = sorted(
                glob.glob(os.path.join(root, "**", "*.npy"), recursive=True) +
                glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
            )
        self.n_synth = n_synth
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files) + self.n_synth

    def __getitem__(self, i: int) -> torch.Tensor:
        if i < len(self.files):
            d = load_depth(self.files[i])
        else:
            d = sample_frame(np.random.default_rng(self.seed * 1_000_003 + i))[0]
        return torch.from_numpy(preprocess_depth(d))


def _augment(depth: np.ndarray, uv: np.ndarray, rng: np.random.Generator):
    """Depth-specific augmentation: rotation, scale, noise, dropout."""
    h, w = depth.shape
    # Random rotation ±15 deg
    angle = rng.uniform(-15.0, 15.0)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    depth = cv2.warpAffine(depth, M, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.ones((uv.shape[0], 1), np.float32)
    uv_h = np.hstack([uv, ones])
    uv = (M @ uv_h.T).T[:, :2]
    # Random scale 0.9-1.1 around center
    scale = rng.uniform(0.9, 1.1)
    M2 = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
    depth = cv2.warpAffine(depth, M2, (w, h), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    uv_h = np.hstack([uv, ones])
    uv = (M2 @ uv_h.T).T[:, :2]
    # Multiplicative depth noise
    noise = rng.normal(1.0, 0.02, depth.shape).astype(np.float32)
    depth = np.where(depth > 0, depth * noise, 0.0)
    # Random dropout
    drop = rng.random(depth.shape) < 0.01
    depth[drop] = 0.0
    return depth, uv


class SyntheticPoseDataset(Dataset):
    """Labeled synthetic: (image [1,256,256], heatmaps [17,64,64], joints_xy_256 [17,2])."""

    def __init__(self, n: int, seed: int = 0, augment: bool = True):
        self.n = n
        self.seed = seed
        self.augment = augment

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        depth, _, uv = sample_frame(rng)
        if self.augment:
            depth, uv = _augment(depth, uv, rng)
        j256 = to_crop(uv, *depth.shape)
        return (
            torch.from_numpy(preprocess_depth(depth)),
            torch.from_numpy(make_heatmaps(j256)),
            torch.from_numpy(j256.astype(np.float32)),
        )
