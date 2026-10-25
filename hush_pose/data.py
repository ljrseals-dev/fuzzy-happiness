"""Datasets: unlabeled pretraining frames (real .npy/.png + synthetic), labeled synthetic."""
import glob
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from common import make_heatmaps, to_crop
from preprocess import preprocess_depth
from synth import sample_frame


def load_depth(path):
    a = np.load(path) if path.endswith(".npy") else cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return a.astype(np.float32) / 1000.0 if a.dtype == np.uint16 else a.astype(np.float32)


class DepthPretrainDataset(Dataset):
    """Unlabeled depth frames: files from `root` (16-bit mm PNG or .npy meters/uint16 mm)
    followed by `n_synth` on-the-fly synthetic frames."""
    def __init__(self, root=None, n_synth=0, seed=123):
        self.files = sorted(glob.glob(os.path.join(root, "**", "*.npy"), recursive=True) +
                            glob.glob(os.path.join(root, "**", "*.png"), recursive=True)) if root else []
        self.n_synth, self.seed = n_synth, seed

    def __len__(self):
        return len(self.files) + self.n_synth

    def __getitem__(self, i):
        if i < len(self.files):
            d = load_depth(self.files[i])
        else:
            d = sample_frame(np.random.default_rng(self.seed * 1_000_003 + i))[0]
        return torch.from_numpy(preprocess_depth(d))


class SyntheticPoseDataset(Dataset):
    """Labeled: (image [1,256,256], heatmaps [17,64,64], joints_xy_256 [17,2])."""
    def __init__(self, n, seed=0):
        self.n, self.seed = n, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        depth, _, uv = sample_frame(rng)
        j256 = to_crop(uv, *depth.shape)
        return (torch.from_numpy(preprocess_depth(depth)),
                torch.from_numpy(make_heatmaps(j256)), torch.from_numpy(j256))
