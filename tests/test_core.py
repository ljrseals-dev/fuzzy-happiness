"""Core tests. Run:  python tests/test_core.py   (or: pytest tests/)
numpy/opencv/scipy only; PyTorch tests are skipped automatically when torch is missing."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "hush_pose"))

from common import KINECT_V2, NUM_JOINTS, HM, IMG, make_heatmaps, to_crop, from_crop  # noqa: E402
from postprocess import decode_skeleton, nms_peaks  # noqa: E402
from preprocess import preprocess_depth, hha_encode  # noqa: E402
from smoothing import OneEuroFilter, SkeletonSmoother  # noqa: E402
from synth import sample_frame  # noqa: E402


def test_preprocess_shape_and_range():
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = preprocess_depth(sample_frame(rng)[0])
        assert x.shape == (1, IMG, IMG) and x.dtype == np.float32
        assert 0.0 <= x.min() and x.max() <= 1.0 and (x > 0).mean() > 0.01


def test_crop_roundtrip():
    uv = np.array([[100.0, 50.0], [400.0, 300.0]], np.float32)
    assert np.allclose(from_crop(to_crop(uv, 424, 512), 424, 512), uv, atol=1e-3)


def test_heatmap_peak_matches_label():
    j = np.tile(np.array([[100.3, 77.8]], np.float32), (NUM_JOINTS, 1))
    pk = nms_peaks(make_heatmaps(j))[0][0]
    x256, y256 = (pk[0] + .5) * IMG / HM - .5, (pk[1] + .5) * IMG / HM - .5
    assert abs(x256 - 100.3) < 2.5 and abs(y256 - 77.8) < 2.5


def test_decode_roundtrip_3d():
    rng = np.random.default_rng(3)
    errs, zerr = [], []
    for _ in range(20):
        d, J, uv = sample_frame(rng)
        sk = decode_skeleton(make_heatmaps(to_crop(uv, *d.shape)), d, KINECT_V2)
        for k, jt in enumerate(sk["joints"]):
            if jt["xyz"]:
                x, y, z = jt["xyz"]
                u, v = x * KINECT_V2.fx / z + KINECT_V2.cx, y * KINECT_V2.fy / z + KINECT_V2.cy
                errs.append(np.hypot(u - uv[k, 0], v - uv[k, 1]))
                zerr.append(abs(z - J[k, 2]))
    assert len(errs) > 200
    assert np.median(errs) < 3.0 and np.median(zerr) < 0.08


def test_hha():
    h, w = 424, 512
    plane = np.full((h, w), 2.0, np.float32)            # fronto-parallel wall at 2 m
    hha = hha_encode(plane, KINECT_V2)
    assert hha.shape == (3, IMG, IMG) and hha.dtype == np.float32
    assert 0.0 <= hha.min() and hha.max() <= 1.0
    c = hha[:, IMG // 2, IMG // 2]
    expect_disp = (1 / 2.0 - 1 / 4.5) / (1 / 0.5 - 1 / 4.5)
    assert abs(c[0] - expect_disp) < 1e-3                # disparity channel
    assert abs(c[2] - 0.5) < 0.02                        # wall normal is 90 degrees from "up"
    assert hha[1, 10, IMG // 2] > hha[1, IMG - 10, IMG // 2]   # higher in the image = higher above ground


def test_one_euro_reduces_jitter_and_tracks_motion():
    rng = np.random.default_rng(0)
    f = OneEuroFilter()
    t = np.arange(300) / 30.0
    still = np.array([[1.0, 0.5, 2.0]]) + rng.normal(0, 0.01, (300, 3))
    out = np.array([f(x, ti) for x, ti in zip(still, t)])
    assert out[30:].std(0).mean() < 0.5 * still[30:].std(0).mean()
    f = OneEuroFilter()
    ramp = np.stack([np.linspace(0, 1.5, 300), np.zeros(300), np.full(300, 2.0)], 1)   # 5 m/s sweep
    out = np.array([f(x, ti) for x, ti in zip(ramp, t)])
    assert np.abs(out[100:] - ramp[100:]).max() < 0.1    # little lag at speed


def test_smoother_resets_lost_joint():
    s = SkeletonSmoother()
    mk = lambda xyz: {"joints": [{"name": "j", "xyz": xyz, "conf": 1.0}] + [{"name": "k", "xyz": None, "conf": 0}] * 16}
    s(mk([0.0, 0.0, 2.0]), 0.0)
    s(mk(None), 0.03)
    out = s(mk([1.0, 0.0, 2.0]), 0.06)
    assert out["joints"][0]["xyz"] == [1.0, 0.0, 2.0]    # no stale blending after dropout


# ---- torch-dependent (skipped if torch is missing) ----
def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def test_model_shapes():
    torch = _torch()
    if torch is None:
        print("  (skipped: torch missing)")
        return
    from models import MAE, StudentPose, TeacherPose
    x = torch.rand(2, 1, 256, 256)
    assert StudentPose().eval()(x).shape == (2, 17, 64, 64)
    assert TeacherPose().eval()(x).shape == (2, 17, 64, 64)
    assert torch.isfinite(MAE()(x))


def test_augmentation_keeps_labels_on_body():
    torch = _torch()
    if torch is None:
        print("  (skipped: torch missing)")
        return
    from data import SyntheticPoseDataset
    ds = SyntheticPoseDataset(20, seed=2, augment=True)
    hits = tot = 0
    for i in range(len(ds)):
        x, hm, j = ds[i]
        x, j = x.numpy()[0], j.numpy()
        for (u, v) in j:
            if 2 <= u < IMG - 2 and 2 <= v < IMG - 2:
                tot += 1
                hits += (x[int(v) - 2:int(v) + 3, int(u) - 2:int(u) + 3] > 0).any()
    assert tot > 100 and hits / tot > 0.9, (hits, tot)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)):
        try:
            fn()
            print("PASS", name)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print("FAIL", name, "->", repr(e))
    sys.exit(1 if failed else 0)
