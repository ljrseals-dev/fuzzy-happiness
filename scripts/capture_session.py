#!/usr/bin/env python3
"""Kinect capture session → capture_YYYYMMDD_HHMMSS/{calib,unlabeled,posed}/

Detects Freenect2 or Azure Kinect at runtime. Does not run without a device.

Usage:
  python scripts/capture_session.py --mode all --duration 180 --out-root /data/captures
  python scripts/capture_session.py --mode calib --duration 60
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hush_pose"))


def open_source():
    """Return (source, backend_name) or raise with a clear message."""
    errors = []
    try:
        from sources import FreenectV2Source
        src = FreenectV2Source()
        return src, "freenect2"
    except Exception as e:
        errors.append(f"freenect2: {e}")
    try:
        from sources import AzureKinectSource
        src = AzureKinectSource()
        return src, "azure"
    except Exception as e:
        errors.append(f"azure: {e}")
    raise RuntimeError(
        "No Kinect backend available.\n" + "\n".join(errors) +
        "\nInstall pylibfreenect2 (Kinect v2) or pyk4a (Azure Kinect) and connect a device."
    )


def capture_mode(src, out_dir: Path, duration_s: float, need_rgb: bool, log):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n = 0
    drops = 0
    while time.time() - t0 < duration_s:
        try:
            depth = src.read()
        except Exception as e:
            log.write(f"read error: {e}\n")
            drops += 1
            time.sleep(0.01)
            continue
        if depth is None:
            drops += 1
            continue
        import numpy as np
        # Store as float32 meters (pipeline contract). Original mm can be recovered *1000.
        np.save(out_dir / f"{n:06d}_depth.npy", depth.astype(np.float32))
        n += 1
        if n % 30 == 0:
            print(f"  {out_dir.name}: {n} frames", flush=True)
    log.write(f"{out_dir.name}: frames={n} drops={drops} duration={time.time()-t0:.1f}s\n")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["calib", "unlabeled", "posed", "all"], default="all")
    ap.add_argument("--duration", type=float, default=60.0, help="seconds per mode")
    ap.add_argument("--out-root", default=".")
    a = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(a.out_root) / f"capture_{stamp}"
    root.mkdir(parents=True, exist_ok=True)
    log = open(root / "session.log", "w")

    print("Opening Kinect…")
    src, backend = open_source()
    print(f"backend={backend}")

    modes = ["calib", "unlabeled", "posed"] if a.mode == "all" else [a.mode]
    counts = {}
    for m in modes:
        print(f"=== {m} ({a.duration:.0f}s) ===")
        counts[m] = capture_mode(src, root / m, a.duration, need_rgb=(m != "unlabeled"), log=log)

    # meta
    intr = getattr(src, "intr", None)
    meta = {
        "backend": backend,
        "timestamp": stamp,
        "counts": counts,
        "intrinsics": {
            "fx": getattr(intr, "fx", None),
            "fy": getattr(intr, "fy", None),
            "cx": getattr(intr, "cx", None),
            "cy": getattr(intr, "cy", None),
            "width": getattr(intr, "width", None),
            "height": getattr(intr, "height", None),
        } if intr else None,
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    (root / "README.txt").write_text(
        f"Capture session {stamp}\nbackend={backend}\ncounts={counts}\n"
        "See docs/KINECT_CAPTURE.md for protocol.\n"
    )
    src.close()
    log.close()
    print(f"Done → {root}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
