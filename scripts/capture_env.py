#!/usr/bin/env python3
"""Write env_report.json describing the local ML stack.

Usage:
  python scripts/capture_env.py --out /tmp/env.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone


def _run(cmd, timeout=30):
    try:
        out = subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.STDOUT)
        return out.strip()
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="env_report.json")
    a = ap.parse_args()

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
    }

    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "list", "--format=freeze"],
            text=True, timeout=60,
        )
        report["pip_list_hash"] = hashlib.sha256(out.encode()).hexdigest()[:16]
        report["pip_package_count"] = len(out.strip().splitlines())
    except Exception as e:
        report["pip_list_error"] = str(e)

    try:
        import torch
        t = {
            "version": torch.__version__,
            "cuda_built": getattr(torch.version, "cuda", None),
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            t["device_name"] = torch.cuda.get_device_name(0)
            free, total = torch.cuda.mem_get_info(0)
            t["vram_total_mb"] = int(total // (1024 * 1024))
            t["vram_free_mb"] = int(free // (1024 * 1024))
        report["torch"] = t
    except Exception as e:
        report["torch"] = {"error": str(e)}

    try:
        import onnxruntime as ort
        report["onnxruntime"] = {
            "version": ort.__version__,
            "providers": ort.get_available_providers(),
        }
    except Exception as e:
        report["onnxruntime"] = {"error": str(e)}

    try:
        import tensorrt as trt
        report["tensorrt"] = {"version": trt.__version__}
    except Exception as e:
        report["tensorrt"] = {"error": str(e)}

    try:
        import cv2
        report["opencv"] = cv2.__version__
    except Exception as e:
        report["opencv"] = {"error": str(e)}

    report["nvidia_smi"] = _run(["nvidia-smi", "-q"])
    report["nvcc"] = _run(["nvcc", "--version"])
    report["trtexec"] = _run(["trtexec", "--version"])

    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {a.out}")
    # Print a one-line summary
    cuda = report.get("torch", {})
    if isinstance(cuda, dict) and cuda.get("cuda_available"):
        print(f"GPU: {cuda.get('device_name')}  VRAM={cuda.get('vram_total_mb')} MB")
    else:
        print("GPU: NOT AVAILABLE (CPU-only environment)")


if __name__ == "__main__":
    main()
