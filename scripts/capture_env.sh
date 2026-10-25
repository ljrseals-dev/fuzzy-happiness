#!/usr/bin/env bash
# Thin wrapper: run capture_env.py then append dpkg package list for CUDA/TRT/NVIDIA.
set -euo pipefail
OUT="${1:-env_report.json}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 "$ROOT/scripts/capture_env.py" --out "$OUT"

# Append apt package snapshot if available
if command -v dpkg >/dev/null 2>&1; then
  python3 - <<PY
import json
path = "$OUT"
with open(path) as f:
    r = json.load(f)
import subprocess
try:
    out = subprocess.check_output(
        ["bash", "-c", "dpkg -l | grep -E 'cuda|tensorrt|nvidia' || true"],
        text=True, timeout=30,
    )
    r["dpkg_cuda_tensorrt_nvidia"] = out.strip().splitlines()
except Exception as e:
    r["dpkg_cuda_tensorrt_nvidia"] = {"error": str(e)}
with open(path, "w") as f:
    json.dump(r, f, indent=2)
print("appended dpkg snapshot to", path)
PY
fi
