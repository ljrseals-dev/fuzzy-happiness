#!/usr/bin/env bash
# Quick CPU smoke tests – no GPU or Kinect required.
set -euo pipefail
cd "$(dirname "$0")/hush_pose"

echo "=== 0. Core unit tests ==="
python ../tests/test_core.py

echo "=== 1. MAE pretrain smoke ==="
python pretrain_vit.py --synth 64 --bs 4 --epochs 1 --smoke --device cpu --out /tmp/hush_mae_smoke

echo "=== 2. Fine-tune baseline smoke ==="
python finetune_pose.py --mode baseline --n-train 64 --epochs 1 --bs 4 --smoke --device cpu --out /tmp/hush_base_smoke.pt

echo "=== 3. Synthetic data generation ==="
python generate_synthetic_data.py --out /tmp/hush_synth --n 8

echo "=== 4. Visualization ==="
python visualize_skeleton.py --depth /tmp/hush_synth/000000_depth.npy --xyz /tmp/hush_synth/000000_xyz.npy --out /tmp/hush_vis.png

echo "=== 5. ONNX export (optional – needs onnx) ==="
if python -c "import onnx" 2>/dev/null; then
  python export_onnx.py --ckpt /tmp/hush_base_smoke.pt --out /tmp/hush_student.onnx
  echo "ONNX export OK"
else
  echo "onnx not installed – skipped"
fi

echo ""
echo "All smoke tests passed."
