# Hush Kinect Pose

Open-source, GPU-accelerated 2D/3D pose estimation from Kinect depth  
(Kinect v2 via libfreenect2, Azure Kinect via pyk4a).

Pipeline: **MAE pretrain on unlabeled depth → ViT teacher → distilled separable-conv student → ONNX → TensorRT FP16**.

Target platform: Ubuntu 22.04, CUDA 12.x, TensorRT ≥ 8.6, RTX 3060 or Jetson Orin NX.  
License: MIT.

> **Status**
>
> - **Python training / export**: runnable today; validated on CPU smoke tests (`./smoke_test.sh`, `tests/test_core.py`).
> - **C++ path**: runnable in `HUSH_MOCK_TRT` mode on CPU (parity-verified against Python `decode_skeleton`). Live TensorRT path requires GPU validation — see `docs/GPU_VALIDATION.md`.
> - **INT8**: experimental. Requires ≥500 real Kinect depth frames for calibration. Not production-ready until then.
> - **Synthetic data**: stick-figure is the development fallback. SMPL renderer (`--renderer smpl --require-smpl`) needs external model downloads (smplx + PyBullet + SMPL_MODEL_PATH).
## Layout

```
hush_pose/          Python training + inference package
  common.py         Joint definitions, intrinsics, heatmap helpers
  models.py         DepthViT, MAE, TeacherPose, StudentPose
  data.py           Datasets + depth augmentation
  synth.py          Stick-figure synthetic generator (always available)
  generate_synthetic_data.py
  preprocess.py     Depth normalization / subject band
  postprocess.py    NMS, back-projection, skeleton JSON
  smoothing.py      One-Euro temporal filter (pipeline.py --smooth)
  pretrain_vit.py   MAE pre-training
  finetune_pose.py  baseline / teacher / student
  export_onnx.py
  build_tensorrt_engine.py
  pipeline.py       4-stage threaded real-time pipeline
  sources.py        Freenect2 / Azure / Synthetic sources
  visualize_skeleton.py
  metrics.py / benchmark.py
cpp/                HushKinectPoseEstimator C++ API (TensorRT + OpenCV), uncompiled until you build it
tests/              test_core.py (numpy-only tests run anywhere; torch tests auto-skip)
INTEGRATION.md      How to embed the estimator into Hush
```

## Quick start (CPU smoke tests – no GPU required)

```bash
cd hush_pose
python -m venv .venv && source .venv/bin/activate
pip install torch numpy opencv-python scipy onnx onnxruntime   # CPU torch is fine

# 1. MAE smoke
python pretrain_vit.py --synth 128 --bs 8 --epochs 1 --smoke --device cpu --out /tmp/mae_smoke

# 2. Fine-tune smoke
python finetune_pose.py --mode baseline --n-train 128 --epochs 1 --bs 8 --smoke --device cpu --out /tmp/base.pt

# 3. Synthetic data
python generate_synthetic_data.py --out /tmp/synth --n 32
python visualize_skeleton.py --depth /tmp/synth/000000_depth.npy --xyz /tmp/synth/000000_xyz.npy --meta /tmp/synth/meta.json --out /tmp/vis.png
```

## Full training order (GPU recommended)

```bash
cd hush_pose
pip install -r requirements.txt

# Pre-train
python pretrain_vit.py --synth 50000 --epochs 50 --bs 64 --out ckpt/mae

# Fine-tune three models
python finetune_pose.py --mode teacher  --pretrained ckpt/mae/encoder_last.pt --out ckpt/teacher.pt
python finetune_pose.py --mode student  --teacher ckpt/teacher.pt --alpha 0.5 --out ckpt/student.pt
python finetune_pose.py --mode baseline --out ckpt/baseline.pt

# Export + TensorRT
python export_onnx.py --ckpt ckpt/student.pt --out student.onnx
python build_tensorrt_engine.py --onnx student.onnx --out student_fp16.plan --fp16 --verify

# Benchmark
python benchmark.py --baseline ckpt/baseline.pt --teacher ckpt/teacher.pt --student ckpt/student.pt

# Real-time pipeline (synthetic source)
python pipeline.py --source synth --engine student_fp16.plan --seconds 20
```

## Tensors

- Input: `[N,1,256,256]` float32, near=1, far=0, invalid=0  
- Output: heatmaps `[N,17,64,64]`  
- 17 joints defined in `common.JOINT_NAMES`  
- Skeleton JSON: `{"frame": id, "joints": [{"name", "xyz", "conf"}, ...], "bones": [...]}`

## Performance

No numbers have been measured yet. Run `benchmark.py` (PyTorch) and `pipeline.py` / `trtexec` (TensorRT) and record them here.
Rough expectations only: the student should be a few ms per frame in FP16 on an RTX 3060, the ViT-S teacher several times slower, and live end-to-end output is capped at 30 FPS by the Kinect.

| Config | PCK@0.1 (synth) | Latency | FPS |
|---|---|---|---|
| baseline student | – | – | – |
| MAE teacher | – | – | – |
| distilled student | – | – | – |

## C++ API

See `cpp/` and `INTEGRATION.md`. Build:

```bash
cd cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

## Validation checklist

0. `python tests/test_core.py` passes.  
1. Smoke tests above complete without error.  
2. MAE loss decreases.  
3. Teacher PCK > baseline on held-out synthetic.  
4. Student close to teacher, higher FPS.  
5. ONNX parity and TensorRT `--verify` pass.  
6. Pipeline reports ≥30 FPS on synthetic source.  
7. C++ example builds and returns a skeleton.  

## Limitations & next steps

- Stick-figure synthetic data is a development fallback. Replace with SMPL/AMASS mesh rendering for real accuracy.  
- INT8 requires real calibration frames; stay on FP16 until then.  
- Single person only. Multi-person needs a bottom-up model with Part Affinity Fields, which is not trained here (`nms_peaks(topk>1)` exposes candidates).  
- HHA encoding exists (`preprocess.hha_encode`) but the shipped models, TensorRT runner and C++ estimator take 1 channel; using HHA means widening the first conv and re-exporting.  
- Label a few hundred real Kinect frames (OpenPose on RGB + manual correction) for final evaluation.

## License

MIT – see LICENSE.
