# Hardware handoff — STOPPED at Phase A

**Date:** 2026-10-25  
**Reason:** The execution environment has no GPU, no CUDA driver, no TensorRT, no Kinect.

## What was checked (Phase A)

| Check | Result |
|-------|--------|
| `torch.cuda.is_available()` | **false** |
| `nvidia-smi` | not found |
| `nvcc` | not found |
| `tensorrt` Python module | not installed |
| `trtexec` | not found |
| OpenCV (`cv2`) | not installed in this env |
| Kinect / libfreenect2 / pyk4a | not present |

Full dump: `docs/ENV_BASELINE.md` (CPU-only; replace after a GPU run).

## What is already done (no GPU needed)

- Python training/export path: CPU smoke tests (`./smoke_test.sh`, `tests/test_core.py`)
- C++ postprocess: parity-verified in `HUSH_MOCK_TRT` mode (Gate 5 can run on CPU)
- SMPL renderer, visualizer, One-Euro smoothing, docs

## What the GPU+Kinect operator must do

Follow `docs/GPU_VALIDATION.md` gates **1 → 6 in order**. Do not skip.

1. **Phase A on the GPU box**
   ```bash
   ./scripts/capture_env.sh /tmp/env_gpu.json
   # Replace docs/ENV_BASELINE.md with the GPU report
   ```

2. **Phase B — Six gates**  
   Exact commands are in `docs/GPU_VALIDATION.md`.  
   Record measured numbers only. On failure: write `docs/GATE_FAILURE_<N>.md` and stop.

3. **Phase C — Kinect capture** (after Gate 6 passes)  
   Use `scripts/capture_session.py` (scaffold provided) + protocol in `docs/KINECT_CAPTURE.md`.

4. **Phases D–F** — INT8, labeling, README fill-in — only after real frames exist.

## Do not

- Paste estimates into README performance tables
- Commit `.plan` / `.onnx` / capture directories into the repo
- Proceed past a failing gate without a failure doc

## Unblock requirements

| Resource | Minimum |
|----------|---------|
| GPU | RTX 3060 12 GB (or better), CUDA 12.x driver |
| TensorRT | 8.6+ (10.x preferred; C++ dual-path supports both) |
| Python | torch with CUDA, onnxruntime-gpu, tensorrt, pycuda, opencv-python |
| Kinect | Xbox One Kinect v2 (libfreenect2) **or** Azure Kinect DK (pyk4a) |
| Disk | ~5 GB for engines + capture session |
