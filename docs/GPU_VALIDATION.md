# GPU Validation Runbook

Follow these six gates in order on a machine with CUDA + TensorRT.
Record the numbers in the results table at the bottom and paste them into the README “Expected ranges” section.

Prerequisites on the GPU box:
```bash
# From repo root
cd hush_pose
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# plus: tensorrt, pycuda matching your CUDA / TRT install
```

---

## Gate 1 — Python training smoke

```bash
cd hush_pose
python pretrain_vit.py --synth 256 --bs 16 --epochs 1 --smoke --out /tmp/mae_g1
python finetune_pose.py --mode baseline --n-train 256 --epochs 1 --bs 16 --smoke --out /tmp/base_g1.pt
```

**Pass criterion:** both commands exit 0; MAE loss decreases across iterations; fine-tune prints a PCK number.

**Record:** wall-clock time, peak GPU VRAM (`nvidia-smi`), torch / CUDA versions.

---

## Gate 2 — ONNX parity

```bash
python export_onnx.py --ckpt /tmp/base_g1.pt --out /tmp/student_g2.onnx
# export_onnx.py prints “ONNX parity max abs diff: …”
```

**Pass criterion:** max abs diff < 1e-4.

**Record:** diff value, onnx / onnxruntime versions.

---

## Gate 3 — TensorRT parity

```bash
python build_tensorrt_engine.py --onnx /tmp/student_g2.onnx --out /tmp/student_fp16.plan --fp16 --verify
```

**Pass criterion:** “TRT verification OK” (max abs < 1e-3 for FP16).

**Record:** TRT version (`python -c "import tensorrt; print(tensorrt.__version__)"`), CUDA version, engine size (MB), build wall-clock.

---

## Gate 4 — Pipeline throughput

```bash
python pipeline.py --source synth --engine /tmp/student_fp16.plan --seconds 20
```

**Pass criterion:** reported end-to-end FPS ≥ 30 on an RTX 3060-class GPU (synth source is unlimited; live Kinect is sensor-capped at 30).

**Record:** mean / p95 latency per stage (capture, preprocess, inference, postprocess), overall FPS, GPU model.

---

## Gate 5 — C++ parity (CPU, no GPU needed)

```bash
# On any machine with OpenCV + CMake:
cd cpp && mkdir -p build && cd build
cmake .. -DHUSH_MOCK_TRT=ON -DCMAKE_BUILD_TYPE=Release
make -j parity_test

# Generate fixtures (needs numpy/opencv/scipy only):
cd ../../hush_pose
python ../cpp/tests/dump_reference.py --out /tmp/fix1_fixtures --n 5

# Run
../cpp/build/parity_test --fixtures /tmp/fix1_fixtures
```

**Pass criterion:** “All cases passed (tolerance 0.01 m)”; every joint prints OK.

**Record:** compiler, OpenCV version, per-case max joint delta.

---

## Gate 6 — C++ live (GPU)

```bash
cd cpp/build
cmake .. -DHUSH_MOCK_TRT=OFF -DCMAKE_BUILD_TYPE=Release   # needs TensorRT + CUDA
make -j hush_pose_example
./hush_pose_example /tmp/student_fp16.plan
# Or wire processFrame into your Kinect capture loop — see INTEGRATION.md
```

**Pass criterion:** example prints a non-empty skeleton; on a real depth frame, joints land on the subject.

**Record:** TRT version, CUDA version, GPU model, processFrame wall-clock (ms).

---

## Results table (fill in and paste back)

| Gate | Pass? | Key numbers |
|------|-------|-------------|
| 1 Training smoke |  | time=, VRAM=, torch=, CUDA= |
| 2 ONNX parity |  | max_abs= |
| 3 TRT parity |  | TRT=, engine_MB=, build_s= |
| 4 Pipeline FPS |  | FPS=, p95_infer_ms=, GPU= |
| 5 C++ parity |  | max_delta_m= |
| 6 C++ live |  | processFrame_ms= |
