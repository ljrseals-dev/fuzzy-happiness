# Hush Kinect Pose

Open-source, GPU-accelerated pose estimation from Kinect depth (Kinect v2 via libfreenect2, Azure Kinect via pyk4a).
Masked-autoencoder pretraining on unlabeled depth -> ViT teacher -> distilled separable-conv student -> ONNX -> TensorRT.
Target: Ubuntu 22.04, CUDA 12.x, TensorRT 8.6+, RTX 3060 or Jetson Orin NX. License: MIT.

> **Status: untested scaffold.** Written without access to a GPU or sensor. Expect to fix small issues.
> Performance numbers in the design notes are estimates, not measurements.
> Kinect depth is 30 FPS max, so live end-to-end output is sensor-limited to 30 FPS; inference alone can run much faster.
> The built-in synthetic stick-figure generator is a fallback. For real accuracy, render SMPL/mocap meshes
> (same `(depth_m, joints_xyz, joints_uv)` contract as `synth.sample_frame`) and label a few hundred real frames for evaluation.

## Layout
```
hush_pose/   Python pipeline (run scripts from inside this directory; imports are flat)
cpp/         HushKinectPoseEstimator C++ API + CMake (STUB, see TODOs)
```

## Run order
```bash
cd hush_pose && pip install -r requirements.txt
python pretrain_vit.py --data /data/frames --synth 50000 --out ckpt/mae
python finetune_pose.py --mode teacher  --pretrained ckpt/mae/encoder_last.pt --out ckpt/teacher.pt
python finetune_pose.py --mode student  --teacher ckpt/teacher.pt --out ckpt/student.pt
python finetune_pose.py --mode baseline --out ckpt/baseline.pt
python export_onnx.py --ckpt ckpt/student.pt --out student.onnx
python build_tensorrt_engine.py --onnx student.onnx --out student_fp16.plan --fp16
python benchmark.py --baseline ckpt/baseline.pt --teacher ckpt/teacher.pt --student ckpt/student.pt
python pipeline.py --source synth --engine student_fp16.plan     # then --source v2 | azure
```
`--data` takes a folder of 16-bit mm PNGs or `.npy` depth frames (omit it to pretrain on synthetic only).

## Tensors
Input `[N,1,256,256]` fp32 (near=1, far=0, invalid=0) -> heatmaps `[N,17,64,64]`. 17 joints, see `common.JOINT_NAMES`.
Skeleton output: JSON with per-joint `xyz` (meters, camera frame) and `conf`.

## Checklist
1. Eyeball `synth` + `preprocess_depth` output. 2. MAE smoke test (loss falls). 3. Teacher > baseline on PCK.
4. Student (`--alpha 0.5`) > baseline. 5. ONNX parity, then TensorRT FP16 vs PyTorch PCK.
6. INT8 only with real calibration frames; keep FP16 if PCK drops. 7. Measure on hand-labeled real frames.
8. Swap in SMPL/mocap rendering. 9. Profile live pipeline; move preprocess to GPU if it dominates.
10. Implement the C++ estimator (cpp/src TODOs).
