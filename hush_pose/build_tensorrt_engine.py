"""Build a TensorRT engine (TRT 8.6+).

CLI equivalent for FP16:
  trtexec --onnx=student.onnx --fp16 --saveEngine=student_fp16.plan \
    --minShapes=depth:1x1x256x256 --optShapes=depth:1x1x256x256 \
    --maxShapes=depth:4x1x256x256
"""
from __future__ import annotations
import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--max-batch", type=int, default=4)
    ap.add_argument("--verify", action="store_true",
                    help="Compare TRT vs ONNXRuntime (max abs < 1e-3 for FP16)")
    a = ap.parse_args()

    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: tensorrt Python package not found. Install TensorRT 8.6+.", file=sys.stderr)
        sys.exit(1)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(a.onnx, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i), file=sys.stderr)
            sys.exit(1)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB
    if a.fp16 or a.int8:
        config.set_flag(trt.BuilderFlag.FP16)
    if a.int8:
        from calibrate_int8 import DepthCalibrator
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = DepthCalibrator(a.calib_dir)

    profile = builder.create_optimization_profile()
    profile.set_shape("depth",
                      (1, 1, 256, 256),
                      (1, 1, 256, 256),
                      (a.max_batch, 1, 256, 256))
    config.add_optimization_profile(profile)

    print("Building engine (this may take a minute)...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("ERROR: engine build failed", file=sys.stderr)
        sys.exit(1)
    with open(a.out, "wb") as f:
        f.write(serialized)
    print(f"wrote {a.out}  ({len(serialized)/1e6:.1f} MB)")

    if a.verify:
        import numpy as np
        try:
            import onnxruntime as ort
            from trt_runner import TRTRunner
        except ImportError as e:
            print(f"verify skipped: {e}")
            return
        x = np.random.randn(1, 1, 256, 256).astype(np.float32)
        y_ort = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"]).run(
            None, {"depth": x})[0]
        runner = TRTRunner(a.out)
        y_trt = runner.infer(x)
        runner.close()
        diff = float(np.abs(y_ort - y_trt).max())
        print(f"TRT vs ONNX max abs diff: {diff:.2e}")
        if diff >= 1e-3:
            print("WARNING: difference exceeds 1e-3 (expected for FP16)")
        else:
            print("TRT verification OK")


if __name__ == "__main__":
    main()
