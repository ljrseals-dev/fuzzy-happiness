"""Build a TensorRT engine (TRT 8.6+). Equivalent CLI for FP16:
  trtexec --onnx=student.onnx --fp16 --saveEngine=student_fp16.plan \
    --minShapes=depth:1x1x256x256 --optShapes=depth:1x1x256x256 --maxShapes=depth:4x1x256x256"""
import argparse
import tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--fp16", action="store_true")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--calib-dir", default=None)
ap.add_argument("--max-batch", type=int, default=4)
a = ap.parse_args()

log = trt.Logger(trt.Logger.INFO)
b = trt.Builder(log)
net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
p = trt.OnnxParser(net, log)
if not p.parse(open(a.onnx, "rb").read()):
    raise SystemExit("\n".join(str(p.get_error(i)) for i in range(p.num_errors)))
cfg = b.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
if a.fp16 or a.int8:
    cfg.set_flag(trt.BuilderFlag.FP16)           # FP16 fallback for INT8-unfriendly layers
if a.int8:
    from calibrate_int8 import DepthCalibrator
    cfg.set_flag(trt.BuilderFlag.INT8)
    cfg.int8_calibrator = DepthCalibrator(a.calib_dir)
prof = b.create_optimization_profile()
prof.set_shape("depth", (1, 1, 256, 256), (1, 1, 256, 256), (a.max_batch, 1, 256, 256))
cfg.add_optimization_profile(prof)
blob = b.build_serialized_network(net, cfg)
if blob is None:
    raise SystemExit("engine build failed")
open(a.out, "wb").write(blob)
print("wrote", a.out, len(blob) / 1e6, "MB")
