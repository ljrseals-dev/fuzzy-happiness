"""Export student (or teacher) to ONNX with dynamic batch + ORT parity check."""
from __future__ import annotations
import argparse
import numpy as np
import torch
from models import StudentPose, TeacherPose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="student.onnx")
    ap.add_argument("--arch", choices=["student", "teacher"], default="student")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    cls = StudentPose if a.arch == "student" else TeacherPose
    m = cls()
    ck = torch.load(a.ckpt, map_location="cpu")
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    m.eval()

    x = torch.randn(1, 1, 256, 256)
    torch.onnx.export(
        m, x, a.out,
        input_names=["depth"],
        output_names=["heatmaps"],
        opset_version=a.opset,
        dynamic_axes={"depth": {0: "N"}, "heatmaps": {0: "N"}},
        do_constant_folding=True,
    )
    print("wrote", a.out)

    # Parity check with onnxruntime
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
        y_ort = sess.run(None, {"depth": x.numpy()})[0]
        y_pt = m(x).detach().numpy()
        diff = float(np.abs(y_ort - y_pt).max())
        print(f"ONNX parity max abs diff: {diff:.2e}")
        if diff >= 1e-4:
            print("WARNING: ONNX/PyTorch mismatch larger than expected")
        else:
            print("ONNX parity OK")
    except ImportError:
        print("onnxruntime not installed – skipped parity check")


if __name__ == "__main__":
    main()
