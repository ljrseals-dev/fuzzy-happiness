"""Export the student to ONNX (dynamic batch) and verify parity with onnxruntime."""
import argparse
import numpy as np
import torch
from models import StudentPose

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", default="student.onnx")
a = ap.parse_args()
m = StudentPose()
m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
m.eval()
x = torch.rand(1, 1, 256, 256)
torch.onnx.export(m, x, a.out, input_names=["depth"], output_names=["heatmaps"], opset_version=17,
                  dynamic_axes={"depth": {0: "n"}, "heatmaps": {0: "n"}})
import onnxruntime as ort                                   # parity check
y = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(None, {"depth": x.numpy()})[0]
print("max abs diff vs PyTorch:", np.abs(y - m(x).detach().numpy()).max())
